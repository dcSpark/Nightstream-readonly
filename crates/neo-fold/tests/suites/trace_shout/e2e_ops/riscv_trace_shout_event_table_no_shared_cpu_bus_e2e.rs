#![allow(non_snake_case)]

#[path = "../../../common/riscv_shout_event_table_packed.rs"]
mod event_table_packed;

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::suite::{default_mixers, setup_ajtai_committer};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::F;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32ShoutEventRow, Rv32ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_event_table_packed_prove_verify() {
    // Program:
    // - RV32I bitwise/shifts/compares (includes EQ branches).
    // - HALT
    let program = vec![
        // x1 = 0x8000_0001
        RiscvInstruction::Lui { rd: 1, imm: 0x80000 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 1,
            rs1: 1,
            imm: 1,
        },
        // x2 = 37 (shamt=5)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 37,
        },
        // Shifts.
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sra,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        // Bitwise.
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 6,
            rs1: 3,
            rs2: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::And,
            rd: 7,
            rs1: 6,
            rs2: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Xor,
            rd: 8,
            rs1: 6,
            rs2: 1,
        },
        // Sub + compares.
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sltu,
            rd: 11,
            rs1: 1,
            rs2: 2,
        },
        // Build x17 = x1 - 4096 to get nontrivial EQ/NEQ rows.
        // LUI x17, 1 => 4096; SUB x17, x1, x17 => x1 - 4096.
        RiscvInstruction::Lui { rd: 17, imm: 1 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 17,
            rs1: 1,
            rs2: 17,
        },
        // EQ/NEQ branches (imm=4 keeps control flow linear).
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 1,
            imm: 4,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 17,
            imm: 4,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 1,
            rs2: 17,
            imm: 4,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Params + committer.
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    // Main CPU trace witness commitment.
    let z_cpu: Vec<F> = x.iter().copied().chain(w.iter().copied()).collect();
    let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_cpu);
    let c_cpu = l.commit(&Z_cpu);
    let mcs = (
        McsInstance {
            c: c_cpu,
            x: x.clone(),
            m_in: layout.m_in,
        },
        McsWitness { w, Z: Z_cpu },
    );

    // Event table extraction.
    let event_table = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");
    assert!(!event_table.rows.is_empty(), "expected non-empty Shout event table");

    // Group by shout_id (stable) and build one event-table packed instance per opcode.
    let mut by_id: BTreeMap<u32, (RiscvOpcode, Vec<Rv32ShoutEventRow>)> = BTreeMap::new();
    for row in event_table.rows.iter() {
        let opcode = row
            .opcode
            .ok_or_else(|| format!("missing opcode for shout_id={}", row.shout_id))
            .unwrap();
        let entry = by_id
            .entry(row.shout_id)
            .or_insert_with(|| (opcode, Vec::new()));
        if entry.0 != opcode {
            panic!(
                "opcode mismatch for shout_id={}: {:?} vs {:?}",
                row.shout_id, entry.0, opcode
            );
        }
        entry.1.push(row.clone());
    }

    let ell_n = event_table_packed::ell_n_from_ccs_n(ccs.n);
    assert!(ell_n >= 3, "event-table packed requires ell_n>=3 (got ell_n={ell_n})");
    assert!(ell_n <= 64, "event-table packed requires ell_n<=64 (got ell_n={ell_n})");

    let tables = RiscvShoutTables::new(32);
    let expected: BTreeMap<u32, (RiscvOpcode, usize)> = [
        (RiscvOpcode::And, 1usize),
        (RiscvOpcode::Xor, 2),
        (RiscvOpcode::Or, 1),
        (RiscvOpcode::Add, 1),
        (RiscvOpcode::Sub, 2),
        (RiscvOpcode::Slt, 1),
        (RiscvOpcode::Sltu, 1),
        (RiscvOpcode::Sll, 1),
        (RiscvOpcode::Srl, 1),
        (RiscvOpcode::Sra, 1),
        (RiscvOpcode::Eq, 3),
    ]
    .into_iter()
    .map(|(op, count)| (tables.opcode_to_id(op).0, (op, count)))
    .collect();
    let got: BTreeMap<u32, (RiscvOpcode, usize)> = by_id
        .iter()
        .map(|(shout_id, (opcode, rows))| (*shout_id, (*opcode, rows.len())))
        .collect();
    assert_eq!(got, expected, "unexpected event-table opcode coverage");

    let mut lut_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)> = Vec::new();
    for (_shout_id, (opcode, rows)) in by_id.into_iter() {
        let steps = rows.len();
        let z = event_table_packed::build_shout_event_table_bus_z(ccs.m, layout.m_in, steps, ell_n, opcode, &rows, &x)
            .unwrap_or_else(|e| panic!("event-table z build failed (opcode={opcode:?}): {e}"));
        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        let c = l.commit(&Z);

        // `d = time_bits + base_d(opcode)`
        let base_d =
            event_table_packed::rv32_packed_base_d(opcode).unwrap_or_else(|e| panic!("opcode {opcode:?}: {e}"));
        let d = ell_n + base_d;

        let inst = LutInstance::<Cmt, F> {
            comms: vec![c],
            k: 0,
            d,
            n_side: 2,
            steps,
            lanes: 1,
            ell: 1,
            table_spec: Some(LutTableSpec::RiscvOpcodeEventTablePacked {
                opcode,
                xlen: 32,
                time_bits: ell_n,
            }),
            table: Vec::new(),
        };
        let wit = LutWitness { mats: vec![Z] };
        lut_instances.push((inst, wit));
    }

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances,
        mem_instances: Vec::new(),
        _phantom: PhantomData,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-event-table-packed");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove");

    assert!(
        !proof.steps[0].mem.shout_me_claims_time.is_empty(),
        "expected Shout ME(time) claims in no-shared-bus mode"
    );
    assert_eq!(
        proof.steps[0].mem.shout_me_claims_time.len(),
        steps_witness[0].lut_instances.len(),
        "expected 1 Shout ME(time) claim per Shout instance"
    );
    assert_eq!(
        proof.steps[0].shout_time_fold.len(),
        steps_witness[0].lut_instances.len(),
        "expected 1 shout_time_fold per Shout instance"
    );

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-event-table-packed");
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");
}
