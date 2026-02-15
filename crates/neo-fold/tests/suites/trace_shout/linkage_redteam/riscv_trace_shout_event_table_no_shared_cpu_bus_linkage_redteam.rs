#![allow(non_snake_case)]

#[path = "../../../common/riscv_shout_event_table_packed.rs"]
mod event_table_packed;

use std::collections::BTreeMap;
use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::F;
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32ShoutEventRow, Rv32ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn flip_time_bit0_for_all_events(
    z: &mut [F],
    m: usize,
    m_in: usize,
    steps: usize,
    ell_addr: usize,
) -> Result<(), String> {
    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        steps,
        core::iter::once((ell_addr, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    let cols = &bus.shout_cols[0].lanes[0];
    let time_bit0_col = cols.addr_bits.start;
    for j in 0..steps {
        let idx = bus.bus_cell(time_bit0_col, j);
        z[idx] = if z[idx] == F::ZERO { F::ONE } else { F::ZERO };
    }
    Ok(())
}

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_event_table_linkage_redteam() {
    // Minimal program; we tamper with an event-table time bit so the hash linkage fails.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 1,
            rs2: 2,
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

    let mut lut_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)> = Vec::new();
    let mut did_tamper = false;
    for (_shout_id, (opcode, rows)) in by_id.into_iter() {
        let steps = rows.len();
        let base_d =
            event_table_packed::rv32_packed_base_d(opcode).unwrap_or_else(|e| panic!("opcode {opcode:?}: {e}"));
        let d = ell_n + base_d;
        let ell_addr = d;

        let mut z =
            event_table_packed::build_shout_event_table_bus_z(ccs.m, layout.m_in, steps, ell_n, opcode, &rows, &x)
                .unwrap_or_else(|e| panic!("event-table z build failed (opcode={opcode:?}): {e}"));

        // Tamper with the time-bit prefix of the first instance only (keeps booleanity).
        if !did_tamper {
            flip_time_bit0_for_all_events(&mut z, ccs.m, layout.m_in, steps, ell_addr).expect("tamper time bit");
            did_tamper = true;
        }

        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        let c = l.commit(&Z);

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
    assert!(did_tamper, "expected to tamper at least one Shout instance");

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances,
        mem_instances: Vec::new(),
        _phantom: PhantomData,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-event-table-packed-redteam");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
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

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-event-table-packed-redteam");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect_err("verification should fail due to event-table hash linkage mismatch");

    // Keep the assertion stable but informative.
    let _ = err;
}
