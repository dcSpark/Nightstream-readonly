#![allow(non_snake_case)]

use std::collections::HashMap;
use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::F;
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables,
    PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_memory::riscv::trace::extract_twist_lanes_over_time;
use neo_memory::witness::{LutWitness, MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn write_u64_bits_lsb(dst_bits: &mut [F], x: u64) {
    for (i, b) in dst_bits.iter_mut().enumerate() {
        *b = if ((x >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

fn build_twist_only_bus_z(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[neo_memory::riscv::trace::TwistLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_twist_only_bus_z: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.len() != lanes {
        return Err(format!(
            "build_twist_only_bus_z: lane_data.len()={} != lanes={}",
            lane_data.len(),
            lanes
        ));
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::empty::<(usize, usize)>(),
        core::iter::once((ell_addr, lanes)),
    )?;
    if bus.twist_cols.len() != 1 || !bus.shout_cols.is_empty() {
        return Err("build_twist_only_bus_z: expected 1 twist instance and 0 shout instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let twist = &bus.twist_cols[0];
    for (lane_idx, cols) in twist.lanes.iter().enumerate() {
        let lane = &lane_data[lane_idx];
        if lane.has_read.len() != t || lane.has_write.len() != t {
            return Err("build_twist_only_bus_z: lane length mismatch".into());
        }
        for j in 0..t {
            let has_r = lane.has_read[j];
            let has_w = lane.has_write[j];

            z[bus.bus_cell(cols.has_read, j)] = if has_r { F::ONE } else { F::ZERO };
            z[bus.bus_cell(cols.has_write, j)] = if has_w { F::ONE } else { F::ZERO };

            z[bus.bus_cell(cols.rv, j)] = if has_r { F::from_u64(lane.rv[j]) } else { F::ZERO };
            z[bus.bus_cell(cols.wv, j)] = if has_w { F::from_u64(lane.wv[j]) } else { F::ZERO };
            z[bus.bus_cell(cols.inc, j)] = if has_w { lane.inc_at_write_addr[j] } else { F::ZERO };

            {
                // ra_bits / wa_bits
                let mut tmp = vec![F::ZERO; ell_addr];
                write_u64_bits_lsb(&mut tmp, lane.ra[j]);
                for (bit_idx, col_id) in cols.ra_bits.clone().enumerate() {
                    z[bus.bus_cell(col_id, j)] = tmp[bit_idx];
                }
                tmp.fill(F::ZERO);
                write_u64_bits_lsb(&mut tmp, lane.wa[j]);
                for (bit_idx, col_id) in cols.wa_bits.clone().enumerate() {
                    z[bus.bus_cell(col_id, j)] = tmp[bit_idx];
                }
            }
        }
    }

    Ok(z)
}

#[test]
#[ignore = "RV32 trace no-shared fallback is legacy-only after shared-bus decode/width lookup cutover"]
fn riscv_trace_no_shared_cpu_bus_linkage_rejects_tampered_prog_addr_bits() {
    // Program:
    // - ADDI x1, x0, 1
    // - SW x1, 0(x0)
    // - LW x2, 0(x0)
    // - HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    // Force padding so we have inactive rows after HALT.
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 5).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let (prog_layout, prog_init) = prog_rom_layout_and_init_words::<F>(PROG_ID, /*base=*/ 0, &program_bytes)
        .expect("prog_rom_layout_and_init_words");
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

    // Mem instances: PROG, REG (2 lanes), RAM.
    let prog_init_pairs: Vec<(u64, F)> = {
        let mut pairs: Vec<(u64, F)> = prog_init
            .into_iter()
            .filter_map(|((mem_id, addr), v)| (mem_id == PROG_ID.0 && v != F::ZERO).then_some((addr, v)))
            .collect();
        pairs.sort_by_key(|(addr, _)| *addr);
        pairs
    };
    let prog_mem_init = if prog_init_pairs.is_empty() {
        MemInit::Zero
    } else {
        MemInit::Sparse(prog_init_pairs)
    };

    let t = exec.rows.len();
    let ram_d = 2usize; // k=4, address bits=2
    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    let twist_lanes = extract_twist_lanes_over_time(&exec, &init_regs, &init_ram, /*ram_ell_addr=*/ ram_d)
        .expect("extract twist lanes");

    // PROG (baseline)
    let prog_mem_inst_base = MemInstance::<Cmt, F> {
        mem_id: PROG_ID.0,
        comms: Vec::new(), // filled after commit
        k: prog_layout.k,
        d: prog_layout.d,
        n_side: prog_layout.n_side,
        steps: t,
        lanes: 1,
        ell: 1,
        init: prog_mem_init,
    };
    let prog_z_base = build_twist_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ prog_mem_inst_base.d * prog_mem_inst_base.ell,
        /*lanes=*/ 1,
        &[twist_lanes.prog.clone()],
        &x,
    )
    .expect("prog z base");

    // Tamper a PROG ra_bit on a padding row: pick the last row (should be inactive).
    let tamper_row = t - 1;
    assert!(!exec.rows[tamper_row].active, "expected padding row at t-1");
    let ell_addr_prog = prog_mem_inst_base.d * prog_mem_inst_base.ell;
    let bus_prog = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        ccs.m,
        layout.m_in,
        t,
        core::iter::empty::<(usize, usize)>(),
        core::iter::once((ell_addr_prog, 1usize)),
    )
    .expect("prog bus");
    let prog_lane_cols = &bus_prog.twist_cols[0].lanes[0];
    let first_ra_bit_col_id = prog_lane_cols
        .ra_bits
        .clone()
        .next()
        .expect("ra_bits non-empty");
    let tamper_idx = bus_prog.bus_cell(first_ra_bit_col_id, tamper_row);

    let mut prog_z_bad = prog_z_base.clone();
    prog_z_bad[tamper_idx] = F::ONE;

    let prog_Z_base = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &prog_z_base);
    let prog_c_base = l.commit(&prog_Z_base);
    let prog_mem_inst_base = MemInstance::<Cmt, F> {
        comms: vec![prog_c_base],
        ..prog_mem_inst_base
    };
    let prog_mem_wit_base = MemWitness {
        mats: vec![prog_Z_base],
    };

    let prog_Z_bad = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &prog_z_bad);
    let prog_c_bad = l.commit(&prog_Z_bad);
    let prog_mem_inst_bad = MemInstance::<Cmt, F> {
        comms: vec![prog_c_bad],
        ..prog_mem_inst_base.clone()
    };
    let prog_mem_wit_bad = MemWitness { mats: vec![prog_Z_bad] };

    // REG
    let reg_mem_inst = MemInstance::<Cmt, F> {
        mem_id: REG_ID.0,
        comms: Vec::new(),
        k: 32,
        d: 5,
        n_side: 2,
        steps: t,
        lanes: 2,
        ell: 1,
        init: MemInit::Zero,
    };
    let reg_z = build_twist_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ reg_mem_inst.d * reg_mem_inst.ell,
        /*lanes=*/ 2,
        &[twist_lanes.reg_lane0.clone(), twist_lanes.reg_lane1.clone()],
        &x,
    )
    .expect("reg z");
    let reg_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &reg_z);
    let reg_c = l.commit(&reg_Z);
    let reg_mem_inst = MemInstance::<Cmt, F> {
        comms: vec![reg_c],
        ..reg_mem_inst
    };
    let reg_mem_wit = MemWitness { mats: vec![reg_Z] };

    // RAM
    let ram_mem_inst = MemInstance::<Cmt, F> {
        mem_id: RAM_ID.0,
        comms: Vec::new(),
        k: 1usize << ram_d,
        d: ram_d,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        init: MemInit::Zero,
    };
    let ram_z = build_twist_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ ram_mem_inst.d * ram_mem_inst.ell,
        /*lanes=*/ 1,
        &[twist_lanes.ram.clone()],
        &x,
    )
    .expect("ram z");
    let ram_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &ram_z);
    let ram_c = l.commit(&ram_Z);
    let ram_mem_inst = MemInstance::<Cmt, F> {
        comms: vec![ram_c],
        ..ram_mem_inst
    };
    let ram_mem_wit = MemWitness { mats: vec![ram_Z] };

    // Baseline: prove+verify ok.
    let empty_lut_wit: LutWitness<F> = LutWitness { mats: Vec::new() };
    let steps_witness_ok = vec![StepWitnessBundle {
        mcs: mcs.clone(),
        lut_instances: Vec::new(),
        mem_instances: vec![
            (prog_mem_inst_base, prog_mem_wit_base),
            (reg_mem_inst.clone(), reg_mem_wit.clone()),
            (ram_mem_inst.clone(), ram_mem_wit.clone()),
        ],
        _phantom: PhantomData,
    }];
    let steps_instance_ok: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> = steps_witness_ok
        .iter()
        .map(StepInstanceBundle::from)
        .collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-linkage");
    let proof_ok = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness_ok,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove ok");
    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-linkage");
    let _ = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance_ok,
        &[],
        &proof_ok,
        mixers,
    )
    .expect("verify ok");

    // Tampered PROG witness: should verify fail due to trace linkage.
    let steps_witness_bad = vec![StepWitnessBundle {
        mcs,
        lut_instances: Vec::new(),
        mem_instances: vec![
            (prog_mem_inst_bad, prog_mem_wit_bad),
            (reg_mem_inst, reg_mem_wit),
            (ram_mem_inst, ram_mem_wit),
        ],
        _phantom: PhantomData,
    }];
    let steps_instance_bad: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> = steps_witness_bad
        .iter()
        .map(StepInstanceBundle::from)
        .collect();
    let mut tr_prove_bad = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-linkage-bad");
    let proof_bad = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove_bad,
        &params,
        &ccs,
        &steps_witness_bad,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove bad (linkage checked by verifier)");
    let mut tr_verify_bad = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-linkage-bad");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify_bad,
        &params,
        &ccs,
        &steps_instance_bad,
        &[],
        &proof_bad,
        mixers,
    )
    .expect_err("verify must fail under PROG addr-bit tamper");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("trace linkage"),
        "expected trace linkage failure, got: {msg}"
    );

    let _ = empty_lut_wit;
}
