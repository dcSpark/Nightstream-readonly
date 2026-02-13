#![allow(non_snake_case)]

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
    decode_program, encode_program, interleave_bits, uninterleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory,
    RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use neo_vm_trace::ShoutEvent;
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn build_shout_only_bus_z(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[neo_memory::riscv::trace::ShoutLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 34 {
        return Err(format!(
            "build_shout_only_bus_z: expected ell_addr=34 for packed MULHU (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_only_bus_z: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.len() != lanes {
        return Err(format!(
            "build_shout_only_bus_z: lane_data.len()={} != lanes={}",
            lane_data.len(),
            lanes
        ));
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::once((ell_addr, lanes)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_only_bus_z: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let shout = &bus.shout_cols[0];
    for (lane_idx, cols) in shout.lanes.iter().enumerate() {
        let lane = lane_data
            .get(lane_idx)
            .ok_or_else(|| format!("missing lane_data[{lane_idx}]"))?;
        for j in 0..t {
            let has_lookup = lane.has_lookup[j];
            z[bus.bus_cell(cols.has_lookup, j)] = if has_lookup { F::ONE } else { F::ZERO };

            if has_lookup {
                z[bus.bus_cell(cols.val, j)] = F::from_u64(lane.value[j]);
            }

            if has_lookup {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane.key[j] as u128);
                let lhs = lhs_u64 as u32;
                let rhs = rhs_u64 as u32;
                let wide = (lhs as u64) * (rhs as u64);
                let lo = (wide & 0xffff_ffff) as u32;

                // Packed-key layout (ell_addr=34): [lhs_u32, rhs_u32, lo_bits[0..32]].
                let mut packed = vec![F::ZERO; ell_addr];
                packed[0] = F::from_u64(lhs as u64);
                packed[1] = F::from_u64(rhs as u64);
                for bit in 0..32 {
                    packed[2 + bit] = if ((lo >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }

                for (idx, col_id) in cols.addr_bits.clone().enumerate() {
                    z[bus.bus_cell(col_id, j)] = packed[idx];
                }
            }
        }
    }

    Ok(z)
}

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_mulhu_prove_verify() {
    // Program:
    // - LUI x1, 16          (x1 = 65536)
    // - LUI x2, 16          (x2 = 65536)
    // - MULHU x3, x1, x2    (hi=1, lo=0)
    // - MULHU x4, x2, x1    (hi=1, lo=0)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 16 },
        RiscvInstruction::Lui { rd: 2, imm: 16 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 4,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let mut exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    // RV32 B1 does not currently emit MULHU Shout events. Inject one per MULHU instruction row so we can
    // exercise the packed-key proving path without the legacy `ell_addr=64` encoding.
    let mulhu_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Mulhu);
    let mut injected = 0usize;
    for row in exec.rows.iter_mut() {
        if !row.active {
            continue;
        }
        let Some(RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu, ..
        }) = row.decoded
        else {
            continue;
        };
        if !row.shout_events.is_empty() {
            continue;
        }
        let rs1 = row.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0) as u32;
        let rs2 = row.reg_read_lane1.as_ref().map(|io| io.value).unwrap_or(0) as u32;
        let wide = (rs1 as u64) * (rs2 as u64);
        let hi = (wide >> 32) as u32;
        let key = interleave_bits(rs1 as u64, rs2 as u64) as u64;
        row.shout_events.push(ShoutEvent {
            shout_id: mulhu_id,
            key,
            value: hi as u64,
        });
        injected += 1;
    }
    assert!(injected > 0, "expected to inject at least one MULHU Shout event");
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

    // Shout instance: MULHU table, 1 lane.
    let t = exec.rows.len();
    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Mulhu).0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids).expect("extract shout lanes");

    let mulhu_lut_inst = LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: 0,
        d: 34,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Mulhu,
            xlen: 32,
        }),
        table: Vec::new(),
    };
    let mulhu_z = build_shout_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ mulhu_lut_inst.d * mulhu_lut_inst.ell,
        /*lanes=*/ 1,
        &shout_lanes,
        &x,
    )
    .expect("MULHU Shout z");
    let mulhu_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &mulhu_z);
    let mulhu_c = l.commit(&mulhu_Z);
    let mulhu_lut_inst = LutInstance::<Cmt, F> {
        comms: vec![mulhu_c],
        ..mulhu_lut_inst
    };
    let mulhu_lut_wit = LutWitness { mats: vec![mulhu_Z] };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(mulhu_lut_inst, mulhu_lut_wit)],
        mem_instances: Vec::new(),
        _phantom: PhantomData,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-mulhu");
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

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-mulhu");
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
