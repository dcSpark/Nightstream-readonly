#![allow(non_snake_case)]

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, CommitMixers};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
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
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

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
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_twist_prove_verify() {
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
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
    //
    // NOTE: In no-shared-bus mode, each mem instance must provide its own committed witness mat.
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
    let ram_d = 2usize; // k=4, address bits=2 (keeps the test tiny)

    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    let twist_lanes = extract_twist_lanes_over_time(&exec, &init_regs, &init_ram, /*ram_ell_addr=*/ ram_d)
        .expect("extract twist lanes");

    // PROG
    let prog_mem_inst = MemInstance::<Cmt, F> {
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
    let prog_z = build_twist_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ prog_mem_inst.d * prog_mem_inst.ell,
        /*lanes=*/ 1,
        &[twist_lanes.prog.clone()],
        &x,
    )
    .expect("prog z");
    let prog_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &prog_z);
    let prog_c = l.commit(&prog_Z);
    let prog_mem_inst = MemInstance::<Cmt, F> {
        comms: vec![prog_c],
        ..prog_mem_inst
    };
    let prog_mem_wit = MemWitness { mats: vec![prog_Z] };

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

    let empty_lut_wit: LutWitness<F> = LutWitness { mats: Vec::new() };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: Vec::new(),
        mem_instances: vec![
            (prog_mem_inst, prog_mem_wit),
            (reg_mem_inst, reg_mem_wit),
            (ram_mem_inst, ram_mem_wit),
        ],
        _phantom: PhantomData,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-twist");
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

    // Sanity: no-shared-bus mode should emit Twist ME(time) claims and fold them.
    assert!(
        !proof.steps[0].mem.twist_me_claims_time.is_empty(),
        "expected Twist ME(time) claims in no-shared-bus mode"
    );
    assert!(
        !proof.steps[0].twist_time_fold.is_empty(),
        "expected twist_time_fold proofs in no-shared-bus mode"
    );
    assert!(
        !proof.steps[0].mem.val_me_claims.is_empty(),
        "expected val_me_claims in no-shared-bus mode"
    );
    assert!(
        !proof.steps[0].val_fold.is_empty(),
        "expected val_fold proofs in no-shared-bus mode"
    );

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-twist");
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

    // Quiet unused warning.
    let _ = empty_lut_wit;
}
