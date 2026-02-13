#![allow(non_snake_case)]

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

fn build_shout_only_bus_z_packed_bitwise(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lane_data: &neo_memory::riscv::trace::ShoutLaneOverTime,
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 34 {
        return Err(format!(
            "build_shout_only_bus_z_packed_bitwise: expected ell_addr=34 (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_only_bus_z_packed_bitwise: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.has_lookup.len() != t {
        return Err("build_shout_only_bus_z_packed_bitwise: lane length mismatch".into());
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::once((ell_addr, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_only_bus_z_packed_bitwise: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let cols = &bus.shout_cols[0].lanes[0];
    for j in 0..t {
        let has = lane_data.has_lookup[j];
        z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
        z[bus.bus_cell(cols.val, j)] = if has { F::from_u64(lane_data.value[j]) } else { F::ZERO };

        // Packed-key layout (ell_addr=34):
        // [lhs_u32, rhs_u32, lhs_digits[0..16], rhs_digits[0..16]] where each digit is base-4 in {0,1,2,3}.
        let mut packed = [F::ZERO; 34];
        if has {
            let (lhs_u64, rhs_u64) = uninterleave_bits(lane_data.key[j] as u128);
            let lhs_u32 = lhs_u64 as u32;
            let rhs_u32 = rhs_u64 as u32;

            packed[0] = F::from_u64(lhs_u32 as u64);
            packed[1] = F::from_u64(rhs_u32 as u64);

            for i in 0..16usize {
                let a = (lhs_u32 >> (2 * i)) & 3;
                let b = (rhs_u32 >> (2 * i)) & 3;
                packed[2 + i] = F::from_u64(a as u64);
                packed[18 + i] = F::from_u64(b as u64);
            }
        }

        for (idx, col_id) in cols.addr_bits.clone().enumerate() {
            z[bus.bus_cell(col_id, j)] = packed[idx];
        }
    }

    Ok(z)
}

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_bitwise_packed_prove_verify() {
    // Program:
    // - LUI  x1, 0x80000        (x1 = 0x80000000)
    // - XORI x2, x0, 1          (x2 = 1)
    // - OR   x3, x1, x2         (x3 = 0x80000001)
    // - ANDI x4, x3, 3          (x4 = 1)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 0x80000 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 4,
            rs1: 3,
            imm: 3,
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

    let mut exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    // Inject a single ANDN event into an otherwise shout-free row so we can exercise packed ANDN.
    {
        let shout = RiscvShoutTables::new(/*xlen=*/ 32);
        let shout_id = shout.opcode_to_id(RiscvOpcode::Andn);
        let lhs: u32 = 0x8000_0001;
        let rhs: u32 = 0x0000_0003;
        let val: u32 = lhs & !rhs;
        exec.rows[0].shout_events.clear();
        exec.rows[0].shout_events.push(ShoutEvent {
            shout_id,
            key: interleave_bits(lhs as u64, rhs as u64) as u64,
            value: val as u64,
        });
    }

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

    // Shout instances: AND/ANDN/OR/XOR packed, 1 lane each.
    let t = exec.rows.len();
    let shout_table_ids = vec![
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::And).0,
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Or).0,
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Xor).0,
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Andn).0,
    ];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids).expect("extract shout lanes");
    assert_eq!(shout_lanes.len(), 4);

    let mut lut_instances = Vec::new();
    for (idx, opcode) in [RiscvOpcode::And, RiscvOpcode::Or, RiscvOpcode::Xor, RiscvOpcode::Andn]
        .into_iter()
        .enumerate()
    {
        let inst = LutInstance::<Cmt, F> {
            comms: Vec::new(),
            k: 0,
            d: 34,
            n_side: 2,
            steps: t,
            lanes: 1,
            ell: 1,
            table_spec: Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen: 32 }),
            table: Vec::new(),
        };

        let z = build_shout_only_bus_z_packed_bitwise(ccs.m, layout.m_in, t, inst.d * inst.ell, &shout_lanes[idx], &x)
            .expect("packed bitwise z");
        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        let c = l.commit(&Z);
        let inst = LutInstance::<Cmt, F> { comms: vec![c], ..inst };
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

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-bitwise-packed");
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

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-bitwise-packed");
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
