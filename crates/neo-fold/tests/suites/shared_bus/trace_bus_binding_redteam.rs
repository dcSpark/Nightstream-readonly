use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::session::FoldingSession;
use neo_math::K;
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

type StepWit = neo_memory::witness::StepWitnessBundle<neo_ajtai::Commitment, F, K>;

fn prove_run(program: Vec<RiscvInstruction>, max_steps: usize) -> Rv32TraceWiringRun {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(max_steps)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");
    run
}

fn step_bundle_recommit_after_private_tamper(
    params: &neo_params::NeoParams,
    committer: &AjtaiSModule,
    step: &mut StepWit,
    idx_to_tamper: usize,
    delta: F,
) {
    let (ref mut inst, ref mut wit) = step.mcs;
    let m_in = inst.m_in;
    assert!(
        idx_to_tamper >= m_in,
        "expected idx_to_tamper to be in the private witness region (idx={idx_to_tamper}, m_in={m_in})"
    );

    let mut z = Vec::with_capacity(m_in + wit.w.len());
    z.extend_from_slice(&inst.x);
    z.extend_from_slice(&wit.w);
    assert!(idx_to_tamper < z.len(), "idx_to_tamper out of range");

    z[idx_to_tamper] += delta;
    wit.w = z[m_in..].to_vec();
    wit.Z = encode_vector_balanced_to_mat(params, &z);
    inst.c = neo_ccs::traits::SModuleHomomorphism::commit(committer, &wit.Z);
}

fn prove_main_shard_proof_or_verify_fails(run: &Rv32TraceWiringRun, steps_bad: Vec<StepWit>) {
    let mut sess = FoldingSession::new(FoldingMode::Optimized, run.params().clone(), run.committer().clone());
    sess.set_step_linking(run.step_linking_config());
    sess.add_step_bundles(steps_bad);

    let Ok(proof_bad) = sess.fold_and_prove(run.ccs()) else {
        return;
    };
    let res = sess.verify_collected(run.ccs(), &proof_bad);
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "malicious main proof unexpectedly verified"
    );
}

#[test]
fn trace_cpu_vs_bus_twist_rv_mismatch_must_fail() {
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let run = prove_run(program, 2);

    let layout = run.layout();
    let t = layout.t;
    let trace = &layout.trace;

    let idx_ram_rv = layout.trace_base + trace.ram_rv * t + 0;

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    step_bundle_recommit_after_private_tamper(run.params(), run.committer(), &mut steps_bad[0], idx_ram_rv, F::ONE);

    prove_main_shard_proof_or_verify_fails(&run, steps_bad);
}

#[test]
fn trace_cpu_vs_bus_shout_val_mismatch_must_fail() {
    let run = prove_run(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::Halt,
        ],
        2,
    );

    let layout = run.layout();
    let t = layout.t;
    let trace = &layout.trace;

    let idx_shout_val = layout.trace_base + trace.shout_val * t + 0;

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    step_bundle_recommit_after_private_tamper(run.params(), run.committer(), &mut steps_bad[0], idx_shout_val, F::ONE);

    prove_main_shard_proof_or_verify_fails(&run, steps_bad);
}
