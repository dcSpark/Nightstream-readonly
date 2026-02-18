use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::session::FoldingSession;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, PROG_ID, REG_ID};
use neo_memory::MemInit;
use p3_goldilocks::Goldilocks as F;

type StepWit = neo_memory::witness::StepWitnessBundle<neo_ajtai::Commitment, F, K>;

fn addi_halt_program_bytes(imm: i32) -> Vec<u8> {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm,
        },
        RiscvInstruction::Halt,
    ];
    encode_program(&program)
}

fn mem_idx(steps: &[StepWit], mem_id: u32) -> usize {
    steps[0]
        .mem_instances
        .iter()
        .position(|(inst, _)| inst.mem_id == mem_id)
        .unwrap_or_else(|| panic!("missing mem_id={mem_id} in step mem_instances"))
}

fn verifier_only_session_for_steps(run: &Rv32TraceWiringRun, steps: Vec<StepWit>) -> FoldingSession<AjtaiSModule> {
    let mut sess = FoldingSession::new(FoldingMode::Optimized, run.params().clone(), run.committer().clone());
    sess.set_step_linking(run.step_linking_config());
    sess.add_step_bundles(steps);
    sess
}

#[test]
fn trace_main_proof_truncated_steps_must_fail() {
    let program_bytes = addi_halt_program_bytes(1);
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    // Baseline: full step set verifies.
    let steps_ok: Vec<StepWit> = run.steps_witness().to_vec();
    let sess_ok = verifier_only_session_for_steps(&run, steps_ok);
    assert_eq!(
        sess_ok.verify_collected(run.ccs(), run.proof()).expect("main proof verify"),
        true
    );

    // Trace mode: entire trace = single step bundle, so the meaningful
    // truncation attack is providing ZERO steps.
    let steps_bad: Vec<StepWit> = Vec::new();
    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(matches!(res, Err(_) | Ok(false)), "zero steps must not verify");
}

#[test]
fn trace_main_proof_tamper_prog_init_must_fail() {
    let program_bytes = addi_halt_program_bytes(1);
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    let prog_idx = mem_idx(&steps_bad, PROG_ID.0);
    steps_bad[0].mem_instances[prog_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering PROG Twist init in public input must fail verification"
    );
}

#[test]
fn trace_main_proof_tamper_reg_init_must_fail() {
    let program_bytes = addi_halt_program_bytes(1);
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .reg_init_u32(2, 7)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    let reg_idx = mem_idx(&steps_bad, REG_ID.0);
    steps_bad[0].mem_instances[reg_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering REG Twist init in public input must fail verification"
    );
}

// NOTE: Step reordering is not applicable in trace mode because the entire
// trace is folded as a single step bundle. The intra-step state chain
// (PC chain, cycle chain) is enforced by the trace wiring CCS constraints,
// verified in riscv_trace_ccs_diverse_programs::trace_ccs_rejects_tampered_pc_after
// and trace_ccs_rejects_tampered_cycle.

#[test]
fn trace_main_proof_splicing_across_runs_must_fail() {
    let program_bytes_a = addi_halt_program_bytes(1);
    let mut run_a = Rv32TraceWiring::from_rom(0, &program_bytes_a)
        .max_steps(2)
        .prove()
        .expect("prove A");
    run_a.verify().expect("baseline verify A");

    let program_bytes_b = addi_halt_program_bytes(2);
    let mut run_b = Rv32TraceWiring::from_rom(0, &program_bytes_b)
        .max_steps(2)
        .prove()
        .expect("prove B");
    run_b.verify().expect("baseline verify B");

    let steps_bad: Vec<StepWit> = run_b.steps_witness().to_vec();
    let sess_bad = verifier_only_session_for_steps(&run_a, steps_bad);
    let res = sess_bad.verify_collected(run_a.ccs(), run_a.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "splicing main proof across runs must not verify"
    );
}
