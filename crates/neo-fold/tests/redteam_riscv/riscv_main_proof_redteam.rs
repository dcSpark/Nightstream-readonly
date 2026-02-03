use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_shard::{rv32_b1_step_linking_config, Rv32B1, Rv32B1Run};
use neo_fold::session::FoldingSession;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, PROG_ID, REG_ID};
use neo_memory::MemInit;
use neo_math::K;
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

fn mem_idx(run: &Rv32B1Run, mem_id: u32) -> usize {
    let mut mem_ids: Vec<u32> = run.mem_layouts().keys().copied().collect();
    mem_ids.sort_unstable();
    mem_ids
        .iter()
        .position(|&id| id == mem_id)
        .unwrap_or_else(|| panic!("missing mem_id={mem_id} in mem_layouts"))
}

fn verifier_only_session_for_steps(run: &Rv32B1Run, steps: Vec<StepWit>) -> FoldingSession<AjtaiSModule> {
    let mut sess = FoldingSession::new(FoldingMode::Optimized, run.params().clone(), run.committer().clone());
    sess.set_step_linking(rv32_b1_step_linking_config(run.layout()));
    sess.add_step_bundles(steps);
    sess
}

#[test]
fn rv32_b1_main_proof_truncated_steps_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove");

    // Baseline: full verification (includes sidecars).
    run.verify().expect("baseline verify");

    // Baseline: main proof alone verifies when steps match.
    let steps_ok: Vec<StepWit> = run.steps_witness().to_vec();
    let sess_ok = verifier_only_session_for_steps(&run, steps_ok);
    assert_eq!(
        sess_ok
            .verify_collected(run.ccs(), run.proof())
            .expect("main proof verify"),
        true
    );

    // Truncate steps (verifier-side) and reuse the original proof.
    let steps_bad: Vec<StepWit> = run.steps_witness().iter().cloned().take(1).collect();
    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "truncated steps must not verify"
    );
}

#[test]
fn rv32_b1_main_proof_tamper_prog_init_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove");

    run.verify().expect("baseline verify");

    let prog_idx = mem_idx(&run, PROG_ID.0);

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    steps_bad[0].mem_instances[prog_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering PROG Twist init in public input must fail verification"
    );
}

#[test]
fn rv32_b1_main_proof_tamper_reg_init_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        // Make REG init non-trivial in the public statement.
        .reg_init_u32(/*reg=*/ 2, /*value=*/ 7)
        .prove()
        .expect("prove");

    run.verify().expect("baseline verify");

    let reg_idx = mem_idx(&run, REG_ID.0);

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    steps_bad[0].mem_instances[reg_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering REG Twist init in public input must fail verification"
    );
}

#[test]
fn rv32_b1_main_proof_step_reordering_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    assert!(steps_bad.len() >= 2, "expected at least 2 steps for reordering test");
    steps_bad.swap(0, 1);

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "reordering shard steps must not verify"
    );
}

#[test]
fn rv32_b1_main_proof_splicing_across_runs_must_fail() {
    let program_bytes_a = addi_halt_program_bytes(/*imm=*/ 1);
    let mut run_a = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes_a)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove A");
    run_a.verify().expect("baseline verify A");

    let program_bytes_b = addi_halt_program_bytes(/*imm=*/ 2);
    let mut run_b = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes_b)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove B");
    run_b.verify().expect("baseline verify B");

    // Attempt to verify run A's main proof against run B's public step bundles.
    let steps_bad: Vec<StepWit> = run_b.steps_witness().to_vec();
    let sess_bad = verifier_only_session_for_steps(&run_a, steps_bad);
    let res = sess_bad.verify_collected(run_a.ccs(), run_a.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "splicing main proof across runs must not verify"
    );
}
