use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::session::FoldingSession;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode, RAM_ID};
use neo_memory::witness::LutTableSpec;
use neo_memory::MemInit;
use p3_goldilocks::Goldilocks as F;

type StepWit = neo_memory::witness::StepWitnessBundle<neo_ajtai::Commitment, F, K>;

fn verifier_only_session_for_steps(run: &Rv32TraceWiringRun, steps: Vec<StepWit>) -> FoldingSession<AjtaiSModule> {
    let mut sess = FoldingSession::new(FoldingMode::Optimized, run.params().clone(), run.committer().clone());
    sess.set_step_linking(run.step_linking_config());
    sess.add_step_bundles(steps);
    sess
}

#[test]
fn trace_twist_instances_reordered_must_fail() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    assert!(
        steps_bad[0].mem_instances.len() >= 2,
        "expected at least 2 Twist instances"
    );
    steps_bad[0].mem_instances.swap(0, 1);

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "reordering Twist instances must not verify"
    );
}

#[test]
fn trace_shout_table_spec_tamper_must_fail() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    assert!(
        !steps_bad[0].lut_instances.is_empty(),
        "expected at least 1 Shout instance"
    );
    let lut_inst = &mut steps_bad[0].lut_instances[0].0;
    assert!(
        matches!(&lut_inst.table_spec, Some(LutTableSpec::RiscvOpcode { .. })),
        "expected a virtual RISC-V opcode table (table_spec=Some)"
    );
    lut_inst.table_spec = Some(LutTableSpec::RiscvOpcode {
        opcode: RiscvOpcode::Xor,
        xlen: 32,
    });

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering Shout table_spec must not verify"
    );
}

#[test]
fn trace_shout_instances_reordered_must_fail() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 2,
            rs1: 1,
            imm: 3,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(4)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    assert!(
        steps_bad[0].lut_instances.len() >= 2,
        "expected at least 2 Shout instances for ADDI+ORI program"
    );
    steps_bad[0].lut_instances.swap(0, 1);

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "reordering Shout instances must not verify"
    );
}

#[test]
fn trace_ram_init_statement_tamper_must_fail() {
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .max_steps(2)
        .ram_init_u32(0, 7)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let ram_idx = run
        .steps_witness()[0]
        .mem_instances
        .iter()
        .position(|(inst, _)| inst.mem_id == RAM_ID.0)
        .expect("missing RAM Twist instance");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    steps_bad[0].mem_instances[ram_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), run.proof());
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering RAM Twist init in public input must fail verification"
    );
}
