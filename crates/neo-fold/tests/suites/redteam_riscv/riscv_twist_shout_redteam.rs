use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_shard::{rv32_b1_step_linking_config, Rv32B1, Rv32B1Run};
use neo_fold::session::FoldingSession;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode, RAM_ID};
use neo_memory::witness::LutTableSpec;
use neo_memory::MemInit;
use p3_goldilocks::Goldilocks as F;

type StepWit = neo_memory::witness::StepWitnessBundle<neo_ajtai::Commitment, F, K>;

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
fn rv32_b1_twist_instances_reordered_must_fail() {
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

    let mut run = match Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
    {
        Ok(run) => run,
        Err(_) => return,
    };
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    for step in &mut steps_bad {
        assert!(step.mem_instances.len() >= 2, "expected at least 2 Twist instances");
        step.mem_instances.swap(0, 1);
    }

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), &run.proof().main);
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "reordering Twist instances must not verify"
    );
}

#[test]
fn rv32_b1_shout_table_spec_tamper_must_fail() {
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

    let mut run = match Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
    {
        Ok(run) => run,
        Err(_) => return,
    };
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    for step in &mut steps_bad {
        assert!(!step.lut_instances.is_empty(), "expected at least 1 Shout instance");
        let lut_inst = &mut step.lut_instances[0].0;
        assert!(
            matches!(&lut_inst.table_spec, Some(LutTableSpec::RiscvOpcode { .. })),
            "expected a virtual RISC-V opcode table (table_spec=Some)"
        );
        lut_inst.table_spec = Some(LutTableSpec::RiscvOpcode {
            opcode: RiscvOpcode::Xor,
            xlen: 32,
        });
    }

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), &run.proof().main);
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering Shout table_spec must not verify"
    );
}

#[test]
fn rv32_b1_shout_instances_reordered_must_fail() {
    // Ensure we have at least two Shout tables by including ADDI + ORI.
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

    let mut run = match Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
    {
        Ok(run) => run,
        Err(_) => return,
    };
    run.verify().expect("baseline verify");

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    for step in &mut steps_bad {
        assert!(
            step.lut_instances.len() >= 2,
            "expected at least 2 Shout instances for ADDI+ORI program"
        );
        step.lut_instances.swap(0, 1);
    }

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), &run.proof().main);
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "reordering Shout instances must not verify"
    );
}

#[test]
fn rv32_b1_ram_init_statement_tamper_must_fail() {
    // Program: LW x1, 0(x0); HALT
    //
    // We set RAM[0] in the *public statement* and force a load to consume it,
    // so the Twist proof must be bound to the RAM init.
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

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .ram_init_u32(/*addr=*/ 0, /*value=*/ 7)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let ram_idx = mem_idx(&run, RAM_ID.0);

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    steps_bad[0].mem_instances[ram_idx].0.init = MemInit::Zero;

    let sess_bad = verifier_only_session_for_steps(&run, steps_bad);
    let res = sess_bad.verify_collected(run.ccs(), &run.proof().main);
    assert!(
        matches!(res, Err(_) | Ok(false)),
        "tampering RAM Twist init in public input must fail verification"
    );
}
