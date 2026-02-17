use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_shard::{rv32_b1_step_linking_config, Rv32B1, Rv32B1Run};
use neo_fold::session::FoldingSession;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, RiscvShoutTables};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::helpers::{step_bundle_recommit_after_private_tamper, StepWit};

fn prove_run(program: Vec<RiscvInstruction>, max_steps: usize) -> Rv32B1Run {
    let program_bytes = encode_program(&program);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(max_steps)
        .ram_bytes(0x200)
        // Keep this fixture explicit: these tests rely on XOR lookups in tiny programs,
        // and we don't want them coupled to shout auto-inference details.
        .shout_ops([RiscvOpcode::Add, RiscvOpcode::Xor])
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");
    run
}

fn prove_main_shard_proof_or_verify_fails(run: &Rv32B1Run, steps_bad: Vec<StepWit>) {
    let mut sess = FoldingSession::new(FoldingMode::Optimized, run.params().clone(), run.committer().clone());
    sess.set_step_linking(rv32_b1_step_linking_config(run.layout()));
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
fn rv32_b1_cpu_vs_bus_twist_rv_mismatch_must_fail() {
    // Program: LW x1, 0(x0); HALT, with RAM[0]=7
    let program = vec![
        RiscvInstruction::Load {
            op: neo_memory::riscv::lookups::RiscvMemOp::Lw,
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

    let idx_mem_rv = run.layout().mem_rv(0);

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    step_bundle_recommit_after_private_tamper(run.params(), run.committer(), &mut steps_bad[0], idx_mem_rv, F::ONE);

    prove_main_shard_proof_or_verify_fails(&run, steps_bad);
}

#[test]
fn rv32_b1_cpu_vs_bus_shout_val_mismatch_must_fail() {
    // Program: XOR x1, x0, x0; HALT (forces a Shout XOR lookup).
    let run = prove_run(
        vec![
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Xor,
                rd: 1,
                rs1: 0,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
        /*max_steps=*/ 2,
    );

    // Sanity: XOR table must be present in this run's Shout instances.
    let shout = RiscvShoutTables::new(32);
    let xor_table_id = shout.opcode_to_id(RiscvOpcode::Xor).0;
    let _ = run
        .layout()
        .shout_idx(xor_table_id)
        .expect("missing XOR Shout table");

    let idx_alu_out = run.layout().alu_out(0);

    let mut steps_bad: Vec<StepWit> = run.steps_witness().to_vec();
    step_bundle_recommit_after_private_tamper(run.params(), run.committer(), &mut steps_bad[0], idx_alu_out, F::ONE);

    prove_main_shard_proof_or_verify_fails(&run, steps_bad);
}
