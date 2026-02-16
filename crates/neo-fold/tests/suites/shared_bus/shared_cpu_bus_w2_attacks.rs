use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::trace::Rv32DecodeSidecarLayout;
use p3_field::PrimeCharacteristicRing;

fn prove_w2_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    // Program exercises both ALU-imm and ALU-reg decode/linkage paths.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    (run, proof)
}

fn tamper_w2_opening_scalar(proof: &mut ShardProof, decode_col: usize) {
    let layout = Rv32DecodeSidecarLayout::new();
    assert_eq!(
        proof.steps[0].mem.w2_decode_me_claims.len(),
        1,
        "expected one W2 decode ME claim"
    );
    let me = &mut proof.steps[0].mem.w2_decode_me_claims[0];
    let core_t = me
        .y_scalars
        .len()
        .checked_sub(layout.cols)
        .expect("W2 ME opening shape");
    me.y_scalars[core_t + decode_col] += K::ONE;
}

#[test]
fn w2_write_gate_tamper_is_rejected() {
    let (run, mut proof) = prove_w2_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_w2_opening_scalar(&mut proof, layout.op_alu_imm_write);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W2 write-gate opening must fail verification"
    );
}

#[test]
fn w2_alu_table_delta_tamper_is_rejected() {
    let (run, mut proof) = prove_w2_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_w2_opening_scalar(&mut proof, layout.alu_reg_table_delta);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W2 ALU table-delta opening must fail verification"
    );
}
