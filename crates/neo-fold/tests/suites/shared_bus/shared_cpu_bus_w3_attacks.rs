use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use neo_memory::riscv::trace::Rv32WidthSidecarLayout;
use p3_field::PrimeCharacteristicRing;

fn prove_w3_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    // Program exercises load/store selector and width semantics:
    //   ADDI x1, x0, 1
    //   SW   x1, 0(x0)
    //   LW   x2, 0(x0)
    //   HALT
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

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    (run, proof)
}

fn tamper_w3_opening_scalar(proof: &mut ShardProof, width_col: usize) {
    let layout = Rv32WidthSidecarLayout::new();
    assert_eq!(
        proof.steps[0].mem.w3_width_me_claims.len(),
        1,
        "expected one W3 width ME claim"
    );
    let me = &mut proof.steps[0].mem.w3_width_me_claims[0];
    let core_t = me
        .y_scalars
        .len()
        .checked_sub(layout.cols)
        .expect("W3 ME opening shape");
    me.y_scalars[core_t + width_col] += K::ONE;
}

#[test]
fn w3_low_bit_tamper_is_rejected() {
    let (run, mut proof) = prove_w3_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_w3_opening_scalar(&mut proof, layout.ram_rv_low_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W3 low-bit opening must fail verification"
    );
}

#[test]
fn w3_selector_tamper_is_rejected() {
    let (run, mut proof) = prove_w3_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_w3_opening_scalar(&mut proof, layout.is_lb);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W3 selector opening must fail verification"
    );
}

#[test]
fn w3_load_semantics_tamper_is_rejected() {
    let (run, mut proof) = prove_w3_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_w3_opening_scalar(&mut proof, layout.ram_rv_q16);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W3 load-semantics opening must fail verification"
    );
}

#[test]
fn w3_store_semantics_tamper_is_rejected() {
    let (run, mut proof) = prove_w3_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_w3_opening_scalar(&mut proof, layout.rs2_low_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered W3 store-semantics opening must fail verification"
    );
}
