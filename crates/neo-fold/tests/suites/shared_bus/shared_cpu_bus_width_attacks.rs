use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use neo_memory::riscv::trace::{rv32_width_lookup_backed_cols, Rv32WidthSidecarLayout};
use p3_field::PrimeCharacteristicRing;

fn prove_width_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
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

fn tamper_width_opening_scalar(proof: &mut ShardProof, width_col: usize) {
    let layout = Rv32WidthSidecarLayout::new();
    let width_open_cols = rv32_width_lookup_backed_cols(&layout);
    assert_eq!(
        proof.steps[0].mem.wp_me_claims.len(),
        1,
        "expected one WP ME claim carrying width lookup openings"
    );
    let me = &mut proof.steps[0].mem.wp_me_claims[0];
    let width_open_start = me
        .y_scalars
        .len()
        .checked_sub(width_open_cols.len())
        .expect("width openings must be appended to WP ME tail");
    let width_idx = width_open_cols
        .iter()
        .position(|&c| c == width_col)
        .expect("expected width lookup opening column");
    me.y_scalars[width_open_start + width_idx] += K::ONE;
}

#[test]
fn width_low_bit_tamper_is_rejected() {
    let (run, mut proof) = prove_width_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_width_opening_scalar(&mut proof, layout.ram_rv_low_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered width stage low-bit opening must fail verification"
    );
}

#[test]
fn width_load_semantics_tamper_is_rejected() {
    let (run, mut proof) = prove_width_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_width_opening_scalar(&mut proof, layout.ram_rv_q16);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered width stage load-semantics opening must fail verification"
    );
}

#[test]
fn width_store_semantics_tamper_is_rejected() {
    let (run, mut proof) = prove_width_trace_program();
    let layout = Rv32WidthSidecarLayout::new();
    tamper_width_opening_scalar(&mut proof, layout.rs2_low_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered width stage store-semantics opening must fail verification"
    );
}
