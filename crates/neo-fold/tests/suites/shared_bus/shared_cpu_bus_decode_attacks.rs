use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::trace::{rv32_decode_lookup_backed_cols, Rv32DecodeSidecarLayout};
use p3_field::PrimeCharacteristicRing;

fn prove_decode_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
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

fn tamper_decode_opening_scalar(proof: &mut ShardProof, decode_col: usize) {
    let layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_backed_cols(&layout);
    assert_eq!(
        proof.steps[0].mem.wp_me_claims.len(),
        1,
        "expected one WP ME claim carrying decode openings"
    );
    let me = &mut proof.steps[0].mem.wp_me_claims[0];
    let decode_start = me
        .y_scalars
        .len()
        .checked_sub(decode_open_cols.len())
        .expect("decode openings must be appended to WP ME tail");
    let open_idx = decode_open_cols
        .iter()
        .position(|&c| c == decode_col)
        .expect("decode col must be present in WP decode opening tail");
    me.y_scalars[decode_start + open_idx] += K::ONE;
}

#[test]
fn decode_write_gate_tamper_is_rejected() {
    let (run, mut proof) = prove_decode_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_decode_opening_scalar(&mut proof, layout.op_alu_imm);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered decode stage opcode-class opening must fail verification"
    );
}

#[test]
fn decode_alu_table_delta_tamper_is_rejected() {
    let (run, mut proof) = prove_decode_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_decode_opening_scalar(&mut proof, layout.rs2);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered decode stage rs2-decode opening must fail verification"
    );
}
