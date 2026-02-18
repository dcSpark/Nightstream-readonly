use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::trace::{rv32_decode_lookup_backed_cols, Rv32DecodeSidecarLayout, Rv32TraceLayout};
use p3_field::PrimeCharacteristicRing;

fn prove_control_trace_program(program: Vec<RiscvInstruction>) -> (Rv32TraceWiringRun, ShardProof) {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

fn rv32_wp_opening_cols(layout: &Rv32TraceLayout) -> Vec<usize> {
    vec![
        layout.active,
        layout.instr_word,
        layout.rs1_addr,
        layout.rs1_val,
        layout.rs2_addr,
        layout.rs2_val,
        layout.rd_addr,
        layout.rd_val,
        layout.ram_addr,
        layout.ram_rv,
        layout.ram_wv,
        layout.shout_has_lookup,
        layout.shout_val,
        layout.shout_lhs,
        layout.shout_rhs,
        layout.jalr_drop_bit,
        layout.pc_before,
        layout.pc_after,
    ]
}

fn tamper_control_decode_opening_scalar(proof: &mut ShardProof, decode_col: usize) {
    let layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_backed_cols(&layout);
    assert_eq!(
        proof.steps[0].mem.wp_me_claims.len(),
        1,
        "expected one WP ME claim carrying decode openings for control stage checks"
    );
    let me = &mut proof.steps[0].mem.wp_me_claims[0];
    let decode_start = me
        .y_scalars
        .len()
        .checked_sub(decode_open_cols.len())
        .expect("control stage decode opening shape in WP ME tail");
    let open_idx = decode_open_cols
        .iter()
        .position(|&c| c == decode_col)
        .expect("decode col must be present in control stage decode opening set");
    me.y_scalars[decode_start + open_idx] += K::ONE;
}

fn tamper_control_wp_opening_scalar(proof: &mut ShardProof, trace_col: usize) {
    let layout = Rv32TraceLayout::new();
    let open_cols = rv32_wp_opening_cols(&layout);
    let decode_open_cols = rv32_decode_lookup_backed_cols(&Rv32DecodeSidecarLayout::new());
    let open_idx = open_cols
        .iter()
        .position(|&c| c == trace_col)
        .expect("trace col must be present in control stage WP opening set");
    assert_eq!(
        proof.steps[0].mem.wp_me_claims.len(),
        1,
        "expected one WP ME claim reused by control stage checks"
    );
    let me = &mut proof.steps[0].mem.wp_me_claims[0];
    let core_t = me
        .y_scalars
        .len()
        .checked_sub(decode_open_cols.len())
        .expect("control stage decode opening tail shape")
        .checked_sub(open_cols.len())
        .expect("control stage WP opening shape");
    me.y_scalars[core_t + open_idx] += K::ONE;
}

#[test]
fn control_jal_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::Jal { rd: 1, imm: 8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.imm_j);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage JAL target opening must fail verification"
    );
}

#[test]
fn control_jalr_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 8,
        },
        RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.imm_i);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage JALR target opening must fail verification"
    );
}

#[test]
fn control_branch_decision_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.funct3_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage branch decision/target opening must fail verification"
    );
}

#[test]
fn control_control_writeback_tamper_is_rejected() {
    let program = vec![RiscvInstruction::Auipc { rd: 1, imm: 1 }, RiscvInstruction::Halt];
    let (run, mut proof) = prove_control_trace_program(program);
    let trace = Rv32TraceLayout::new();
    tamper_control_wp_opening_scalar(&mut proof, trace.rd_val);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage control-writeback opening must fail verification"
    );
}
