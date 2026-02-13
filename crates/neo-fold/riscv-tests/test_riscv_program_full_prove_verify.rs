//! End-to-end prove+verify for small RV32 programs under the B1 shared-bus step circuit.
//!
//! This exercises:
//! - B1 instruction fetch via `PROG_ID` Twist reads
//! - shared CPU bus tail wiring (Twist + Shout)
//! - Shout addr-pre masking (skipping inactive lookups)
//! - decode + semantics sidecar proofs (required for soundness)

#![allow(non_snake_case)]

use neo_fold::riscv_shard::{rv32_b1_enforce_chunk0_mem_init_matches_statement, Rv32B1};
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode, RAM_ID};
use neo_memory::riscv::shard::extract_boundary_state;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_program_full_prove_verify() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 1 }, // x1 = 0x1000
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        }, // x1 = 0x1005
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // x2 = 7
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 0x100c
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0x100,
        }, // mem[0x100] = x3
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 4,
            rs1: 0,
            imm: 0x100,
        }, // x4 = mem[0x100]
        RiscvInstruction::Auipc { rd: 5, imm: 0 }, // x5 = pc
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();

    let program_bytes = encode_program(&program);

    // Keep the Shout bus lean: this program only needs ADD (for ADD/ADDI and effective address calculation).
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .ram_bytes(0x200)
        .chunk_size(1)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    // Ensure the Shout addr-pre proof skips inactive tables.
    // This program uses only the ADD lookup; LUI and HALT use no Shout lookups and should skip entirely.
    let proof = run.proof();
    let mut saw_skipped = false;
    let mut saw_add_only = false;
    for step in &proof.main.steps {
        let pre = &step.mem.shout_addr_pre;
        let active_lanes: Vec<u32> = pre
            .groups
            .iter()
            .flat_map(|g| g.active_lanes.iter().copied())
            .collect();
        if active_lanes.is_empty() {
            assert!(pre.groups.iter().all(|g| g.round_polys.is_empty()));
            saw_skipped = true;
            continue;
        }
        // With `shout_ops([ADD])`, there is exactly one Shout lane and it is lane 0.
        assert_eq!(
            active_lanes,
            vec![0u32],
            "expected ADD-only Shout addr-pre active_lanes"
        );
        let rounds_total: usize = pre.groups.iter().map(|g| g.round_polys.len()).sum();
        assert_eq!(rounds_total, 1, "ADD-only step must include 1 proof");
        saw_add_only = true;
    }
    assert!(saw_skipped, "expected at least one no-Shout step (mask=0)");
    assert!(saw_add_only, "expected at least one ADD-lookup step (mask=ADD)");

    // Tamper: change Shout addr-pre active_lanes; verification must fail.
    let mut bad_bundle = proof.clone();
    let tamper_step = bad_bundle
        .main
        .steps
        .iter_mut()
        .find(|s| {
            s.mem
                .shout_addr_pre
                .groups
                .iter()
                .any(|g| !g.active_lanes.is_empty())
        })
        .expect("expected at least one active Shout addr-pre step");
    let group = tamper_step
        .mem
        .shout_addr_pre
        .groups
        .iter_mut()
        .find(|g| !g.active_lanes.is_empty())
        .expect("expected at least one active Shout addr-pre group");
    group.active_lanes.clear();
    group.round_polys.clear();
    assert!(
        run.verify_proof_bundle(&bad_bundle).is_err(),
        "expected Shout addr-pre active_lanes mismatch failure"
    );
}

#[test]
fn test_riscv_statement_mem_init_mismatch_fails() {
    let xlen = 32usize;
    let program = vec![RiscvInstruction::Halt];
    let max_steps = program.len();

    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .ram_bytes(0x40)
        .chunk_size(1)
        .max_steps(max_steps)
        // This program uses no Shout lookups, but keep ADD to keep the bus schema stable.
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    // External verifier check: the *statement* initial memory must match chunk0's public MemInit.
    let steps_public = run.steps_public();
    rv32_b1_enforce_chunk0_mem_init_matches_statement(run.mem_layouts(), run.initial_mem(), &steps_public)
        .expect("statement mem init must match");

    // Mismatch the *statement* initial memory (RAM starts non-zero) while keeping the proof fixed.
    // The statement check must fail.
    let mut bad_statement_initial_mem = run.initial_mem().clone();
    bad_statement_initial_mem.insert((RAM_ID.0, 0u64), F::ONE);
    assert!(
        rv32_b1_enforce_chunk0_mem_init_matches_statement(run.mem_layouts(), &bad_statement_initial_mem, &steps_public)
            .is_err(),
        "expected statement init mismatch failure"
    );
}

#[test]
#[ignore = "manual benchmark sweep; run with --ignored --nocapture"]
fn perf_rv32_b1_chunk_size_sweep() {
    use std::time::Instant;

    fn opcode_from_table_id(id: u32) -> RiscvOpcode {
        match id {
            0 => RiscvOpcode::And,
            1 => RiscvOpcode::Xor,
            2 => RiscvOpcode::Or,
            3 => RiscvOpcode::Add,
            4 => RiscvOpcode::Sub,
            5 => RiscvOpcode::Slt,
            6 => RiscvOpcode::Sltu,
            7 => RiscvOpcode::Sll,
            8 => RiscvOpcode::Srl,
            9 => RiscvOpcode::Sra,
            10 => RiscvOpcode::Eq,
            11 => RiscvOpcode::Neq,
            12 => RiscvOpcode::Mul,
            13 => RiscvOpcode::Mulh,
            14 => RiscvOpcode::Mulhu,
            15 => RiscvOpcode::Mulhsu,
            16 => RiscvOpcode::Div,
            17 => RiscvOpcode::Divu,
            18 => RiscvOpcode::Rem,
            19 => RiscvOpcode::Remu,
            _ => panic!("unsupported RV32 B1 table_id={id}"),
        }
    }

    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::Branch {
            cond: neo_memory::riscv::lookups::BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // not taken
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        }, // mem[0] = x1
        RiscvInstruction::Jal { rd: 5, imm: 8 }, // jump over the next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 123,
        }, // skipped
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 3,
            rs1: 0,
            imm: 0,
        }, // x3 = mem[0]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let max_steps = 64usize;

    let profiles: &[(&str, &[u32])] = &[
        ("min3", neo_memory::riscv::ccs::RV32_B1_SHOUT_PROFILE_MIN3),
        ("full12", neo_memory::riscv::ccs::RV32_B1_SHOUT_PROFILE_FULL12),
    ];

    for (profile_name, table_ids) in profiles {
        let ops: Vec<RiscvOpcode> = table_ids
            .iter()
            .copied()
            .map(opcode_from_table_id)
            .collect();
        println!("\n== profile={profile_name} shout_tables={} ==", table_ids.len());

        for chunk_size in [1usize, 2, 4, 8, 16] {
            let t_total = Instant::now();
            let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
                .xlen(xlen)
                .ram_bytes(0x40)
                .chunk_size(chunk_size)
                .max_steps(max_steps)
                .shout_ops(ops.iter().copied())
                .prove()
                .expect("prove");
            let total_dur = t_total.elapsed();
            let prove_dur = run.prove_duration();

            run.verify().expect("verify");
            let verify_dur = run.verify_duration().expect("verify duration");
            let chunks = run.steps_public().len();

            println!(
                "chunk_size={chunk_size:<2} chunks={chunks:<3} prove={:?} verify={:?} total={:?}",
                prove_dur, verify_dur, total_dur
            );
        }
    }
}

#[test]
fn test_riscv_program_chunk_size_equivalence() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        }, // mem[0] = x1
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0,
        }, // x2 = mem[0]
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();
    let program_bytes = encode_program(&program);

    let mut run_1 = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .ram_bytes(0x40)
        .chunk_size(1)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove chunk_size=1");
    run_1.verify().expect("verify chunk_size=1");
    let steps_1 = run_1.steps_public();

    let mut run_2 = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .ram_bytes(0x40)
        .chunk_size(2)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove chunk_size=2");
    run_2.verify().expect("verify chunk_size=2");
    let steps_2 = run_2.steps_public();

    let start_1 = extract_boundary_state(run_1.layout(), &steps_1[0].mcs_inst.x).expect("boundary");
    let start_2 = extract_boundary_state(run_2.layout(), &steps_2[0].mcs_inst.x).expect("boundary");
    assert_eq!(start_1.pc0, start_2.pc0, "pc0 must be chunk-size invariant");

    let end_1 =
        extract_boundary_state(run_1.layout(), &steps_1.last().expect("non-empty").mcs_inst.x).expect("boundary");
    let end_2 =
        extract_boundary_state(run_2.layout(), &steps_2.last().expect("non-empty").mcs_inst.x).expect("boundary");
    assert_eq!(end_1.pc_final, end_2.pc_final, "pc_final must be chunk-size invariant");

    // Stronger equivalence: each chunk boundary in chunk_size=2 corresponds to the same boundary
    // after the same number of steps in chunk_size=1.
    let n = steps_1.len();
    let k = 2usize;
    assert_eq!(n, max_steps, "chunk_size=1 should produce one chunk per step");
    assert_eq!(steps_2.len(), n.div_ceil(k), "unexpected chunk count for chunk_size=2");

    for c in 0..steps_2.len() {
        let s = c * k;
        let e = ((c + 1) * k).min(n) - 1;
        let st_k = extract_boundary_state(run_2.layout(), &steps_2[c].mcs_inst.x).expect("boundary");
        let st_1s = extract_boundary_state(run_1.layout(), &steps_1[s].mcs_inst.x).expect("boundary");
        let st_1e = extract_boundary_state(run_1.layout(), &steps_1[e].mcs_inst.x).expect("boundary");

        assert_eq!(st_k.pc0, st_1s.pc0, "pc0 mismatch at chunk {c}");
        assert_eq!(st_k.halted_in, st_1s.halted_in, "halted_in mismatch at chunk {c}");

        assert_eq!(st_k.pc_final, st_1e.pc_final, "pc_final mismatch at chunk {c}");
        assert_eq!(st_k.halted_out, st_1e.halted_out, "halted_out mismatch at chunk {c}");
    }
}

#[test]
fn test_riscv_program_rv32m_full_prove_verify() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -6,
        }, // x1 = -6
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();
    let program_bytes = encode_program(&program);

    // Minimal table set:
    // - ADD (for ADD/ADDI and address/PC wiring),
    // - SLTU (for signed DIV/REM remainder-bound check when divisor != 0).
    //
    // Note: RV32 B1 proves RV32M MUL* via the RV32M event sidecar CCS (no Shout table required).
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .ram_bytes(0x40)
        .chunk_size(1)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add, RiscvOpcode::Sltu])
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    let steps = run.steps_public();
    let mut rv32m_chunks: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, step)| {
            let count = step.mcs_inst.x[run.layout().rv32m_count];
            (count != F::ZERO).then_some(chunk_idx)
        })
        .collect();
    rv32m_chunks.sort_unstable();
    assert_eq!(rv32m_chunks, vec![2, 3], "expected RV32M rows on the MUL/DIV chunks");

    let rv32m = run
        .proof()
        .rv32m
        .as_ref()
        .expect("expected RV32M sidecar proofs");
    let mut proof_chunks: Vec<usize> = rv32m.iter().map(|p| p.chunk_idx).collect();
    proof_chunks.sort_unstable();
    assert_eq!(proof_chunks, vec![2, 3], "expected one RV32M proof per M chunk");
    for p in rv32m {
        assert_eq!(p.lanes, vec![0u32], "chunk_size=1 => M op must be lane 0");
    }
}
