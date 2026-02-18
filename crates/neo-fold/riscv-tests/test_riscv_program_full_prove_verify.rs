//! End-to-end prove+verify for small RV32 programs under the trace wiring circuit.

#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
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

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .chunk_rows(1)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    // Ensure the Shout addr-pre proof skips inactive tables.
    let proof = run.proof();
    let mut saw_active = false;
    for step in &proof.steps {
        let pre = &step.mem.shout_addr_pre;
        let active_lanes: Vec<u32> = pre
            .groups
            .iter()
            .flat_map(|g| g.active_lanes.iter().copied())
            .collect();
        if active_lanes.is_empty() {
            assert!(pre.groups.iter().all(|g| g.round_polys.is_empty()));
            continue;
        }
        let rounds_total: usize = pre.groups.iter().map(|g| g.round_polys.len()).sum();
        assert!(rounds_total > 0, "active lanes must carry addr-pre round polys");
        saw_active = true;
    }
    assert!(saw_active, "expected at least one active addr-pre step");

    // Tamper: change Shout addr-pre active_lanes; verification must fail.
    let mut bad_proof = proof.clone();
    let tamper_step = bad_proof
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
        run.verify_proof(&bad_proof).is_err(),
        "expected Shout addr-pre active_lanes mismatch failure"
    );
}

#[test]
fn test_riscv_wrong_output_claim_fails_verify() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0x100,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(1)
        .max_steps(program.len())
        .output_claim(0x100, F::from_u64(8))
        .prove()
    {
        Ok(mut run) => assert!(
            run.verify().is_err(),
            "wrong output claim must fail verification"
        ),
        Err(_) => {}
    }
}

#[test]
#[ignore = "manual benchmark sweep; run with --ignored --nocapture"]
fn perf_rv32_trace_chunk_rows_sweep() {
    use std::time::Instant;

    let xlen = 32usize;
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
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        },
        RiscvInstruction::Jal { rd: 5, imm: 8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 123,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 3,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let max_steps = 64usize;

    let profiles: &[(&str, &[RiscvOpcode])] = &[
        ("minimal", &[RiscvOpcode::Add]),
        ("extended", &[RiscvOpcode::Add, RiscvOpcode::Sub, RiscvOpcode::Sltu]),
    ];

    for (profile_name, ops) in profiles {
        println!("\n== profile={profile_name} shout_tables={} ==", ops.len());

        for chunk_rows in [1usize, 2, 4, 8, 16] {
            let t_total = Instant::now();
            let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
                .xlen(xlen)
                .chunk_rows(chunk_rows)
                .max_steps(max_steps)
                .shout_ops(ops.iter().copied())
                .prove()
                .expect("prove");
            let total_dur = t_total.elapsed();
            let prove_dur = run.prove_duration();

            run.verify().expect("verify");
            let verify_dur = run.verify_duration().expect("verify duration");
            let folds = run.fold_count();

            println!(
                "chunk_rows={chunk_rows:<2} folds={folds:<3} prove={:?} verify={:?} total={:?}",
                prove_dur, verify_dur, total_dur
            );
        }
    }
}

#[test]
fn test_riscv_program_chunk_rows_equivalence() {
    let xlen = 32usize;
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
    let max_steps = program.len();
    let program_bytes = encode_program(&program);

    let mut run_1 = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .chunk_rows(1)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove chunk_rows=1");
    run_1.verify().expect("verify chunk_rows=1");

    let mut run_2 = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .chunk_rows(2)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove chunk_rows=2");
    run_2.verify().expect("verify chunk_rows=2");

    let first_1 = run_1.exec_table().rows.first().expect("non-empty trace");
    let first_2 = run_2.exec_table().rows.first().expect("non-empty trace");
    assert_eq!(first_1.pc_before, first_2.pc_before, "pc_before must be invariant");

    let last_1 = run_1.exec_table().rows.last().expect("non-empty trace");
    let last_2 = run_2.exec_table().rows.last().expect("non-empty trace");
    assert_eq!(last_1.pc_after, last_2.pc_after, "pc_after must be invariant");
    assert_eq!(run_1.trace_len(), run_2.trace_len(), "trace length must match");
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
            op: RiscvOpcode::Slt,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 1 (signed compare)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sltu,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // x4 = 0 (unsigned compare)
        RiscvInstruction::Halt,
    ];
    let max_steps = program.len();
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(xlen)
        .chunk_rows(1)
        .min_trace_len(max_steps)
        .max_steps(max_steps)
        .shout_ops([RiscvOpcode::Add, RiscvOpcode::Slt, RiscvOpcode::Sltu])
        .reg_output_claim(/*reg=*/ 3, F::from_u64(1))
        .reg_output_claim(/*reg=*/ 4, F::from_u64(0))
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    let compare_rows: Vec<usize> = run
        .exec_table()
        .rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            matches!(
                row.decoded,
                Some(RiscvInstruction::RAlu {
                    op: RiscvOpcode::Slt | RiscvOpcode::Sltu,
                    ..
                })
            )
            .then_some(idx)
        })
        .collect();
    assert_eq!(compare_rows, vec![2, 3], "expected compare rows on SLT/SLTU steps");
}
