#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

fn mul_program() -> Vec<RiscvInstruction> {
    vec![
        // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        // x2 = 13
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 13,
        },
        // x3 = x1 * x2 (MUL: low 32 bits = 91)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = MULHU(x1, x2) (high 32 bits, should be 0 for small values)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        // x5 = x3 * x1 (91 * 7 = 637)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 5,
            rs1: 3,
            rs2: 1,
        },
        // Load large negative: x6 = -1 (0xFFFFFFFF)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: -1,
        },
        // x7 = MULH(-1, 13) = signed high word
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 7,
            rs1: 6,
            rs2: 2,
        },
        // x8 = MULHSU(-1, 13) = signed*unsigned high word
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 8,
            rs1: 6,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ]
}

#[test]
#[ignore = "integration: cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_mul_prove_verify"]
fn debug_mul_prove_verify() {
    let program = mul_program();
    let program_bytes = encode_program(&program);
    let steps = program.len();

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .chunk_rows(steps)
        .max_steps(steps)
        .prove()
        .expect("MUL trace prove");

    println!(
        "MUL prove ok: ccs_n={} ccs_m={} trace_len={} folds={} prove={:?}",
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.trace_len(),
        run.fold_count(),
        run.prove_duration(),
    );

    run.verify().expect("MUL trace verify");
    println!("MUL verify ok: {:?}", run.verify_duration());

    let used_tables = run.used_shout_table_ids();
    println!("Used shout table IDs: {used_tables:?}");
    assert!(
        used_tables.contains(&neo_memory::riscv::mul_decomp::MUL8_TABLE_ID),
        "Mul8 table should be registered"
    );
    assert!(
        used_tables.contains(&neo_memory::riscv::mul_decomp::ADD8ACC_TABLE_ID),
        "Add8Acc table should be registered"
    );
}
