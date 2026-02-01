use std::collections::HashMap;

use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::build_rv32_b1_step_ccs;
use neo_memory::riscv::lookups::{
    encode_program, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use p3_goldilocks::Goldilocks as F;

#[test]
fn nightstream_single_addi_constraint_counts() {
    // Program: ADDI x1, x0, 1; HALT
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

    let (prog_layout, _prog_init) =
        prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, &program_bytes)
            .expect("prog_rom_layout_and_init_words");

    let mem_layouts = HashMap::from([
        (
            RAM_ID.0,
            PlainMemLayout {
                k: 4,
                d: 2,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            REG_ID.0,
            PlainMemLayout {
                k: 32,
                d: 5,
                n_side: 2,
                lanes: 2,
            },
        ),
        (PROG_ID.0, prog_layout),
    ]);

    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let shout_table_ids = vec![shout.opcode_to_id(RiscvOpcode::Add).0];

    let (ccs, _layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, /*chunk_size=*/ 1)
        .expect("build_rv32_b1_step_ccs");

    let nightstream_constraints = ccs.n;
    let nightstream_witness_cols = ccs.m;
    let nightstream_constraints_p2 = nightstream_constraints.next_power_of_two();
    let nightstream_witness_cols_p2 = nightstream_witness_cols.next_power_of_two();

    assert!(nightstream_constraints > 0);

    println!();
    println!(
        "{:<36} {:>4}   {:<14} {:>11} {:>12}   {}",
        "System", "XLEN", "Instruction", "Constraints", "Witness cols", "Notes"
    );
    println!("{}", "-".repeat(110));
    println!(
        "{:<36} {:>4}   {:<14} {:>11} {:>12}   shout_tables={}, constraints_p2={}, witness_cols_p2={}",
        "Nightstream (RV32 B1 step CCS)",
        32,
        "ADDI x1,x0,1",
        nightstream_constraints,
        nightstream_witness_cols,
        shout_table_ids.len(),
        nightstream_constraints_p2,
        nightstream_witness_cols_p2
    );
    println!();
}
