use std::collections::HashMap;

use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_decode_plumbing_sidecar_ccs, build_rv32_b1_semantics_sidecar_ccs, build_rv32_b1_step_ccs,
    estimate_rv32_b1_all_ccs_counts,
};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_b1_all_ccs_count_estimator_matches_built_ccs() {
    // Program: ADDI x1,x0,1; HALT
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

    let (prog_layout, _prog_init) = prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, &program_bytes)
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
            neo_memory::riscv::lookups::REG_ID.0,
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

    let (step_ccs, layout) =
        build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, /*chunk_size=*/ 1).expect("build_rv32_b1_step_ccs");
    let decode_ccs = build_rv32_b1_decode_plumbing_sidecar_ccs(&layout).expect("decode plumbing sidecar ccs");
    let semantics_ccs = build_rv32_b1_semantics_sidecar_ccs(&layout, &mem_layouts).expect("semantics sidecar ccs");

    let counts = estimate_rv32_b1_all_ccs_counts(&mem_layouts, &shout_table_ids, /*chunk_size=*/ 1)
        .expect("estimate_rv32_b1_all_ccs_counts");

    assert_eq!(counts.step.n, step_ccs.n);
    assert_eq!(counts.step.m, step_ccs.m);
    assert_eq!(counts.step.semantic + counts.step.injected, counts.step.n);

    assert_eq!(counts.decode_plumbing_n, decode_ccs.n);
    assert_eq!(counts.semantics_n, semantics_ccs.n);
}
