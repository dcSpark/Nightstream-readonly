use std::collections::HashMap;

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_semantics_sidecar_ccs, build_rv32_b1_step_ccs, rv32_b1_chunk_to_full_witness_checked,
};
use neo_memory::riscv::lookups::{
    encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn mem_layouts_for_program(program_bytes: &[u8]) -> HashMap<u32, PlainMemLayout> {
    let (prog_layout, _prog_init) = prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, program_bytes)
        .expect("prog_rom_layout_and_init_words");

    HashMap::from([
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
    ])
}

#[test]
fn rv32m_masked_columns_are_tied_to_real_witness() {
    // Program:
    //   ADDI x1,x0,3
    //   ADDI x2,x0,5
    //   MULH x3,x1,x2
    //   HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mem_layouts = mem_layouts_for_program(&program_bytes);

    // Minimal Shout set needed to execute the ADDI instructions above.
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let shout_table_ids = vec![shout.opcode_to_id(RiscvOpcode::Add).0];

    let (_main_ccs, layout) =
        build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, /*chunk_size=*/ 1).expect("build_rv32_b1_step_ccs");
    let semantics_ccs =
        build_rv32_b1_semantics_sidecar_ccs(&layout, &mem_layouts).expect("build_rv32_b1_semantics_sidecar_ccs");

    // Trace the program to obtain per-step events (PROG/REG/RAM + Shout).
    let mut cpu_vm = RiscvCpu::new(/*xlen=*/ 32);
    cpu_vm.load_program(/*base=*/ 0, program);
    let memory = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu_vm, memory, shout, /*max_steps=*/ 16).expect("trace_program");
    assert!(trace.did_halt(), "expected program to halt");
    assert!(
        trace.steps.len() >= 3,
        "expected at least 3 executed steps, got {}",
        trace.steps.len()
    );

    // Non-M row (ADDI): masked columns must be 0 and are enforced by semantics CCS.
    {
        let step = &trace.steps[0];
        let z = rv32_b1_chunk_to_full_witness_checked(&layout, core::slice::from_ref(step)).expect("witness");
        let (x, w) = z.split_at(layout.m_in);
        check_ccs_rowwise_zero(&semantics_ccs, x, w).expect("semantics CCS must accept honest witness");

        let mut z_bad = z.clone();
        z_bad[layout.rv32m_rs1_val(0)] = F::ONE;
        let (x_bad, w_bad) = z_bad.split_at(layout.m_in);
        assert!(
            check_ccs_rowwise_zero(&semantics_ccs, x_bad, w_bad).is_err(),
            "expected masking constraint failure on non-RV32M row"
        );
    }

    // M-sidecar row (MULH): masked columns must equal the real operands/output and are enforced by semantics CCS.
    {
        let step = &trace.steps[2];
        let z = rv32_b1_chunk_to_full_witness_checked(&layout, core::slice::from_ref(step)).expect("witness");
        let (x, w) = z.split_at(layout.m_in);
        check_ccs_rowwise_zero(&semantics_ccs, x, w).expect("semantics CCS must accept honest witness");

        let mut z_bad = z.clone();
        z_bad[layout.rv32m_rs1_val(0)] = F::ZERO;
        let (x_bad, w_bad) = z_bad.split_at(layout.m_in);
        assert!(
            check_ccs_rowwise_zero(&semantics_ccs, x_bad, w_bad).is_err(),
            "expected masking constraint failure on RV32M row"
        );
    }
}
