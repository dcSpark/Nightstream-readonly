//! Tests for the RV32 B1 shared-bus step CCS.

use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_ccs::traits::SModuleHomomorphism;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_step_ccs, rv32_b1_chunk_to_full_witness_checked, rv32_b1_chunk_to_witness,
    rv32_b1_shared_cpu_bus_config,
};
use neo_memory::riscv::lookups::{
    decode_instruction, encode_program, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory,
    RiscvOpcode, RiscvShoutTables, JOLT_CYCLE_TRACK_ECALL_NUM, JOLT_PRINT_ECALL_NUM, PROG_ID, RAM_ID,
    REG_ID,
};
use neo_memory::riscv::rom_init::prog_init_words;
use neo_memory::witness::LutTableSpec;
use neo_memory::{CpuArithmetization, R1csCpu};
use neo_params::NeoParams;
use neo_vm_trace::{trace_program, StepTrace, TwistEvent, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[derive(Clone, Copy, Default)]
struct NoopCommit;

impl SModuleHomomorphism<F, ()> for NoopCommit {
    fn commit(&self, _z: &Mat<F>) -> () {}

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn pow2_ceil_k(min_k: usize) -> (usize, usize) {
    let k = min_k.next_power_of_two().max(2);
    let d = k.trailing_zeros() as usize;
    (k, d)
}

fn with_reg_layout(mut mem_layouts: HashMap<u32, PlainMemLayout>) -> HashMap<u32, PlainMemLayout> {
    mem_layouts.insert(
        REG_ID.0,
        PlainMemLayout {
            k: 32,
            d: 5,
            n_side: 2,
            lanes: 2,
        },
    );
    mem_layouts
}

fn load_u32_imm(rd: u8, value: u32) -> Vec<RiscvInstruction> {
    let upper = ((value as i64 + 0x800) >> 12) as i32;
    let lower = (value as i32) - (upper << 12);
    vec![
        RiscvInstruction::Lui { rd, imm: upper },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd,
            rs1: rd,
            imm: lower,
        },
    ]
}

const RV32I_SHOUT_TABLE_IDS: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

fn rv32i_table_specs(xlen: usize) -> HashMap<u32, LutTableSpec> {
    HashMap::from([
        (
            0u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::And,
                xlen,
            },
        ),
        (
            1u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Xor,
                xlen,
            },
        ),
        (
            2u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Or,
                xlen,
            },
        ),
        (
            3u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            4u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sub,
                xlen,
            },
        ),
        (
            5u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Slt,
                xlen,
            },
        ),
        (
            6u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sltu,
                xlen,
            },
        ),
        (
            7u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sll,
                xlen,
            },
        ),
        (
            8u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Srl,
                xlen,
            },
        ),
        (
            9u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sra,
                xlen,
            },
        ),
        (
            10u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Eq,
                xlen,
            },
        ),
        (
            11u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Neq,
                xlen,
            },
        ),
    ])
}

#[test]
fn rv32_b1_ccs_happy_path_small_program() {
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

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    // mem_layouts: keep k small to reduce bus tail width.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200); // covers addresses up to 0x1ff
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            2u32,
            PlainMemLayout {
                k: 32,
                d: 5,
                n_side: 2,
                lanes: 2,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32i_fence_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::Fence { pred: 0, succ: 0 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 2,
        }, // x2 = 3
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let regs = &trace.steps.last().expect("steps").regs_after;
    assert_eq!(regs[1], 1, "ADD before FENCE");
    assert_eq!(regs[2], 3, "ADD after FENCE");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32i_ecall_markers_program() {
    let xlen = 32usize;
    let mut program = Vec::new();
    program.extend(load_u32_imm(10, JOLT_CYCLE_TRACK_ECALL_NUM));
    program.push(RiscvInstruction::Halt);
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 1,
        rs1: 0,
        imm: 7,
    });
    program.extend(load_u32_imm(10, JOLT_PRINT_ECALL_NUM));
    program.push(RiscvInstruction::Halt);
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 2,
        rs1: 1,
        imm: 1,
    });
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 10,
        rs1: 0,
        imm: 0,
    });
    program.push(RiscvInstruction::Halt);

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let regs = &trace.steps.last().expect("steps").regs_after;
    assert_eq!(regs[1], 7, "instruction after ECALL marker executes");
    assert_eq!(regs[2], 8, "instruction after ECALL print executes");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32m_program() {
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
            op: RiscvOpcode::Divu,
            rd: 4,
            rs1: 3,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Minimal table set for this program:
    // - ADD (address/ALU wiring + ADDI),
    // - SLTU (DIVU/REMU remainder bound check).
    let shout_tables = RiscvShoutTables::new(xlen);
    let add_id = shout_tables.opcode_to_id(RiscvOpcode::Add).0;
    let sltu_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu).0;
    let shout_table_ids: [u32; 2] = [add_id, sltu_id];
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = HashMap::from([
        (
            add_id,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            sltu_id,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sltu,
                xlen,
            },
        ),
    ]);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32m_signed_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0,
        }, // x1 = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: -3,
        }, // x2 = -3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 0 / -3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // x4 = 0 % -3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: -4,
        }, // x5 = -4
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 2,
        }, // x6 = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 7,
            rs1: 5,
            rs2: 6,
        }, // x7 = -4 % 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 8,
            rs1: 5,
            rs2: 6,
        }, // x8 = -4 / 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 9,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 10,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 11,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 12,
            rs1: 5,
            rs2: 0,
        }, // div by zero
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 13,
            rs1: 5,
            rs2: 0,
        }, // rem by zero
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let regs = &trace.steps.last().expect("steps").regs_after;
    assert_eq!(regs[3], 0, "DIV 0 / -3");
    assert_eq!(regs[4], 0, "REM 0 % -3");
    assert_eq!(regs[7], 0, "REM -4 % 2");
    assert_eq!(regs[8], 0xffff_fffe, "DIV -4 / 2");
    assert_eq!(regs[9], 0xffff_ffff, "MULH -4 * 2");
    assert_eq!(regs[10], 0xffff_ffff, "MULHSU -4 * 2");
    assert_eq!(regs[11], 0x0000_0001, "MULHU 0xffff_fffc * 2");
    assert_eq!(regs[12], 0xffff_ffff, "DIV by zero returns -1");
    assert_eq!(regs[13], 0xffff_fffc, "REM by zero returns dividend");

    let shout_tables = RiscvShoutTables::new(xlen);
    let sltu_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu).0;
    for &idx in &[2usize, 3, 6, 7] {
        let events = &trace.steps[idx].shout_events;
        assert_eq!(events.len(), 1, "expected SLTU lookup at step {idx}");
        assert_eq!(events[0].shout_id.0, sltu_id, "expected SLTU table id at step {idx}");
    }
    for &idx in &[11usize, 12] {
        assert!(
            trace.steps[idx].shout_events.is_empty(),
            "expected no lookup for div/rem by zero at step {idx}"
        );
    }

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let add_id = shout_tables.opcode_to_id(RiscvOpcode::Add).0;
    let shout_table_ids: [u32; 2] = [add_id, sltu_id];
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = HashMap::from([
        (
            add_id,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            sltu_id,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sltu,
                xlen,
            },
        ),
    ]);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_witness_bus_alu_step() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");

    let step = trace.steps.first().expect("step").clone();
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (_ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let z = rv32_b1_chunk_to_full_witness_checked(&layout, std::slice::from_ref(&step)).expect("witness");

    let shout_tables = RiscvShoutTables::new(xlen);
    let add_id = shout_tables.opcode_to_id(RiscvOpcode::Add).0;
    let add_idx = layout.shout_idx(add_id).expect("add idx");
    let add_lane = &layout.bus.shout_cols[add_idx].lanes[0];
    let prog_lane = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let ram_lane = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];

    assert_eq!(z[layout.bus.bus_cell(prog_lane.has_read, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(prog_lane.has_write, 0)], F::ZERO);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_read, 0)], F::ZERO);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_write, 0)], F::ZERO);
    assert_eq!(z[layout.bus.bus_cell(add_lane.has_lookup, 0)], F::ONE);

    let shout_ev = step.shout_events.first().expect("shout event");
    assert_eq!(z[layout.bus.bus_cell(add_lane.val, 0)], F::from_u64(shout_ev.value));
    assert_eq!(z[layout.lookup_key(0)], F::ZERO);
    assert_eq!(z[layout.alu_out(0)], F::from_u64(shout_ev.value));
    for (bit_idx, col_id) in add_lane.addr_bits.clone().enumerate() {
        let bit = if bit_idx < 64 { (shout_ev.key >> bit_idx) & 1 } else { 0 };
        let expected = if bit == 1 { F::ONE } else { F::ZERO };
        assert_eq!(z[layout.bus.bus_cell(col_id, 0)], expected);
    }
}

#[test]
fn rv32_b1_witness_bus_lw_step() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 2,
            imm: 0,
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");

    let step = trace.steps.get(2).expect("lw step").clone();
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (_ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let z = rv32_b1_chunk_to_full_witness_checked(&layout, std::slice::from_ref(&step)).expect("witness");

    let shout_tables = RiscvShoutTables::new(xlen);
    let add_id = shout_tables.opcode_to_id(RiscvOpcode::Add).0;
    let add_idx = layout.shout_idx(add_id).expect("add idx");
    let add_lane = &layout.bus.shout_cols[add_idx].lanes[0];
    let shout_ev = step.shout_events.first().expect("shout event");

    let ram_lane = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];
    let ram_read = step
        .twist_events
        .iter()
        .find(|ev| ev.twist_id == RAM_ID && ev.kind == TwistOpKind::Read)
        .expect("ram read");

    assert_eq!(z[layout.bus.bus_cell(add_lane.has_lookup, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(add_lane.val, 0)], F::from_u64(shout_ev.value));
    // RV32 B1 no longer binds the raw 64-bit Shout key into a single field element. The authoritative
    // witness data is in the ADD lane addr_bits.
    assert_eq!(z[layout.lookup_key(0)], F::ZERO);
    assert_eq!(z[layout.alu_out(0)], F::from_u64(shout_ev.value));
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_read, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_write, 0)], F::ZERO);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.rv, 0)], F::from_u64(ram_read.value));
    assert_eq!(z[layout.mem_rv(0)], F::from_u64(ram_read.value));
    assert_eq!(z[layout.eff_addr(0)], F::from_u64(ram_read.addr));

    for (bit_idx, col_id) in ram_lane.ra_bits.clone().enumerate() {
        let bit = if bit_idx < 64 {
            (ram_read.addr >> bit_idx) & 1
        } else {
            0
        };
        let expected = if bit == 1 { F::ONE } else { F::ZERO };
        assert_eq!(z[layout.bus.bus_cell(col_id, 0)], expected);
    }
}

#[test]
fn rv32_b1_witness_bus_amoaddw_step() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 3,
            rs1: 0,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");

    let step = trace.steps.get(3).expect("amoaddw step").clone();
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (_ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let z = rv32_b1_chunk_to_full_witness_checked(&layout, std::slice::from_ref(&step)).expect("witness");

    let shout_tables = RiscvShoutTables::new(xlen);
    let add_id = shout_tables.opcode_to_id(RiscvOpcode::Add).0;
    let add_idx = layout.shout_idx(add_id).expect("add idx");
    let add_lane = &layout.bus.shout_cols[add_idx].lanes[0];
    let shout_ev = step.shout_events.first().expect("shout event");

    let ram_lane = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];
    let ram_read = step
        .twist_events
        .iter()
        .find(|ev| ev.twist_id == RAM_ID && ev.kind == TwistOpKind::Read)
        .expect("ram read");
    let ram_write = step
        .twist_events
        .iter()
        .find(|ev| ev.twist_id == RAM_ID && ev.kind == TwistOpKind::Write)
        .expect("ram write");

    assert_eq!(z[layout.bus.bus_cell(add_lane.has_lookup, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(add_lane.val, 0)], F::from_u64(shout_ev.value));
    assert_eq!(z[layout.lookup_key(0)], F::ZERO);
    assert_eq!(z[layout.alu_out(0)], F::from_u64(shout_ev.value));
    for (bit_idx, col_id) in add_lane.addr_bits.clone().enumerate() {
        let bit = if bit_idx < 64 { (shout_ev.key >> bit_idx) & 1 } else { 0 };
        let expected = if bit == 1 { F::ONE } else { F::ZERO };
        assert_eq!(z[layout.bus.bus_cell(col_id, 0)], expected);
    }
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_read, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.has_write, 0)], F::ONE);
    assert_eq!(z[layout.bus.bus_cell(ram_lane.rv, 0)], F::from_u64(ram_read.value));
    assert_eq!(z[layout.bus.bus_cell(ram_lane.wv, 0)], F::from_u64(ram_write.value));
    assert_eq!(z[layout.mem_rv(0)], F::from_u64(ram_read.value));
    assert_eq!(z[layout.ram_wv(0)], F::from_u64(ram_write.value));

    for (bit_idx, col_id) in ram_lane.ra_bits.clone().enumerate() {
        let bit = if bit_idx < 64 {
            (ram_read.addr >> bit_idx) & 1
        } else {
            0
        };
        let expected = if bit == 1 { F::ONE } else { F::ZERO };
        assert_eq!(z[layout.bus.bus_cell(col_id, 0)], expected);
    }
    for (bit_idx, col_id) in ram_lane.wa_bits.clone().enumerate() {
        let bit = if bit_idx < 64 {
            (ram_write.addr >> bit_idx) & 1
        } else {
            0
        };
        let expected = if bit == 1 { F::ONE } else { F::ZERO };
        assert_eq!(z[layout.bus.bus_cell(col_id, 0)], expected);
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32i_byte_half_load_store_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // x1 = 0x100
        RiscvInstruction::Lui { rd: 2, imm: 0x11223 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 2,
            rs1: 2,
            imm: 0x344,
        }, // x2 = 0x11223344
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 0x11223344
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 0xAA,
        }, // x6 = 0xAA
        RiscvInstruction::Store {
            op: RiscvMemOp::Sb,
            rs1: 1,
            rs2: 6,
            imm: 1,
        }, // mem[0x101] = 0xAA
        RiscvInstruction::Load {
            op: RiscvMemOp::Lb,
            rd: 7,
            rs1: 1,
            imm: 1,
        }, // x7 = signext(0xAA)
        RiscvInstruction::Load {
            op: RiscvMemOp::Lbu,
            rd: 8,
            rs1: 1,
            imm: 1,
        }, // x8 = 0xAA
        RiscvInstruction::Lui { rd: 9, imm: 0x8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 9,
            rs1: 9,
            imm: 1,
        }, // x9 = 0x8001
        RiscvInstruction::Store {
            op: RiscvMemOp::Sh,
            rs1: 1,
            rs2: 9,
            imm: 0,
        }, // mem[0x100..] = 0x8001
        RiscvInstruction::Load {
            op: RiscvMemOp::Lh,
            rd: 10,
            rs1: 1,
            imm: 0,
        }, // x10 = signext(0x8001)
        RiscvInstruction::Load {
            op: RiscvMemOp::Lhu,
            rd: 11,
            rs1: 1,
            imm: 0,
        }, // x11 = 0x8001
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 12,
            rs1: 1,
            imm: 0,
        }, // x12 = 0x11228001
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let regs = &trace.steps.last().expect("steps").regs_after;
    assert_eq!(regs[7], 0xffff_ffaa, "LB sign-extends 0xAA");
    assert_eq!(regs[8], 0x0000_00aa, "LBU zero-extends 0xAA");
    assert_eq!(regs[10], 0xffff_8001, "LH sign-extends 0x8001");
    assert_eq!(regs[11], 0x0000_8001, "LHU zero-extends 0x8001");
    assert_eq!(regs[12], 0x1122_8001, "LW reads merged word");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_byte_store_updates_aligned_word() {
    let xlen = 32usize;
    let mut program = Vec::new();
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 1,
        rs1: 0,
        imm: 0x100,
    }); // x1 = 0x100
    program.extend(load_u32_imm(2, 0x1122_3344)); // x2 = 0x11223344
    program.push(RiscvInstruction::Store {
        op: RiscvMemOp::Sw,
        rs1: 1,
        rs2: 2,
        imm: 0,
    }); // mem[0x100] = 0x11223344
    program.push(RiscvInstruction::Load {
        op: RiscvMemOp::Lb,
        rd: 3,
        rs1: 1,
        imm: 1,
    }); // x3 = mem[0x101] = 0x33
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 4,
        rs1: 0,
        imm: 0xAA,
    }); // x4 = 0xAA
    program.push(RiscvInstruction::Store {
        op: RiscvMemOp::Sb,
        rs1: 1,
        rs2: 4,
        imm: 1,
    }); // mem[0x101] = 0xAA
    program.push(RiscvInstruction::Load {
        op: RiscvMemOp::Lw,
        rd: 5,
        rs1: 1,
        imm: 0,
    }); // x5 = 0x1122AA44
    program.push(RiscvInstruction::Halt);

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let regs = &trace.steps.last().expect("steps").regs_after;
    assert_eq!(regs[3], 0x0000_0033, "LB reads byte from 0x11223344 at +1");
    assert_eq!(regs[5], 0x1122_aa44, "SB updates the aligned LW word");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rejects_misaligned_lh() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lh,
            rd: 1,
            rs1: 0,
            imm: 0x101, // misaligned (addr % 2 != 0)
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);
    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mcs_wit) = steps.remove(0);
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "misaligned LH should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_misaligned_lw() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0x102, // misaligned (addr % 4 != 0)
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);
    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mcs_wit) = steps.remove(0);
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "misaligned LW should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_misaligned_sh() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Store {
            op: RiscvMemOp::Sh,
            rs1: 0,
            rs2: 0,
            imm: 0x101, // misaligned (addr % 2 != 0)
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);
    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mcs_wit) = steps.remove(0);
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "misaligned SH should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_misaligned_sw() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 0,
            imm: 0x102, // misaligned (addr % 4 != 0)
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);
    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mcs_wit) = steps.remove(0);
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "misaligned SW should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_happy_path_rv32a_amoaddw_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // x1 = 0x100
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        }, // x2 = 10
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        }, // x3 = 5
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        }, // x4 = old, mem = old + 5
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 0,
        }, // x5 = new
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rejects_tampered_ram_write_value_for_amoaddw() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let amo_step_idx = 4usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(amo_step_idx);

    let ram_wv_w_idx = layout
        .ram_wv
        .checked_sub(layout.m_in)
        .expect("ram_wv must be in private witness");
    mcs_wit.w[ram_wv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered RAM write value should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_happy_path_rv32a_word_amos_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // x1 = 0x100
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0b1111,
        }, // x2 = 15
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 15
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 0b1010,
        }, // x3 = 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoandW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        }, // mem &= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoorW,
            rd: 5,
            rs1: 1,
            rs2: 3,
        }, // mem |= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoxorW,
            rd: 6,
            rs1: 1,
            rs2: 3,
        }, // mem ^= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoswapW,
            rd: 7,
            rs1: 1,
            rs2: 2,
        }, // mem = 15
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 8,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_chunk_size_2_padding_carries_state() {
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
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    for (mcs_inst, mcs_wit) in chunks {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_branches_and_jal() {
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
            imm: 1,
        }, // x2 = 1
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 99,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        }, // x3 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 77,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 7,
        }, // x4 = 7
        RiscvInstruction::Jal { rd: 5, imm: 8 }, // x5=pc+4, jump over next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 123,
        }, // skipped
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rv32i_alu_ops() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        }, // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 - x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::And,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 5,
            rs1: 1,
            imm: 0x0f,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 6,
            rs1: 1,
            imm: 0x0f,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 11,
            rs1: 2,
            rs2: 1,
        }, // (3 < 5) => 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 12,
            rs1: 2,
            imm: 4,
        }, // (3 < 4) => 1
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_branches_blt_bge_bltu_bgeu() {
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
            cond: BranchCondition::Lt,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 111,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 3,
        }, // x3 = 3
        RiscvInstruction::Branch {
            cond: BranchCondition::Ge,
            rs1: 2,
            rs2: 1,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 222,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 4,
        }, // x4 = 4
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: -1,
        }, // x5 = 0xffff_ffff
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 1,
        }, // x6 = 1
        RiscvInstruction::Branch {
            cond: BranchCondition::Ltu,
            rs1: 5,
            rs2: 6,
            imm: 8,
        }, // not taken -> execute next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 7,
            rs1: 0,
            imm: 7,
        }, // x7 = 7
        RiscvInstruction::Branch {
            cond: BranchCondition::Geu,
            rs1: 5,
            rs2: 6,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 8,
            rs1: 0,
            imm: 888,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 8,
            rs1: 0,
            imm: 8,
        }, // x8 = 8
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_jalr_masks_lsb() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 13,
        }, // x1 = 13
        RiscvInstruction::Jalr { rd: 0, rs1: 1, imm: 0 }, // pc = (13 + 0) & !1 = 12
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 9,
        }, // executed
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rejects_step_after_halt_within_chunk() {
    let xlen = 32usize;
    let program = vec![RiscvInstruction::Halt, RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let w0 = u32::from_le_bytes(program_bytes[0..4].try_into().expect("word0"));
    let w1 = u32::from_le_bytes(program_bytes[4..8].try_into().expect("word1"));

    let regs = vec![0u64; 32];
    let steps: Vec<StepTrace<u64, u64>> = vec![
        StepTrace {
            cycle: 0,
            pc_before: 0,
            pc_after: 4,
            opcode: w0,
            regs_before: regs.clone(),
            regs_after: regs.clone(),
            twist_events: vec![
                TwistEvent {
                    twist_id: PROG_ID,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: w0 as u64,
                    lane: None,
                },
                TwistEvent {
                    twist_id: REG_ID,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: 0,
                    lane: Some(0),
                },
                TwistEvent {
                    twist_id: REG_ID,
                    kind: TwistOpKind::Read,
                    addr: 10,
                    value: 0,
                    lane: Some(1),
                },
            ],
            shout_events: Vec::new(),
            halted: true,
        },
        StepTrace {
            cycle: 1,
            pc_before: 4,
            pc_after: 8,
            opcode: w1,
            regs_before: regs.clone(),
            regs_after: regs.clone(),
            twist_events: vec![
                TwistEvent {
                    twist_id: PROG_ID,
                    kind: TwistOpKind::Read,
                    addr: 4,
                    value: w1 as u64,
                    lane: None,
                },
                TwistEvent {
                    twist_id: REG_ID,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: 0,
                    lane: Some(0),
                },
                TwistEvent {
                    twist_id: REG_ID,
                    kind: TwistOpKind::Read,
                    addr: 10,
                    value: 0,
                    lane: Some(1),
                },
            ],
            shout_events: Vec::new(),
            halted: true,
        },
    ];
    let trace = VmTrace { steps };

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("chunks");
    assert_eq!(chunks.len(), 1, "expected single chunk");
    let (mcs_inst, mcs_wit) = chunks.pop().expect("chunk");
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "step after HALT should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_pc_out() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let pc_out_w_idx = layout
        .pc_out
        .checked_sub(layout.m_in)
        .expect("pc_out must be in private witness");
    mcs_wit.w[pc_out_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered witness should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_non_boolean_prog_read_addr_bit() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let bit_col_id = prog_cols.ra_bits.start + 2; // keep alignment bits [0,1] untouched
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("prog bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    let new_bit = F::from_u64(2);
    mcs_wit.w[bit_w_idx] = new_bit;

    let delta = new_bit - old_bit;
    let pc_in_w_idx = layout
        .pc_in
        .checked_sub(layout.m_in)
        .expect("pc_in must be in private witness");
    let pc_out_w_idx = layout
        .pc_out
        .checked_sub(layout.m_in)
        .expect("pc_out must be in private witness");
    mcs_wit.w[pc_in_w_idx] += delta * F::from_u64(1 << 2);
    mcs_wit.w[pc_out_w_idx] += delta * F::from_u64(1 << 2);

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "non-boolean prog addr bit should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_non_boolean_shout_addr_bit() {
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
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (Shout ADD active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let instr = decode_instruction(trace.steps[add_step_idx].opcode).expect("decode");
    match instr {
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add, ..
        } => {}
        other => panic!("expected ADD at step {add_step_idx}, got {other:?}"),
    }

    // Flip one ADD shout key bit to a non-boolean value, and adjust CPU columns so that
    // all *linear* bindings still hold. Bitness constraints should still reject.
    let add_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Add).0;
    let add_shout_idx = layout.shout_idx(add_id).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // key bit 0 (rs1 bit 0)
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    let new_bit = F::from_u64(2);
    mcs_wit.w[bit_w_idx] = new_bit;
    let delta = new_bit - old_bit;

    // Update packed key (lookup_key) to match the mutated bits.
    let lookup_key_w_idx = layout
        .lookup_key(0)
        .checked_sub(layout.m_in)
        .expect("lookup_key must be in private witness");
    mcs_wit.w[lookup_key_w_idx] += delta;

    // Update rs1_val to match the mutated even-bit packing.
    let rs1_val_w_idx = layout
        .rs1_val(0)
        .checked_sub(layout.m_in)
        .expect("rs1_val must be in private witness");
    mcs_wit.w[rs1_val_w_idx] += delta;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "non-boolean shout addr bit should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_rom_value_mismatch() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let rv_z = layout.bus.bus_cell(prog_cols.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("rv in witness");
    mcs_wit.w[rv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "rom value mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_regfile() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    // Tamper with the regfile (REG_ID) lane0 read value without updating `rs1_val`.
    let reg_lane0 = &layout.bus.twist_cols[layout.reg_twist_idx].lanes[0];
    let rv_z = layout.bus.bus_cell(reg_lane0.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("regfile rv in witness");
    mcs_wit.w[rv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered regfile should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_x0() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let reg_lane0 = &layout.bus.twist_cols[layout.reg_twist_idx].lanes[0];
    let rv_z = layout.bus.bus_cell(reg_lane0.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("regfile rv in witness");
    mcs_wit.w[rv_w_idx] = F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered x0 should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_binds_public_initial_and_final_state() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        }, // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // x2 = 7
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 8usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    assert_eq!(chunks.len(), 1, "chunk_size>N should create one chunk");
    let (mcs_inst, mcs_wit) = chunks.remove(0);

    check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");

    let first = trace.steps.first().expect("trace non-empty");
    assert_eq!(mcs_inst.x[layout.pc0], F::from_u64(first.pc_before));

    let last = trace.steps.last().expect("trace non-empty");
    assert_eq!(mcs_inst.x[layout.pc_final], F::from_u64(last.pc_after));

    let mut x_bad = mcs_inst.x.clone();
    x_bad[layout.pc0] += F::ONE;
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &x_bad, &mcs_wit.w).is_err(),
        "tampered pc0 should not satisfy CCS"
    );

    let mut x_bad = mcs_inst.x.clone();
    x_bad[layout.pc_final] += F::ONE;
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &x_bad, &mcs_wit.w).is_err(),
        "tampered pc_final should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_rom_addr_mismatch() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let bit_col_id = prog_cols.ra_bits.start + 2; // keep alignment bits [0,1] untouched
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("prog bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "rom address mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_decode_bit_mismatch() {
    let xlen = 32usize;
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
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let bit_z = layout.instr_bit(0, 0);
    let bit_w_idx = bit_z
        .checked_sub(layout.m_in)
        .expect("instr_bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "decode bit mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch() {
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
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (Shout ADD active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let add_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Add).0;
    let add_shout_idx = layout.shout_idx(add_id).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // rs1 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_lw_eff_addr() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let lw_step_idx = 0usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(lw_step_idx);

    let add_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Add).0;
    let add_shout_idx = layout.shout_idx(add_id).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (LW effective address) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_amoaddw() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 3,
            rs1: 0,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let amoadd_step_idx = 3usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(amoadd_step_idx);

    let add_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Add).0;
    let add_shout_idx = layout.shout_idx(add_id).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0 (mem_rv bit 0)
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (AMOADD.W operands) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_beq() {
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
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // not taken -> execute HALT
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(4);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let beq_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(beq_step_idx);

    let eq_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Eq).0;
    let eq_shout_idx = layout.shout_idx(eq_id).expect("EQ shout idx");
    let shout_cols = &layout.bus.shout_cols[eq_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (BEQ operands) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_bne() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        }, // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        }, // x2 = 5
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // not taken -> execute next
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 32).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let bne_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(bne_step_idx);

    let neq_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Neq).0;
    let neq_shout_idx = layout.shout_idx(neq_id).expect("NEQ shout idx");
    let shout_cols = &layout.bus.shout_cols[neq_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (BNE operands) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_ori() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x123,
        }, // x1 = 0x123
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 2,
            rs1: 1,
            imm: 0x55,
        }, // x2 = x1 | 0x55 (Shout OR active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let ori_step_idx = 1usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(ori_step_idx);

    let or_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Or).0;
    let or_shout_idx = layout.shout_idx(or_id).expect("OR shout idx");
    let shout_cols = &layout.bus.shout_cols[or_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (ORI imm) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_slli() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 2,
            rs1: 1,
            imm: 3,
        }, // x2 = x1 << 3 (Shout SLL active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let slli_step_idx = 1usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(slli_step_idx);

    let sll_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Sll).0;
    let sll_shout_idx = layout.shout_idx(sll_id).expect("SLL shout idx");
    let shout_cols = &layout.bus.shout_cols[sll_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (SLLI imm) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_sltu_key_bit_mismatch_divu_remainder_check() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 10,
        }, // x1 = 10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 / x2 (sltu(rem, divisor) lookup when divisor != 0)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let divu_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(divu_step_idx);

    let sltu_id = RiscvShoutTables::new(xlen)
        .opcode_to_id(RiscvOpcode::Sltu)
        .0;
    let sltu_shout_idx = layout.shout_idx(sltu_id).expect("SLTU shout idx");
    let shout_cols = &layout.bus.shout_cols[sltu_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0 (remainder bit 0)
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "sltu(rem, divisor) shout key mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch_auipc_pc_operand() {
    let xlen = 32usize;
    let program = vec![RiscvInstruction::Auipc { rd: 1, imm: 0 }, RiscvInstruction::Halt];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let auipc_step_idx = 0usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(auipc_step_idx);

    let add_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Add).0;
    let add_shout_idx = layout.shout_idx(add_id).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let bit_col_id = shout_cols.addr_bits.start + 0; // operand0 bit 0 (pc bit 0)
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch (AUIPC pc operand) should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_cheating_mul_hi_all_ones() {
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
            imm: 1,
        }, // x2 = 1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 1 * 1
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let mul_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(mul_step_idx);

    let mul_hi = u32::MAX as u64;
    let mul_lo = 2u64;

    let mul_hi_z = layout.mul_hi(0);
    let mul_hi_w = mul_hi_z
        .checked_sub(layout.m_in)
        .expect("mul_hi in witness");
    mcs_wit.w[mul_hi_w] = F::from_u64(mul_hi);

    let mul_lo_z = layout.mul_lo(0);
    let mul_lo_w = mul_lo_z
        .checked_sub(layout.m_in)
        .expect("mul_lo in witness");
    mcs_wit.w[mul_lo_w] = F::from_u64(mul_lo);

    let rd_write_z = layout.rd_write_val(0);
    let rd_write_w = rd_write_z
        .checked_sub(layout.m_in)
        .expect("rd_write_val in witness");
    mcs_wit.w[rd_write_w] = F::from_u64(mul_lo);

    let reg_lane0 = &layout.bus.twist_cols[layout.reg_twist_idx].lanes[0];
    let wv_z = layout.bus.bus_cell(reg_lane0.wv, 0);
    let wv_w = wv_z.checked_sub(layout.m_in).expect("regfile wv in witness");
    mcs_wit.w[wv_w] = F::from_u64(mul_lo);

    // Make the u32 bit decompositions consistent with the cheated values.
    for bit in 0..32 {
        let hi_bit_z = layout.mul_hi_bit(bit, 0);
        let hi_bit_w = hi_bit_z
            .checked_sub(layout.m_in)
            .expect("mul_hi_bit in witness");
        mcs_wit.w[hi_bit_w] = F::ONE;

        let lo_bit = (mul_lo >> bit) & 1;
        let lo_bit_z = layout.mul_lo_bit(bit, 0);
        let lo_bit_w = lo_bit_z
            .checked_sub(layout.m_in)
            .expect("mul_lo_bit in witness");
        mcs_wit.w[lo_bit_w] = if lo_bit == 1 { F::ONE } else { F::ZERO };

        let rd_bit_z = layout.rd_write_bit(bit, 0);
        let rd_bit_w = rd_bit_z
            .checked_sub(layout.m_in)
            .expect("rd_write_bit in witness");
        mcs_wit.w[rd_bit_w] = if lo_bit == 1 { F::ONE } else { F::ZERO };
    }
    for k in 0..31 {
        let prefix_z = layout.mul_hi_prefix(k, 0);
        let prefix_w = prefix_z
            .checked_sub(layout.m_in)
            .expect("mul_hi_prefix in witness");
        mcs_wit.w[prefix_w] = F::ONE;
    }

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "cheating MUL decomposition should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_wrong_shout_table_activation() {
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
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (ADD table active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let eq_id = RiscvShoutTables::new(xlen).opcode_to_id(RiscvOpcode::Eq).0;
    let eq_shout_idx = layout.shout_idx(eq_id).expect("EQ shout idx");
    let eq_cols = &layout.bus.shout_cols[eq_shout_idx].lanes[0];
    let has_lookup_z = layout.bus.bus_cell(eq_cols.has_lookup, 0);
    let has_lookup_w_idx = has_lookup_z
        .checked_sub(layout.m_in)
        .expect("has_lookup in witness");
    mcs_wit.w[has_lookup_w_idx] = F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "wrong shout table activation should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_ram_read_value_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 9,
        }, // x1 = 9
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0x100,
        }, // mem[0x100] = x1
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0x100,
        }, // x2 = mem[0x100]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let lw_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(lw_step_idx);

    let ram_cols = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];
    let rv_z = layout.bus.bus_cell(ram_cols.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("rv in witness");
    mcs_wit.w[rv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "ram read value mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_chunk_size_2_continuity_break() {
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
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = with_reg_layout(HashMap::from([
        (
            0u32,
            PlainMemLayout {
                k: k_ram,
                d: d_ram,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            1u32,
            PlainMemLayout {
                k: k_prog,
                d: d_prog,
                n_side: 2,
                lanes: 1,
            },
        ),
    ]));
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    let (mcs_inst, mut mcs_wit) = chunks.remove(0);

    let pc_in_w_idx = layout
        .pc_in(1)
        .checked_sub(layout.m_in)
        .expect("pc_in lane 1 in witness");
    mcs_wit.w[pc_in_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "continuity break should not satisfy CCS"
    );
}
