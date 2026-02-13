use std::collections::HashMap;

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::addr::write_addr_bits_dim_major_le_into_bus;
use neo_memory::cpu::extend_ccs_with_shared_cpu_bus_constraints;
use neo_memory::mem_init::MemInit;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{build_rv32_b1_step_ccs, rv32_b1_chunk_to_witness_checked, rv32_b1_shared_cpu_bus_config};
use neo_memory::riscv::lookups::{
    encode_instruction, encode_program, RiscvCpu, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID,
    REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_memory::witness::{LutInstance, MemInstance};
use neo_vm_trace::{trace_program, Twist, TwistId};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[derive(Clone, Debug, Default)]
struct HashMapTwist {
    data: HashMap<(TwistId, u64), u64>,
}

impl HashMapTwist {
    fn set(&mut self, twist_id: TwistId, addr: u64, value: u64) {
        self.data.insert((twist_id, addr), value);
    }
}

impl Twist<u64, u64> for HashMapTwist {
    fn load(&mut self, twist_id: TwistId, addr: u64) -> u64 {
        self.data.get(&(twist_id, addr)).copied().unwrap_or(0)
    }

    fn store(&mut self, twist_id: TwistId, addr: u64, value: u64) {
        self.data.insert((twist_id, addr), value);
    }
}

fn fill_bus_tail_from_step_events(
    z: &mut [F],
    bus: &neo_memory::cpu::BusLayout,
    step: &neo_vm_trace::StepTrace<u64, u64>,
    table_ids: &[u32],
    mem_ids: &[u32],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) {
    // Shout (single-lane per table in these tests).
    for ev in &step.shout_events {
        let id = ev.shout_id.0;
        let idx = table_ids
            .binary_search(&id)
            .unwrap_or_else(|_| panic!("unexpected shout_id={id}"));
        let cols = &bus.shout_cols[idx].lanes[0];
        // RV32 opcode tables: d=2*xlen=64, n_side=2, ell=1.
        write_addr_bits_dim_major_le_into_bus(z, bus, cols.addr_bits.clone(), /*j=*/ 0, ev.key, 64, 2, 1);
        z[bus.bus_cell(cols.has_lookup, 0)] = F::ONE;
        z[bus.bus_cell(cols.val, 0)] = F::from_u64(ev.value);
    }

    // Twist reads/writes (lane-pinned for REG_ID, lane0 otherwise).
    let mut reads: Vec<Vec<Option<(u64, u64)>>> = bus
        .twist_cols
        .iter()
        .map(|inst| vec![None; inst.lanes.len()])
        .collect();
    let mut writes: Vec<Vec<Option<(u64, u64)>>> = bus
        .twist_cols
        .iter()
        .map(|inst| vec![None; inst.lanes.len()])
        .collect();
    for ev in &step.twist_events {
        let id = ev.twist_id.0;
        let idx = mem_ids
            .binary_search(&id)
            .unwrap_or_else(|_| panic!("unexpected twist_id={id}"));
        let lane_idx = ev.lane.map(|l| l as usize).unwrap_or(0);
        match ev.kind {
            neo_vm_trace::TwistOpKind::Read => reads[idx][lane_idx] = Some((ev.addr, ev.value)),
            neo_vm_trace::TwistOpKind::Write => writes[idx][lane_idx] = Some((ev.addr, ev.value)),
        }
    }

    for (i, &mem_id) in mem_ids.iter().enumerate() {
        let layout = mem_layouts
            .get(&mem_id)
            .expect("mem_layouts missing mem_id");
        let ell = layout.n_side.trailing_zeros() as usize;
        for (lane_idx, cols) in bus.twist_cols[i].lanes.iter().enumerate() {
            if let Some((addr, val)) = reads[i][lane_idx] {
                write_addr_bits_dim_major_le_into_bus(
                    z,
                    bus,
                    cols.ra_bits.clone(),
                    /*j=*/ 0,
                    addr,
                    layout.d,
                    layout.n_side,
                    ell,
                );
                z[bus.bus_cell(cols.rv, 0)] = F::from_u64(val);
                z[bus.bus_cell(cols.has_read, 0)] = F::ONE;
            }
            if let Some((addr, val)) = writes[i][lane_idx] {
                write_addr_bits_dim_major_le_into_bus(
                    z,
                    bus,
                    cols.wa_bits.clone(),
                    /*j=*/ 0,
                    addr,
                    layout.d,
                    layout.n_side,
                    ell,
                );
                z[bus.bus_cell(cols.wv, 0)] = F::from_u64(val);
                z[bus.bus_cell(cols.has_write, 0)] = F::ONE;
            }
        }
    }
}

#[test]
fn rv32_b1_signed_div_rem_shared_bus_constraints_satisfy() {
    let program = vec![
        // x1 = -7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -7,
        },
        // x2 = 3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        // x3 = x1 / x2 = -2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = x1 % x2 = -1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let (prog_layout, _prog_init) =
        prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, &program_bytes).expect("prog_rom_layout");

    let mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::from([
        (
            RAM_ID.0,
            PlainMemLayout {
                k: 512,
                d: 9,
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
    let mut shout_table_ids = vec![
        shout.opcode_to_id(RiscvOpcode::Add).0,
        shout.opcode_to_id(RiscvOpcode::Sltu).0,
    ];
    shout_table_ids.sort_unstable();

    let (ccs_base, layout) =
        build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, /*chunk_size=*/ 1).expect("build_rv32_b1_step_ccs");

    let bus_cfg = rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), HashMap::new())
        .expect("rv32_b1_shared_cpu_bus_config");

    // Canonical bus id order.
    let mut table_ids: Vec<u32> = shout_table_ids.clone();
    table_ids.sort_unstable();
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let mut shout_cpu = Vec::new();
    for id in &table_ids {
        shout_cpu.push(bus_cfg.shout_cpu.get(id).unwrap()[0].clone());
    }
    let mut twist_cpu = Vec::new();
    for id in &mem_ids {
        twist_cpu.extend(bus_cfg.twist_cpu.get(id).unwrap().iter().cloned());
    }

    let lut_insts: Vec<LutInstance<(), F>> = table_ids
        .iter()
        .map(|_| LutInstance {
            comms: Vec::new(),
            k: 0,
            d: 64,
            n_side: 2,
            steps: 1,
            lanes: 1,
            ell: 1,
            table_spec: None,
            table: Vec::new(),
        })
        .collect();
    let mem_insts: Vec<MemInstance<(), F>> = mem_ids
        .iter()
        .map(|id| {
            let l = mem_layouts.get(id).unwrap();
            MemInstance {
                mem_id: *id,
                comms: Vec::new(),
                k: l.k,
                d: l.d,
                n_side: l.n_side,
                steps: 1,
                lanes: l.lanes.max(1),
                ell: l.n_side.trailing_zeros() as usize,
                init: MemInit::Zero,
            }
        })
        .collect();

    let ccs = extend_ccs_with_shared_cpu_bus_constraints(
        &ccs_base,
        layout.m_in,
        layout.const_one,
        &shout_cpu,
        &twist_cpu,
        &lut_insts,
        &mem_insts,
    )
    .expect("inject shared-bus constraints");

    // Build a trace directly from the reference CPU, and then ensure each single-step witness satisfies the CPU CCS.
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(/*base=*/ 0, program.clone());

    let mut twist = HashMapTwist::default();
    for (i, instr) in program.iter().enumerate() {
        let pc = (i as u64) * 4;
        twist.set(TwistId(PROG_ID.0), pc, encode_instruction(instr) as u64);
    }

    let trace = trace_program(cpu, twist, shout, program.len() + 1).expect("trace_program");
    assert!(trace.did_halt(), "program must halt");

    for step in &trace.steps {
        let mut z = rv32_b1_chunk_to_witness_checked(&layout, std::slice::from_ref(step)).expect("witness");
        fill_bus_tail_from_step_events(&mut z, &layout.bus, step, &table_ids, &mem_ids, &mem_layouts);
        let x = &z[..layout.m_in];
        let w = &z[layout.m_in..];
        check_ccs_rowwise_zero(&ccs, x, w).expect("rowwise constraint failure");
    }
}
