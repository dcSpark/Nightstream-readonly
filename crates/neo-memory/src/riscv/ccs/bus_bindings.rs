use std::collections::HashMap;

use p3_goldilocks::Goldilocks as F;

use crate::cpu::constraints::{CpuConstraintBuilder, ShoutCpuBinding, TwistCpuBinding};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};

use super::config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIVU_TABLE_ID, DIV_TABLE_ID, EQ_TABLE_ID, MULHSU_TABLE_ID, MULHU_TABLE_ID,
    MULH_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, RV32_XLEN, SLL_TABLE_ID,
    SLTU_TABLE_ID, SLT_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::{Rv32B1Layout, Rv32TraceCcsLayout};

fn shout_cpu_binding(layout: &Rv32B1Layout, table_id: u32) -> ShoutCpuBinding {
    // NOTE: We intentionally do *not* bind Shout addr_bits to a packed CPU scalar here.
    //
    // In Neo, Ajtai encodes witness scalars using `params.d=54` balanced base-`b` digits. A full
    // 64-bit packed Shout key can exceed that representable range, which breaks the MCS/DEC plumbing.
    //
    // Shout key correctness is enforced by the RV32 B1 decode/semantics sidecar CCS instead.
    let addr = None;
    match table_id {
        AND_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.and_has_lookup,
            addr,
            val: layout.alu_out,
        },
        XOR_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.xor_has_lookup,
            addr,
            val: layout.alu_out,
        },
        OR_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.or_has_lookup,
            addr,
            val: layout.alu_out,
        },
        ADD_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.add_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SUB_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sub_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SLT_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.slt_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SLTU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sltu_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SLL_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sll_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SRL_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.srl_has_lookup,
            addr,
            val: layout.alu_out,
        },
        SRA_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sra_has_lookup,
            addr,
            val: layout.alu_out,
        },
        EQ_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.eq_has_lookup,
            addr,
            val: layout.alu_out,
        },
        NEQ_TABLE_ID => ShoutCpuBinding {
            // Nightstream encodes BNE as EQ + invert, so NEQ is unused.
            has_lookup: layout.zero,
            addr,
            val: layout.zero,
        },
        _ => {
            // Bind unused tables to fixed-zero CPU columns so they are provably inactive.
            let zero = layout.zero;
            ShoutCpuBinding {
                has_lookup: zero,
                addr,
                val: zero,
            }
        }
    }
}

fn twist_cpu_binding(layout: &Rv32B1Layout, mem_id: u32) -> TwistCpuBinding {
    if mem_id == RAM_ID.0 {
        TwistCpuBinding {
            has_read: layout.ram_has_read,
            has_write: layout.ram_has_write,
            read_addr: layout.eff_addr,
            write_addr: layout.eff_addr,
            rv: layout.mem_rv,
            wv: layout.ram_wv,
            inc: None,
        }
    } else if mem_id == PROG_ID.0 {
        let zero = layout.zero;
        TwistCpuBinding {
            has_read: layout.is_active,
            has_write: zero,
            read_addr: layout.pc_in,
            write_addr: zero,
            rv: layout.instr_word,
            wv: zero,
            inc: None,
        }
    } else if mem_id == REG_ID.0 {
        // Regfile lane0 binding (read rs1, write rd).
        TwistCpuBinding {
            has_read: layout.is_active,
            has_write: layout.reg_has_write,
            read_addr: layout.rs1_field,
            write_addr: layout.rd_field,
            rv: layout.rs1_val,
            wv: layout.rd_write_val,
            inc: None,
        }
    } else {
        // Disable any additional Twist instances by binding to fixed-zero CPU columns.
        let zero = layout.zero;
        TwistCpuBinding {
            has_read: zero,
            has_write: zero,
            read_addr: zero,
            write_addr: zero,
            rv: zero,
            wv: zero,
            inc: None,
        }
    }
}

#[inline]
fn trace_cpu_col(layout: &Rv32TraceCcsLayout, trace_col: usize) -> usize {
    layout.cell(trace_col, 0)
}

#[inline]
fn trace_zero_col(layout: &Rv32TraceCcsLayout) -> usize {
    // `jalr_drop_bit[0]` is constrained to 0 on every row in trace CCS.
    trace_cpu_col(layout, layout.trace.jalr_drop_bit[0])
}

#[inline]
fn validate_trace_shout_table_id(table_id: u32) -> Result<(), String> {
    match table_id {
        AND_TABLE_ID | XOR_TABLE_ID | OR_TABLE_ID | ADD_TABLE_ID | SUB_TABLE_ID | SLT_TABLE_ID | SLTU_TABLE_ID
        | SLL_TABLE_ID | SRL_TABLE_ID | SRA_TABLE_ID | EQ_TABLE_ID | NEQ_TABLE_ID | MUL_TABLE_ID | MULH_TABLE_ID
        | MULHU_TABLE_ID | MULHSU_TABLE_ID | DIV_TABLE_ID | DIVU_TABLE_ID | REM_TABLE_ID | REMU_TABLE_ID => Ok(()),
        _ => Err(format!("RV32 trace shared bus: unsupported shout table_id={table_id}")),
    }
}

#[inline]
fn trace_disabled_twist_binding(layout: &Rv32TraceCcsLayout) -> TwistCpuBinding {
    let zero = trace_zero_col(layout);
    TwistCpuBinding {
        has_read: zero,
        has_write: zero,
        read_addr: zero,
        write_addr: zero,
        rv: zero,
        wv: zero,
        inc: None,
    }
}

#[inline]
fn trace_twist_primary_binding(layout: &Rv32TraceCcsLayout, mem_id: u32) -> TwistCpuBinding {
    let active = trace_cpu_col(layout, layout.trace.active);
    let zero = trace_zero_col(layout);
    if mem_id == RAM_ID.0 {
        TwistCpuBinding {
            has_read: trace_cpu_col(layout, layout.trace.ram_has_read),
            has_write: trace_cpu_col(layout, layout.trace.ram_has_write),
            read_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            write_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            rv: trace_cpu_col(layout, layout.trace.ram_rv),
            wv: trace_cpu_col(layout, layout.trace.ram_wv),
            inc: None,
        }
    } else if mem_id == PROG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: zero,
            read_addr: trace_cpu_col(layout, layout.trace.prog_addr),
            write_addr: zero,
            rv: trace_cpu_col(layout, layout.trace.prog_value),
            wv: zero,
            inc: None,
        }
    } else if mem_id == REG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: trace_cpu_col(layout, layout.trace.rd_has_write),
            read_addr: trace_cpu_col(layout, layout.trace.rs1_addr),
            write_addr: trace_cpu_col(layout, layout.trace.rd_addr),
            rv: trace_cpu_col(layout, layout.trace.rs1_val),
            wv: trace_cpu_col(layout, layout.trace.rd_val),
            inc: None,
        }
    } else {
        trace_disabled_twist_binding(layout)
    }
}

/// Shared CPU-bus bindings for the RV32 trace-wiring step circuit.
pub fn rv32_trace_shared_cpu_bus_config(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    let mut table_ids = shout_table_ids.to_vec();
    table_ids.sort_unstable();
    table_ids.dedup();

    let mut shout_cpu = HashMap::new();
    for table_id in table_ids {
        validate_trace_shout_table_id(table_id)?;
        // In trace shared-bus mode, Shout CPU-linkage is checked at Route-A reduction-time
        // aggregates, so per-lane bus linkage is intentionally omitted.
        shout_cpu.insert(table_id, Vec::new());
    }

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut twist_cpu = HashMap::new();
    for mem_id in mem_ids {
        let lanes = mem_layouts
            .get(&mem_id)
            .map(|l| l.lanes.max(1))
            .ok_or_else(|| format!("RV32 trace shared bus: missing mem layout for mem_id={mem_id}"))?;
        if mem_id == REG_ID.0 {
            if lanes < 2 {
                return Err(format!(
                    "RV32 trace shared bus: REG_ID requires lanes>=2 (got lanes={lanes})"
                ));
            }
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(trace_twist_primary_binding(layout, mem_id));
            let zero = trace_zero_col(layout);
            bindings.push(TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: zero,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: zero,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: zero,
                inc: None,
            });
            let disabled = trace_disabled_twist_binding(layout);
            for _ in 2..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        } else {
            let primary = trace_twist_primary_binding(layout, mem_id);
            let disabled = trace_disabled_twist_binding(layout);
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(primary);
            for _ in 1..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        }
    }

    Ok(SharedCpuBusConfig {
        mem_layouts,
        initial_mem,
        const_one_col: layout.const_one,
        shout_cpu,
        twist_cpu,
    })
}

/// Return `(bus_region_len, reserved_rows)` required by trace shared-bus mode.
pub fn rv32_trace_shared_bus_requirements(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<(usize, usize), String> {
    let mut table_ids = shout_table_ids.to_vec();
    table_ids.sort_unstable();
    table_ids.dedup();
    for &table_id in &table_ids {
        validate_trace_shout_table_id(table_id)?;
    }

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let shout_cols: usize = table_ids.iter().map(|_| 2 * RV32_XLEN + 2).sum();
    let mut twist_cols = 0usize;
    let mut twist_shapes = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let mem_layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("RV32 trace shared bus: missing mem layout for mem_id={mem_id}"))?;
        if mem_layout.n_side == 0 || !mem_layout.n_side.is_power_of_two() {
            return Err(format!(
                "RV32 trace shared bus: mem_id={mem_id} n_side={} must be power-of-two",
                mem_layout.n_side
            ));
        }
        let ell = mem_layout.n_side.trailing_zeros() as usize;
        let ell_addr = mem_layout.d * ell;
        let lanes = mem_layout.lanes.max(1);
        if *mem_id == REG_ID.0 && lanes < 2 {
            return Err(format!(
                "RV32 trace shared bus: REG_ID requires lanes>=2 (got lanes={lanes})"
            ));
        }
        twist_cols = twist_cols
            .checked_add((2 * ell_addr + 5) * lanes)
            .ok_or_else(|| "RV32 trace shared bus: twist bus column overflow".to_string())?;
        twist_shapes.push((ell_addr, lanes));
    }
    let bus_cols = shout_cols
        .checked_add(twist_cols)
        .ok_or_else(|| "RV32 trace shared bus: bus column overflow".to_string())?;
    let bus_region_len = bus_cols
        .checked_mul(layout.t)
        .ok_or_else(|| "RV32 trace shared bus: bus region overflow".to_string())?;
    let m_total = layout
        .m
        .checked_add(bus_region_len)
        .ok_or_else(|| "RV32 trace shared bus: total m overflow".to_string())?;

    let bus = crate::cpu::bus_layout::build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m_total,
        layout.m_in,
        layout.t,
        table_ids.iter().map(|_| (2 * RV32_XLEN, 1usize)),
        twist_shapes.iter().copied(),
    )?;

    let mut builder = CpuConstraintBuilder::<F>::new(m_total, m_total, layout.const_one);

    for (i, _table_id) in table_ids.iter().enumerate() {
        builder.add_shout_instance_padding(&bus, &bus.shout_cols[i].lanes[0]);
    }
    for (i, &mem_id) in mem_ids.iter().enumerate() {
        let inst = &bus.twist_cols[i];
        if inst.lanes.is_empty() {
            continue;
        }
        if mem_id == REG_ID.0 {
            let lane0 = trace_twist_primary_binding(layout, mem_id);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            let zero = trace_zero_col(layout);
            let lane1 = TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: zero,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: zero,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: zero,
                inc: None,
            };
            if inst.lanes.len() >= 2 {
                builder.add_twist_instance_bound(&bus, &inst.lanes[1], &lane1);
            }
            if inst.lanes.len() > 2 {
                let disabled = trace_disabled_twist_binding(layout);
                for lane_cols in &inst.lanes[2..] {
                    builder.add_twist_instance_bound(&bus, lane_cols, &disabled);
                }
            }
        } else {
            let lane0 = trace_twist_primary_binding(layout, mem_id);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            if inst.lanes.len() > 1 {
                let disabled = trace_disabled_twist_binding(layout);
                for lane_cols in &inst.lanes[1..] {
                    builder.add_twist_instance_bound(&bus, lane_cols, &disabled);
                }
            }
        }
    }

    Ok((bus_region_len, builder.constraints().len()))
}

pub(super) fn injected_bus_constraints_len(layout: &Rv32B1Layout, table_ids: &[u32], mem_ids: &[u32]) -> usize {
    let shout_cpu: Vec<ShoutCpuBinding> = table_ids
        .iter()
        .map(|&id| shout_cpu_binding(layout, id))
        .collect();
    let mut builder = CpuConstraintBuilder::<F>::new(layout.m, layout.m, layout.const_one);
    for (i, cpu) in shout_cpu.iter().enumerate() {
        builder.add_shout_instance_bound(&layout.bus, &layout.bus.shout_cols[i].lanes[0], cpu);
    }
    for (i, &mem_id) in mem_ids.iter().enumerate() {
        let inst = &layout.bus.twist_cols[i];
        if inst.lanes.is_empty() {
            continue;
        }
        if mem_id == REG_ID.0 {
            // Regfile uses two lanes:
            // - lane0: read rs1, write rd
            // - lane1: read rs2, no write
            let lane0 = twist_cpu_binding(layout, mem_id);
            builder.add_twist_instance_bound(&layout.bus, &inst.lanes[0], &lane0);

            let zero = layout.zero;
            let lane1 = TwistCpuBinding {
                has_read: layout.is_active,
                has_write: zero,
                read_addr: layout.rs2_field,
                write_addr: zero,
                rv: layout.rs2_val,
                wv: zero,
                inc: None,
            };
            if inst.lanes.len() >= 2 {
                builder.add_twist_instance_bound(&layout.bus, &inst.lanes[1], &lane1);
            }
            // Any remaining lanes are disabled.
            if inst.lanes.len() > 2 {
                let disabled = twist_cpu_binding(layout, u32::MAX);
                for lane_cols in &inst.lanes[2..] {
                    builder.add_twist_instance_bound(&layout.bus, lane_cols, &disabled);
                }
            }
        } else {
            // Default: lane0 bound, remaining lanes disabled.
            let lane0 = twist_cpu_binding(layout, mem_id);
            builder.add_twist_instance_bound(&layout.bus, &inst.lanes[0], &lane0);
            if inst.lanes.len() > 1 {
                let disabled = twist_cpu_binding(layout, u32::MAX);
                for lane_cols in &inst.lanes[1..] {
                    builder.add_twist_instance_bound(&layout.bus, lane_cols, &disabled);
                }
            }
        }
    }
    builder.constraints().len()
}

/// Shared CPU-bus bindings for the RV32 B1 step circuit.
///
/// This config:
/// - binds `PROG_ID` reads to `pc_in` / `instr_word`, forces no ROM writes,
/// - binds `RAM_ID` reads/writes to `eff_addr` / `mem_rv` / `ram_wv` (with selectors derived from instruction flags),
/// - binds RV32IM Shout opcode tables (ids 0..=19) to `alu_out` (addr_bits are constrained directly by the step CCS).
pub fn rv32_b1_shared_cpu_bus_config(
    layout: &Rv32B1Layout,
    shout_table_ids: &[u32],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    let (table_ids, _ell_addrs) = derive_shout_ids_and_ell_addrs(shout_table_ids)?;

    let mut shout_cpu = HashMap::new();
    for table_id in table_ids {
        shout_cpu.insert(table_id, vec![shout_cpu_binding(layout, table_id)]);
    }

    let (mem_ids, _ell_addrs) = derive_mem_ids_and_ell_addrs(&mem_layouts)?;
    let mut twist_cpu = HashMap::new();
    for mem_id in mem_ids {
        let lanes = mem_layouts
            .get(&mem_id)
            .map(|l| l.lanes.max(1))
            .unwrap_or(1);

        if mem_id == REG_ID.0 {
            if lanes < 2 {
                return Err(format!(
                    "RV32 B1 shared bus: REG_ID requires lanes>=2 (got lanes={lanes})"
                ));
            }
            let lane0 = twist_cpu_binding(layout, mem_id);
            let zero = layout.zero;
            let lane1 = TwistCpuBinding {
                has_read: layout.is_active,
                has_write: zero,
                read_addr: layout.rs2_field,
                write_addr: zero,
                rv: layout.rs2_val,
                wv: zero,
                inc: None,
            };
            let disabled = twist_cpu_binding(layout, u32::MAX);
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(lane0);
            bindings.push(lane1);
            for _ in 2..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        } else {
            let primary = twist_cpu_binding(layout, mem_id);
            let disabled = twist_cpu_binding(layout, u32::MAX);
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(primary);
            for _ in 1..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        }
    }

    Ok(SharedCpuBusConfig {
        mem_layouts,
        initial_mem,
        const_one_col: layout.const_one,
        shout_cpu,
        twist_cpu,
    })
}
