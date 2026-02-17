use std::collections::HashMap;

use p3_goldilocks::Goldilocks as F;

use crate::cpu::bus_layout::{build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape};
use crate::cpu::constraints::{CpuConstraintBuilder, ShoutCpuBinding, TwistCpuBinding, CPU_BUS_COL_DISABLED};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};
use crate::riscv::trace::{
    rv32_decode_lookup_table_id_for_col, rv32_is_decode_lookup_table_id, rv32_is_width_lookup_table_id,
    rv32_trace_lookup_addr_group_for_table_shape, rv32_trace_lookup_selector_group_for_table_id,
    Rv32DecodeSidecarLayout,
};

use super::config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIVU_TABLE_ID, DIV_TABLE_ID, EQ_TABLE_ID, MULHSU_TABLE_ID, MULHU_TABLE_ID,
    MULH_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, RV32_XLEN, SLL_TABLE_ID,
    SLTU_TABLE_ID, SLT_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::{Rv32B1Layout, Rv32TraceCcsLayout};

/// Additional trace-mode Shout lookup family specification.
///
/// This lets trace shared-bus mode instantiate lookup families beyond the fixed RV32 opcode tables,
/// with table-specific address widths (`ell_addr`) while still using padding-only CPU bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceShoutBusSpec {
    pub table_id: u32,
    pub ell_addr: usize,
    pub n_vals: usize,
}

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
fn trace_shout_binding(layout: &Rv32TraceCcsLayout, table_id: u32) -> Option<ShoutCpuBinding> {
    if rv32_is_decode_lookup_table_id(table_id) {
        // Decode lookup families are keyed by PROG read address (pc_before).
        Some(ShoutCpuBinding {
            has_lookup: CPU_BUS_COL_DISABLED,
            addr: Some(trace_cpu_col(layout, layout.trace.pc_before)),
            val: CPU_BUS_COL_DISABLED,
        })
    } else if rv32_is_width_lookup_table_id(table_id) {
        // Width helper lookup families are keyed by cycle index.
        Some(ShoutCpuBinding {
            has_lookup: CPU_BUS_COL_DISABLED,
            addr: Some(trace_cpu_col(layout, layout.trace.cycle)),
            val: CPU_BUS_COL_DISABLED,
        })
    } else {
        None
    }
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
fn trace_lookup_addr_group_for_table_shape(table_id: u32, ell_addr: usize) -> Option<u32> {
    rv32_trace_lookup_addr_group_for_table_shape(table_id, ell_addr)
}

#[inline]
fn trace_lookup_selector_group_for_table_id(table_id: u32) -> Option<u32> {
    rv32_trace_lookup_selector_group_for_table_id(table_id)
}

#[derive(Clone, Copy, Debug)]
struct TraceShoutShape {
    table_id: u32,
    ell_addr: usize,
    n_vals: usize,
    addr_group: Option<u32>,
    selector_group: Option<u32>,
}

fn derive_trace_shout_shapes(
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
) -> Result<Vec<TraceShoutShape>, String> {
    let mut shape_by_table_id = HashMap::<u32, TraceShoutShape>::new();

    for &table_id in shout_table_ids {
        validate_trace_shout_table_id(table_id)?;
        shape_by_table_id.insert(
            table_id,
            TraceShoutShape {
                table_id,
                ell_addr: 2 * RV32_XLEN,
                n_vals: 1usize,
                addr_group: trace_lookup_addr_group_for_table_shape(table_id, 2 * RV32_XLEN),
                selector_group: trace_lookup_selector_group_for_table_id(table_id),
            },
        );
    }

    for spec in extra_shout_specs {
        if spec.ell_addr == 0 {
            return Err(format!(
                "RV32 trace shared bus: extra shout spec for table_id={} has ell_addr=0",
                spec.table_id
            ));
        }
        if spec.n_vals == 0 {
            return Err(format!(
                "RV32 trace shared bus: extra shout spec for table_id={} has n_vals=0",
                spec.table_id
            ));
        }
        if let Some(prev) = shape_by_table_id.get(&spec.table_id) {
            if prev.ell_addr != spec.ell_addr {
                return Err(format!(
                    "RV32 trace shared bus: conflicting ell_addr for table_id={} (base/spec mismatch: {} vs {})",
                    spec.table_id, prev.ell_addr, spec.ell_addr
                ));
            }
            if prev.n_vals != spec.n_vals {
                return Err(format!(
                    "RV32 trace shared bus: conflicting n_vals for table_id={} (base/spec mismatch: {} vs {})",
                    spec.table_id, prev.n_vals, spec.n_vals
                ));
            }
            let inferred_group = trace_lookup_addr_group_for_table_shape(spec.table_id, spec.ell_addr);
            if prev.addr_group != inferred_group {
                return Err(format!(
                    "RV32 trace shared bus: conflicting addr_group for table_id={} (base/spec mismatch: {:?} vs {:?})",
                    spec.table_id, prev.addr_group, inferred_group
                ));
            }
            let inferred_selector_group = trace_lookup_selector_group_for_table_id(spec.table_id);
            if prev.selector_group != inferred_selector_group {
                return Err(format!(
                    "RV32 trace shared bus: conflicting selector_group for table_id={} (base/spec mismatch: {:?} vs {:?})",
                    spec.table_id, prev.selector_group, inferred_selector_group
                ));
            }
        } else {
            shape_by_table_id.insert(
                spec.table_id,
                TraceShoutShape {
                    table_id: spec.table_id,
                    ell_addr: spec.ell_addr,
                    n_vals: spec.n_vals,
                    addr_group: trace_lookup_addr_group_for_table_shape(spec.table_id, spec.ell_addr),
                    selector_group: trace_lookup_selector_group_for_table_id(spec.table_id),
                },
            );
        }
    }

    let mut shapes: Vec<TraceShoutShape> = shape_by_table_id.into_values().collect();
    shapes.sort_unstable_by_key(|shape| shape.table_id);
    Ok(shapes)
}

fn audit_bus_tail_constraint_coverage(
    builder: &CpuConstraintBuilder<F>,
    bus: &crate::cpu::bus_layout::BusLayout,
) -> Result<(), String> {
    let mut referenced = vec![false; bus.bus_cols];
    let bus_end = bus
        .bus_base
        .checked_add(bus.bus_region_len())
        .ok_or_else(|| "RV32 trace shared bus: bus tail end overflow during coverage audit".to_string())?;

    let mut mark_col = |col: usize| {
        if col >= bus.bus_base && col < bus_end {
            let rel = col - bus.bus_base;
            let col_id = rel / bus.chunk_size;
            if col_id < referenced.len() {
                referenced[col_id] = true;
            }
        }
    };

    for c in builder.constraints() {
        mark_col(c.condition_col);
        for &col in &c.additional_condition_cols {
            mark_col(col);
        }
        for &(col, _) in &c.b_terms {
            mark_col(col);
        }
    }

    let dead: Vec<usize> = referenced
        .iter()
        .enumerate()
        .filter_map(|(i, used)| if *used { None } else { Some(i) })
        .collect();

    if dead.is_empty() {
        return Ok(());
    }

    let preview: Vec<usize> = dead.iter().copied().take(24).collect();
    Err(format!(
        "RV32 trace shared bus: dead bus-tail columns are not referenced by constraints (count={}, first={preview:?})",
        dead.len()
    ))
}

#[inline]
fn trace_disabled_twist_binding(_layout: &Rv32TraceCcsLayout) -> TwistCpuBinding {
    TwistCpuBinding {
        has_read: CPU_BUS_COL_DISABLED,
        has_write: CPU_BUS_COL_DISABLED,
        read_addr: CPU_BUS_COL_DISABLED,
        write_addr: CPU_BUS_COL_DISABLED,
        rv: CPU_BUS_COL_DISABLED,
        wv: CPU_BUS_COL_DISABLED,
        inc: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct TraceDecodeSelectorCols {
    rd_has_write: usize,
    ram_has_read: usize,
    ram_has_write: usize,
}

fn resolve_trace_decode_selector_cols(
    layout: &Rv32TraceCcsLayout,
    shout_shapes: &[TraceShoutShape],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<TraceDecodeSelectorCols, String> {
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
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
        twist_shapes.push((ell_addr, mem_layout.lanes.max(1)));
    }

    let bus = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        layout.m,
        layout.m_in,
        layout.t,
        shout_shapes.iter().map(|shape| ShoutInstanceShape {
            ell_addr: shape.ell_addr,
            lanes: 1usize,
            n_vals: shape.n_vals.max(1),
            addr_group: shape.addr_group.map(|v| v as u64),
            selector_group: shape.selector_group.map(|v| v as u64),
        }),
        twist_shapes.iter().copied(),
    )?;

    trace_decode_selector_cols_from_bus(&bus, shout_shapes)
}

fn trace_decode_selector_cols_from_bus(
    bus: &crate::cpu::bus_layout::BusLayout,
    shout_shapes: &[TraceShoutShape],
) -> Result<TraceDecodeSelectorCols, String> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let rd_has_write_table_id = rv32_decode_lookup_table_id_for_col(decode_layout.rd_has_write);
    let ram_has_read_table_id = rv32_decode_lookup_table_id_for_col(decode_layout.ram_has_read);
    let ram_has_write_table_id = rv32_decode_lookup_table_id_for_col(decode_layout.ram_has_write);
    let table_val_col = |table_id: u32| -> Result<usize, String> {
        let shout_idx = shout_shapes
            .iter()
            .position(|shape| shape.table_id == table_id)
            .ok_or_else(|| {
                format!(
                    "RV32 trace shared bus: missing decode lookup table_id={table_id} required for Twist selector binding"
                )
            })?;
        let inst_cols = bus.shout_cols.get(shout_idx).ok_or_else(|| {
            format!("RV32 trace shared bus: missing shout cols for decode lookup table_id={table_id}")
        })?;
        let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
            format!("RV32 trace shared bus: expected one shout lane for decode lookup table_id={table_id}")
        })?;
        bus.bus_base
            .checked_add(lane0.primary_val() * bus.chunk_size)
            .ok_or_else(|| "RV32 trace shared bus: decode selector column overflow".to_string())
    };
    Ok(TraceDecodeSelectorCols {
        rd_has_write: table_val_col(rd_has_write_table_id)?,
        ram_has_read: table_val_col(ram_has_read_table_id)?,
        ram_has_write: table_val_col(ram_has_write_table_id)?,
    })
}

#[inline]
fn trace_twist_primary_binding(
    layout: &Rv32TraceCcsLayout,
    mem_id: u32,
    decode_selectors: TraceDecodeSelectorCols,
) -> TwistCpuBinding {
    let active = trace_cpu_col(layout, layout.trace.active);
    if mem_id == RAM_ID.0 {
        TwistCpuBinding {
            has_read: decode_selectors.ram_has_read,
            has_write: decode_selectors.ram_has_write,
            read_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            write_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            rv: trace_cpu_col(layout, layout.trace.ram_rv),
            wv: trace_cpu_col(layout, layout.trace.ram_wv),
            inc: None,
        }
    } else if mem_id == PROG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: CPU_BUS_COL_DISABLED,
            read_addr: trace_cpu_col(layout, layout.trace.pc_before),
            write_addr: CPU_BUS_COL_DISABLED,
            rv: trace_cpu_col(layout, layout.trace.instr_word),
            wv: CPU_BUS_COL_DISABLED,
            inc: None,
        }
    } else if mem_id == REG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: decode_selectors.rd_has_write,
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
    rv32_trace_shared_cpu_bus_config_with_specs(layout, shout_table_ids, &[], mem_layouts, initial_mem)
}

/// Shared CPU-bus bindings for trace mode with extra lookup-family Shout specs.
pub fn rv32_trace_shared_cpu_bus_config_with_specs(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    let shout_shapes = derive_trace_shout_shapes(shout_table_ids, extra_shout_specs)?;
    let decode_selectors = resolve_trace_decode_selector_cols(layout, &shout_shapes, &mem_layouts)?;

    let mut shout_cpu = HashMap::new();
    for shape in &shout_shapes {
        // Keep opcode Shout families on reduction-time linkage ownership.
        // Decode/width lookup families also get row-level key-binding constraints
        // to tie bus addr_bits to committed CPU trace columns.
        let binding = trace_shout_binding(layout, shape.table_id);
        shout_cpu.insert(shape.table_id, binding.into_iter().collect());
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
            bindings.push(trace_twist_primary_binding(layout, mem_id, decode_selectors));
            bindings.push(TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: CPU_BUS_COL_DISABLED,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: CPU_BUS_COL_DISABLED,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: CPU_BUS_COL_DISABLED,
                inc: None,
            });
            let disabled = trace_disabled_twist_binding(layout);
            for _ in 2..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        } else {
            let primary = trace_twist_primary_binding(layout, mem_id, decode_selectors);
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
    rv32_trace_shared_bus_requirements_with_specs(layout, shout_table_ids, &[], mem_layouts)
}

/// Return `(bus_region_len, reserved_rows)` required by trace shared-bus mode with extra lookup-family specs.
pub fn rv32_trace_shared_bus_requirements_with_specs(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<(usize, usize), String> {
    let shout_shapes = derive_trace_shout_shapes(shout_table_ids, extra_shout_specs)?;

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let mut shout_cols = 0usize;
    let mut seen_addr_groups = HashMap::<u32, usize>::new();
    let mut seen_selector_groups = std::collections::HashSet::<u32>::new();
    for shape in &shout_shapes {
        if let Some(group) = shape.addr_group {
            if let Some(prev_ell) = seen_addr_groups.insert(group, shape.ell_addr) {
                if prev_ell != shape.ell_addr {
                    return Err(format!(
                        "RV32 trace shared bus: addr_group={} has conflicting ell_addr ({} vs {})",
                        group, prev_ell, shape.ell_addr
                    ));
                }
            } else {
                shout_cols = shout_cols
                    .checked_add(shape.ell_addr)
                    .ok_or_else(|| "RV32 trace shared bus: shout shared-addr width overflow".to_string())?;
            }
        } else {
            shout_cols = shout_cols
                .checked_add(shape.ell_addr)
                .ok_or_else(|| "RV32 trace shared bus: shout lane width overflow".to_string())?;
        }
        if let Some(selector_group) = shape.selector_group {
            if seen_selector_groups.insert(selector_group) {
                shout_cols = shout_cols
                    .checked_add(1)
                    .ok_or_else(|| "RV32 trace shared bus: shout selector width overflow".to_string())?;
            }
        } else {
            shout_cols = shout_cols
                .checked_add(1)
                .ok_or_else(|| "RV32 trace shared bus: shout selector width overflow".to_string())?;
        }
        shout_cols = shout_cols
            .checked_add(shape.n_vals)
            .ok_or_else(|| "RV32 trace shared bus: shout value width overflow".to_string())?;
    }
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

    let bus = crate::cpu::bus_layout::build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_total,
        layout.m_in,
        layout.t,
        shout_shapes.iter().map(|shape| ShoutInstanceShape {
            ell_addr: shape.ell_addr,
            lanes: 1usize,
            n_vals: shape.n_vals.max(1),
            addr_group: shape.addr_group.map(|v| v as u64),
            selector_group: shape.selector_group.map(|v| v as u64),
        }),
        twist_shapes.iter().copied(),
    )?;
    let decode_selectors = trace_decode_selector_cols_from_bus(&bus, &shout_shapes)?;

    let mut builder = CpuConstraintBuilder::<F>::new(m_total, m_total, layout.const_one);

    let mut addr_range_counts = HashMap::<(usize, usize), usize>::new();
    for inst_cols in bus.shout_cols.iter() {
        for lane_cols in inst_cols.lanes.iter() {
            let key = (lane_cols.addr_bits.start, lane_cols.addr_bits.end);
            *addr_range_counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut addr_range_bitness_added = std::collections::HashSet::<(usize, usize)>::new();
    let mut selector_bitness_added = std::collections::HashSet::<usize>::new();
    let mut shout_key_binding_added = std::collections::HashSet::<(bool, usize, usize, usize, usize)>::new();
    for (i, _) in shout_shapes.iter().enumerate() {
        let lane0 = &bus.shout_cols[i].lanes[0];
        if let Some(binding) = trace_shout_binding(layout, shout_shapes[i].table_id) {
            let mut dedup_binding = binding.clone();
            if let Some(addr_base) = dedup_binding.addr {
                let (is_bus_gate, gate_base) = if dedup_binding.has_lookup == CPU_BUS_COL_DISABLED {
                    (true, lane0.has_lookup)
                } else {
                    (false, dedup_binding.has_lookup)
                };
                let key_sig = (
                    is_bus_gate,
                    gate_base,
                    addr_base,
                    lane0.addr_bits.start,
                    lane0.addr_bits.end,
                );
                if !shout_key_binding_added.insert(key_sig) {
                    dedup_binding.addr = None;
                }
            }
            builder.add_shout_instance_linkage_bound(&bus, lane0, &dedup_binding);
        }
        let key = (lane0.addr_bits.start, lane0.addr_bits.end);
        let shared_addr_group = addr_range_counts.get(&key).copied().unwrap_or(0) > 1;
        let selector_first = selector_bitness_added.insert(lane0.has_lookup);
        if shared_addr_group {
            if selector_first {
                builder.add_shout_instance_padding_value_only(&bus, lane0);
            } else {
                builder.add_shout_instance_value_padding_only(&bus, lane0);
            }
            if addr_range_bitness_added.insert(key) {
                builder.add_shout_instance_addr_bit_bitness(&bus, lane0);
            }
        } else {
            if selector_first {
                builder.add_shout_instance_padding(&bus, lane0);
            } else {
                builder.add_shout_instance_padding_without_selector_bitness(&bus, lane0);
            }
        }
    }
    for (i, &mem_id) in mem_ids.iter().enumerate() {
        let inst = &bus.twist_cols[i];
        if inst.lanes.is_empty() {
            continue;
        }
        if mem_id == REG_ID.0 {
            let lane0 = trace_twist_primary_binding(layout, mem_id, decode_selectors);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            let lane1 = TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: CPU_BUS_COL_DISABLED,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: CPU_BUS_COL_DISABLED,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: CPU_BUS_COL_DISABLED,
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
            let lane0 = trace_twist_primary_binding(layout, mem_id, decode_selectors);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            if inst.lanes.len() > 1 {
                let disabled = trace_disabled_twist_binding(layout);
                for lane_cols in &inst.lanes[1..] {
                    builder.add_twist_instance_bound(&bus, lane_cols, &disabled);
                }
            }
        }
    }

    audit_bus_tail_constraint_coverage(&builder, &bus)?;

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
