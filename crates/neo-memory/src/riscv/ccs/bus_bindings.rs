use std::collections::HashMap;

use p3_goldilocks::Goldilocks as F;

use crate::cpu::constraints::{CpuConstraintBuilder, ShoutCpuBinding, TwistCpuBinding};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};

use super::config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, EQ_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, SLL_TABLE_ID, SLTU_TABLE_ID, SLT_TABLE_ID,
    SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::Rv32B1Layout;

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
            has_lookup: layout.is_sub,
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
            has_lookup: layout.is_beq,
            addr,
            val: layout.alu_out,
        },
        NEQ_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_bne,
            addr,
            val: layout.alu_out,
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
        let lanes = mem_layouts.get(&mem_id).map(|l| l.lanes.max(1)).unwrap_or(1);

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
