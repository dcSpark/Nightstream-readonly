#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, CommitMixers};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, interleave_bits, uninterleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory,
    RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use neo_vm_trace::ShoutEvent;
use p3_field::{Field, PrimeCharacteristicRing};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn div_signed(lhs: u32, rhs: u32) -> u32 {
    let lhs_i = lhs as i32;
    let rhs_i = rhs as i32;
    if rhs_i == 0 {
        return u32::MAX;
    }
    if lhs_i == i32::MIN && rhs_i == -1 {
        return lhs; // overflow case: quotient = MIN_INT
    }
    (lhs_i / rhs_i) as u32
}

fn rem_signed(lhs: u32, rhs: u32) -> u32 {
    let lhs_i = lhs as i32;
    let rhs_i = rhs as i32;
    if rhs_i == 0 {
        return lhs;
    }
    if lhs_i == i32::MIN && rhs_i == -1 {
        return 0; // overflow case: remainder = 0
    }
    (lhs_i % rhs_i) as u32
}

fn plan_paged_ell_addrs(
    m: usize,
    m_in: usize,
    steps: usize,
    ell_addr: usize,
    lanes: usize,
) -> Result<Vec<usize>, String> {
    if steps == 0 {
        return Err("plan_paged_ell_addrs: steps=0".into());
    }
    if m_in > m {
        return Err(format!("plan_paged_ell_addrs: m_in({m_in}) > m({m})"));
    }
    let lanes = lanes.max(1);

    let avail = m - m_in;
    let max_bus_cols_total = avail / steps;
    let per_lane_capacity = max_bus_cols_total / lanes;
    if per_lane_capacity < 3 {
        return Err(format!(
            "plan_paged_ell_addrs: insufficient capacity (need >=3 cols/lane for [addr_bits>=1,has_lookup,val], have per_lane_capacity={per_lane_capacity}; m={m}, m_in={m_in}, steps={steps}, lanes={lanes})"
        ));
    }
    let max_addr_cols_per_page = per_lane_capacity - 2;
    if max_addr_cols_per_page == 0 {
        return Err("plan_paged_ell_addrs: max_addr_cols_per_page=0".into());
    }
    if ell_addr == 0 {
        return Err("plan_paged_ell_addrs: ell_addr=0".into());
    }

    let mut pages = Vec::new();
    let mut remaining = ell_addr;
    while remaining > 0 {
        let take = remaining.min(max_addr_cols_per_page);
        pages.push(take);
        remaining -= take;
    }
    Ok(pages)
}

fn build_paged_shout_only_bus_zs_packed_div(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lane_data: &neo_memory::riscv::trace::ShoutLaneOverTime,
    x_prefix: &[F],
) -> Result<Vec<Vec<F>>, String> {
    if ell_addr != 43 {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_div: expected ell_addr=43 (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_div: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.has_lookup.len() != t {
        return Err("build_paged_shout_only_bus_zs_packed_div: lane length mismatch".into());
    }

    let page_ell_addrs = plan_paged_ell_addrs(m, m_in, t, ell_addr, /*lanes=*/ 1)?;

    let mut out = Vec::with_capacity(page_ell_addrs.len());
    let mut base_idx = 0usize;
    for &page_ell_addr in page_ell_addrs.iter() {
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            m_in,
            t,
            core::iter::once((page_ell_addr, /*lanes=*/ 1usize)),
            core::iter::empty::<(usize, usize)>(),
        )?;
        if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
            return Err(
                "build_paged_shout_only_bus_zs_packed_div: expected 1 shout instance and 0 twist instances".into(),
            );
        }

        let mut z = vec![F::ZERO; m];
        z[..m_in].copy_from_slice(x_prefix);

        let cols = &bus.shout_cols[0].lanes[0];
        let addr_cols: Vec<usize> = cols.addr_bits.clone().collect();
        if addr_cols.len() != page_ell_addr {
            return Err("build_paged_shout_only_bus_zs_packed_div: addr_bits len mismatch".into());
        }

        for j in 0..t {
            let has = lane_data.has_lookup[j];
            z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
            z[bus.bus_cell(cols.val, j)] = if has { F::from_u64(lane_data.value[j]) } else { F::ZERO };

            // Full packed-key layout (ell_addr=43):
            // [lhs_u32, rhs_u32, q_abs, r_abs, rhs_inv, rhs_is_zero, lhs_sign, rhs_sign,
            //  q_inv, q_is_zero, diff_u32, diff_bits[0..32]].
            let mut packed = [F::ZERO; 43];
            if has {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane_data.key[j] as u128);
                let lhs = lhs_u64 as u32;
                let rhs = rhs_u64 as u32;
                let out_val = lane_data.value[j] as u32;
                let expected_out = div_signed(lhs, rhs);
                if out_val != expected_out {
                    return Err(format!(
                        "build_paged_shout_only_bus_zs_packed_div: lane.value mismatch at j={j} (got {out_val:#x}, expected {expected_out:#x})"
                    ));
                }

                let lhs_sign = (lhs >> 31) & 1;
                let rhs_sign = (rhs >> 31) & 1;
                let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
                let rhs_abs = if rhs == 0 {
                    0u32
                } else if rhs_sign == 0 {
                    rhs
                } else {
                    rhs.wrapping_neg()
                };

                let rhs_f = F::from_u64(rhs as u64);
                let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
                let rhs_is_zero = if rhs == 0 { 1u32 } else { 0u32 };

                let (q_abs, r_abs) = if rhs == 0 {
                    (0u32, 0u32)
                } else {
                    (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
                };
                let q_is_zero = if q_abs == 0 { 1u32 } else { 0u32 };
                let q_f = F::from_u64(q_abs as u64);
                let q_inv = if q_f == F::ZERO { F::ZERO } else { q_f.inverse() };

                let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

                packed[0] = F::from_u64(lhs as u64);
                packed[1] = F::from_u64(rhs as u64);
                packed[2] = F::from_u64(q_abs as u64);
                packed[3] = F::from_u64(r_abs as u64);
                packed[4] = rhs_inv;
                packed[5] = if rhs_is_zero == 1 { F::ONE } else { F::ZERO };
                packed[6] = if lhs_sign == 1 { F::ONE } else { F::ZERO };
                packed[7] = if rhs_sign == 1 { F::ONE } else { F::ZERO };
                packed[8] = q_inv;
                packed[9] = if q_is_zero == 1 { F::ONE } else { F::ZERO };
                packed[10] = F::from_u64(diff as u64);
                for bit in 0..32usize {
                    packed[11 + bit] = if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }

                // Sanity-check the packed adapter constraints in the base field.
                let two = F::from_u64(2);
                let two32 = F::from_u64(1u64 << 32);
                let lhs_f = packed[0];
                let rhs_f = packed[1];
                let q_abs_f = packed[2];
                let r_abs_f = packed[3];
                let rhs_inv_f = packed[4];
                let z_f = packed[5];
                let lhs_sign_f = packed[6];
                let rhs_sign_f = packed[7];
                let q_inv_f = packed[8];
                let q0_f = packed[9];
                let diff_f = packed[10];

                let lhs_abs_f = lhs_f + lhs_sign_f * (two32 - two * lhs_f);
                let rhs_abs_f = rhs_f + rhs_sign_f * (two32 - two * rhs_f);

                let c0 = rhs_f * rhs_inv_f - (F::ONE - z_f);
                let c1 = z_f * rhs_f;
                let c2 = q_abs_f * q_inv_f - (F::ONE - q0_f);
                let c3 = q0_f * q_abs_f;
                let c4 = (F::ONE - z_f) * (lhs_abs_f - rhs_abs_f * q_abs_f - r_abs_f);
                let c5 = (F::ONE - z_f) * (r_abs_f - rhs_abs_f - diff_f + two32);
                let mut sum = F::ZERO;
                for bit in 0..32usize {
                    sum += packed[11 + bit] * F::from_u64(1u64 << bit);
                }
                let c6 = diff_f - sum;
                for (name, v) in [
                    ("c0", c0),
                    ("c1", c1),
                    ("c2", c2),
                    ("c3", c3),
                    ("c4", c4),
                    ("c5", c5),
                    ("c6", c6),
                ] {
                    if v != F::ZERO {
                        return Err(format!(
                            "build_paged_shout_only_bus_zs_packed_div: adapter constraint {name} != 0 at j={j}"
                        ));
                    }
                }
            }

            for (local_idx, &col_id) in addr_cols.iter().enumerate() {
                let packed_idx = base_idx + local_idx;
                if packed_idx >= ell_addr {
                    return Err("build_paged_shout_only_bus_zs_packed_div: paging overflow".into());
                }
                z[bus.bus_cell(col_id, j)] = packed[packed_idx];
            }
        }

        out.push(z);
        base_idx += page_ell_addr;
    }

    if base_idx != ell_addr {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_div: paging mismatch (got base_idx={base_idx}, expected ell_addr={ell_addr})"
        ));
    }

    Ok(out)
}

fn build_paged_shout_only_bus_zs_packed_rem(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lane_data: &neo_memory::riscv::trace::ShoutLaneOverTime,
    x_prefix: &[F],
) -> Result<Vec<Vec<F>>, String> {
    if ell_addr != 43 {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_rem: expected ell_addr=43 (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_rem: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.has_lookup.len() != t {
        return Err("build_paged_shout_only_bus_zs_packed_rem: lane length mismatch".into());
    }

    let page_ell_addrs = plan_paged_ell_addrs(m, m_in, t, ell_addr, /*lanes=*/ 1)?;

    let mut out = Vec::with_capacity(page_ell_addrs.len());
    let mut base_idx = 0usize;
    for &page_ell_addr in page_ell_addrs.iter() {
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            m_in,
            t,
            core::iter::once((page_ell_addr, /*lanes=*/ 1usize)),
            core::iter::empty::<(usize, usize)>(),
        )?;
        if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
            return Err(
                "build_paged_shout_only_bus_zs_packed_rem: expected 1 shout instance and 0 twist instances".into(),
            );
        }

        let mut z = vec![F::ZERO; m];
        z[..m_in].copy_from_slice(x_prefix);

        let cols = &bus.shout_cols[0].lanes[0];
        let addr_cols: Vec<usize> = cols.addr_bits.clone().collect();
        if addr_cols.len() != page_ell_addr {
            return Err("build_paged_shout_only_bus_zs_packed_rem: addr_bits len mismatch".into());
        }

        for j in 0..t {
            let has = lane_data.has_lookup[j];
            z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
            z[bus.bus_cell(cols.val, j)] = if has { F::from_u64(lane_data.value[j]) } else { F::ZERO };

            // Full packed-key layout (ell_addr=43):
            // [lhs_u32, rhs_u32, q_abs, r_abs, rhs_inv, rhs_is_zero, lhs_sign, rhs_sign,
            //  r_inv, r_is_zero, diff_u32, diff_bits[0..32]].
            let mut packed = [F::ZERO; 43];
            if has {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane_data.key[j] as u128);
                let lhs = lhs_u64 as u32;
                let rhs = rhs_u64 as u32;
                let out_val = lane_data.value[j] as u32;
                let expected_out = rem_signed(lhs, rhs);
                if out_val != expected_out {
                    return Err(format!(
                        "build_paged_shout_only_bus_zs_packed_rem: lane.value mismatch at j={j} (got {out_val:#x}, expected {expected_out:#x})"
                    ));
                }

                let lhs_sign = (lhs >> 31) & 1;
                let rhs_sign = (rhs >> 31) & 1;
                let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
                let rhs_abs = if rhs == 0 {
                    0u32
                } else if rhs_sign == 0 {
                    rhs
                } else {
                    rhs.wrapping_neg()
                };

                let rhs_f = F::from_u64(rhs as u64);
                let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
                let rhs_is_zero = if rhs == 0 { 1u32 } else { 0u32 };

                let (q_abs, r_abs) = if rhs == 0 {
                    (0u32, 0u32)
                } else {
                    (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
                };
                let r_is_zero = if r_abs == 0 { 1u32 } else { 0u32 };
                let r_f = F::from_u64(r_abs as u64);
                let r_inv = if r_f == F::ZERO { F::ZERO } else { r_f.inverse() };

                let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

                packed[0] = F::from_u64(lhs as u64);
                packed[1] = F::from_u64(rhs as u64);
                packed[2] = F::from_u64(q_abs as u64);
                packed[3] = F::from_u64(r_abs as u64);
                packed[4] = rhs_inv;
                packed[5] = if rhs_is_zero == 1 { F::ONE } else { F::ZERO };
                packed[6] = if lhs_sign == 1 { F::ONE } else { F::ZERO };
                packed[7] = if rhs_sign == 1 { F::ONE } else { F::ZERO };
                packed[8] = r_inv;
                packed[9] = if r_is_zero == 1 { F::ONE } else { F::ZERO };
                packed[10] = F::from_u64(diff as u64);
                for bit in 0..32usize {
                    packed[11 + bit] = if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }
            }

            for (local_idx, &col_id) in addr_cols.iter().enumerate() {
                let packed_idx = base_idx + local_idx;
                if packed_idx >= ell_addr {
                    return Err("build_paged_shout_only_bus_zs_packed_rem: paging overflow".into());
                }
                z[bus.bus_cell(col_id, j)] = packed[packed_idx];
            }
        }

        out.push(z);
        base_idx += page_ell_addr;
    }

    if base_idx != ell_addr {
        return Err(format!(
            "build_paged_shout_only_bus_zs_packed_rem: paging mismatch (got base_idx={base_idx}, expected ell_addr={ell_addr})"
        ));
    }

    Ok(out)
}

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_div_rem_packed_prove_verify() {
    // Program:
    // - x1 = -7*4096, x2 = 3*4096  (DIV=-2, REM=-4096)
    // - x1 = -1*4096, x2 = 3*4096  (DIV=0,  REM=-4096; exercises q_is_zero to avoid "-0")
    // - x1 = INT_MIN, x2 = -1      (DIV overflow case; REM=0)
    // - x1 = INT_MIN, x2 = 0       (DIV by zero => -1; REM by zero => lhs)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: -7 },
        RiscvInstruction::Lui { rd: 2, imm: 3 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Lui { rd: 1, imm: -1 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Lui { rd: 1, imm: -524_288 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: -1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    let mut exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    // Keep only DIV/REM shout events so the test provisions exactly the required tables.
    let tables = RiscvShoutTables::new(32);
    let div_id = tables.opcode_to_id(RiscvOpcode::Div);
    let rem_id = tables.opcode_to_id(RiscvOpcode::Rem);
    for row in exec.rows.iter_mut() {
        if !row.active {
            continue;
        }
        row.shout_events.clear();
        let Some(RiscvInstruction::RAlu { op, .. }) = row.decoded else {
            continue;
        };
        let rs1 = row.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0) as u32;
        let rs2 = row.reg_read_lane1.as_ref().map(|io| io.value).unwrap_or(0) as u32;
        let key = interleave_bits(rs1 as u64, rs2 as u64) as u64;
        match op {
            RiscvOpcode::Div => {
                let out = div_signed(rs1, rs2);
                row.shout_events.push(ShoutEvent {
                    shout_id: div_id,
                    key,
                    value: out as u64,
                });
            }
            RiscvOpcode::Rem => {
                let out = rem_signed(rs1, rs2);
                row.shout_events.push(ShoutEvent {
                    shout_id: rem_id,
                    key,
                    value: out as u64,
                });
            }
            _ => {}
        }
    }
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Params + committer.
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    // Main CPU trace witness commitment.
    let z_cpu: Vec<F> = x.iter().copied().chain(w.iter().copied()).collect();
    let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_cpu);
    let c_cpu = l.commit(&Z_cpu);
    let mcs = (
        McsInstance {
            c: c_cpu,
            x: x.clone(),
            m_in: layout.m_in,
        },
        McsWitness { w, Z: Z_cpu },
    );

    // Shout instances: DIV and REM packed, 1 lane each.
    let t = exec.rows.len();
    let shout_table_ids = vec![div_id.0, rem_id.0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids).expect("extract shout lanes");
    assert_eq!(shout_lanes.len(), 2);
    assert!(
        shout_lanes[0].has_lookup.iter().any(|&b| b),
        "expected at least one DIV lookup"
    );
    assert!(
        shout_lanes[1].has_lookup.iter().any(|&b| b),
        "expected at least one REM lookup"
    );

    let div_inst = LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: 0,
        d: 43,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Div,
            xlen: 32,
        }),
        table: Vec::new(),
    };
    let rem_inst = LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: 0,
        d: 43,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Rem,
            xlen: 32,
        }),
        table: Vec::new(),
    };

    let div_zs =
        build_paged_shout_only_bus_zs_packed_div(ccs.m, layout.m_in, t, div_inst.d * div_inst.ell, &shout_lanes[0], &x)
            .expect("DIV packed z");
    let mut div_comms = Vec::with_capacity(div_zs.len());
    let mut div_mats = Vec::with_capacity(div_zs.len());
    for z in div_zs {
        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        div_comms.push(l.commit(&Z));
        div_mats.push(Z);
    }
    let div_inst = LutInstance::<Cmt, F> {
        comms: div_comms,
        ..div_inst
    };
    let div_wit = LutWitness { mats: div_mats };

    let rem_zs =
        build_paged_shout_only_bus_zs_packed_rem(ccs.m, layout.m_in, t, rem_inst.d * rem_inst.ell, &shout_lanes[1], &x)
            .expect("REM packed z");
    let mut rem_comms = Vec::with_capacity(rem_zs.len());
    let mut rem_mats = Vec::with_capacity(rem_zs.len());
    for z in rem_zs {
        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        rem_comms.push(l.commit(&Z));
        rem_mats.push(Z);
    }
    let rem_inst = LutInstance::<Cmt, F> {
        comms: rem_comms,
        ..rem_inst
    };
    let rem_wit = LutWitness { mats: rem_mats };

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(div_inst, div_wit), (rem_inst, rem_wit)],
        mem_instances: Vec::new(),
        _phantom: PhantomData,
    }];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-div-rem-packed");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove");

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-div-rem-packed");
    let _ = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");
}
