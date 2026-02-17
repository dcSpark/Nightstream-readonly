use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;

/// Base lookup table id for width-column lookup families in shared-bus mode.
///
/// Table id for width column `c` is `RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE + c`.
pub const RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE: u32 = 0x5256_5800;
/// Base address-group id for width lookup lanes.
pub const RV32_TRACE_WIDTH_ADDR_GROUP_BASE: u32 = 0x5256_5A00;

#[derive(Clone, Debug)]
pub struct Rv32WidthSidecarLayout {
    pub cols: usize,
    pub ram_rv_q16: usize,
    pub rs2_q16: usize,
    pub ram_rv_low_bit: [usize; 16],
    pub rs2_low_bit: [usize; 16],
}

impl Rv32WidthSidecarLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };

        let ram_rv_q16 = take();
        let rs2_q16 = take();

        let ram_rv_b0 = take();
        let ram_rv_b1 = take();
        let ram_rv_b2 = take();
        let ram_rv_b3 = take();
        let ram_rv_b4 = take();
        let ram_rv_b5 = take();
        let ram_rv_b6 = take();
        let ram_rv_b7 = take();
        let ram_rv_b8 = take();
        let ram_rv_b9 = take();
        let ram_rv_b10 = take();
        let ram_rv_b11 = take();
        let ram_rv_b12 = take();
        let ram_rv_b13 = take();
        let ram_rv_b14 = take();
        let ram_rv_b15 = take();

        let rs2_low_b0 = take();
        let rs2_low_b1 = take();
        let rs2_low_b2 = take();
        let rs2_low_b3 = take();
        let rs2_low_b4 = take();
        let rs2_low_b5 = take();
        let rs2_low_b6 = take();
        let rs2_low_b7 = take();
        let rs2_low_b8 = take();
        let rs2_low_b9 = take();
        let rs2_low_b10 = take();
        let rs2_low_b11 = take();
        let rs2_low_b12 = take();
        let rs2_low_b13 = take();
        let rs2_low_b14 = take();
        let rs2_low_b15 = take();

        debug_assert_eq!(next, 34);
        Self {
            cols: next,
            ram_rv_q16,
            rs2_q16,
            ram_rv_low_bit: [
                ram_rv_b0, ram_rv_b1, ram_rv_b2, ram_rv_b3, ram_rv_b4, ram_rv_b5, ram_rv_b6, ram_rv_b7, ram_rv_b8,
                ram_rv_b9, ram_rv_b10, ram_rv_b11, ram_rv_b12, ram_rv_b13, ram_rv_b14, ram_rv_b15,
            ],
            rs2_low_bit: [
                rs2_low_b0,
                rs2_low_b1,
                rs2_low_b2,
                rs2_low_b3,
                rs2_low_b4,
                rs2_low_b5,
                rs2_low_b6,
                rs2_low_b7,
                rs2_low_b8,
                rs2_low_b9,
                rs2_low_b10,
                rs2_low_b11,
                rs2_low_b12,
                rs2_low_b13,
                rs2_low_b14,
                rs2_low_b15,
            ],
        }
    }
}

#[inline]
pub fn rv32_width_lookup_backed_cols(layout: &Rv32WidthSidecarLayout) -> Vec<usize> {
    (0..layout.cols).collect()
}

#[inline]
pub const fn rv32_width_lookup_table_id_for_col(col: usize) -> u32 {
    RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE + col as u32
}

#[inline]
pub const fn rv32_is_width_lookup_table_id(table_id: u32) -> bool {
    table_id >= RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE
        && table_id < RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE + 34
}

#[inline]
pub fn rv32_width_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    if !rv32_is_width_lookup_table_id(table_id) {
        return None;
    }
    Some(RV32_TRACE_WIDTH_ADDR_GROUP_BASE)
}

#[derive(Clone, Debug)]
pub struct Rv32WidthSidecarWitness {
    pub t: usize,
    pub cols: Vec<Vec<F>>,
}

impl Rv32WidthSidecarWitness {
    pub fn new_zero(layout: &Rv32WidthSidecarLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }
}

pub fn rv32_width_sidecar_witness_from_exec_table(
    layout: &Rv32WidthSidecarLayout,
    exec: &Rv32ExecTable,
) -> Rv32WidthSidecarWitness {
    let cols = exec.to_columns();
    let t = cols.len();
    let mut wit = Rv32WidthSidecarWitness::new_zero(layout, t);

    for i in 0..t {
        if !cols.active[i] {
            continue;
        }

        let rs2_val_u64 = cols.rs2_val[i];
        wit.cols[layout.rs2_q16][i] = F::from_u64(rs2_val_u64 >> 16);
        for (k, &bit_col) in layout.rs2_low_bit.iter().enumerate() {
            wit.cols[bit_col][i] = F::from_u64((rs2_val_u64 >> k) & 1);
        }
    }

    for (i, r) in exec.rows.iter().enumerate() {
        if !r.active {
            continue;
        }
        let mut read_value: Option<u64> = None;
        for e in &r.ram_events {
            if e.kind == neo_vm_trace::TwistOpKind::Read {
                read_value = Some(e.value);
                break;
            }
        }
        if let Some(rv) = read_value {
            wit.cols[layout.ram_rv_q16][i] = F::from_u64(rv >> 16);
            for (k, &bit_col) in layout.ram_rv_low_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rv >> k) & 1);
            }
        }
    }

    wit
}
