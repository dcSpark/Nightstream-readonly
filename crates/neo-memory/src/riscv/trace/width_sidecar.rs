use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;

/// Deterministic width sidecar identifier for RV32 Trace Track-A W3.
pub const RV32_TRACE_W3_WIDTH_ID: u32 = 0x5256_5733;

#[derive(Clone, Debug)]
pub struct Rv32WidthSidecarLayout {
    pub cols: usize,
    pub is_lb: usize,
    pub is_lbu: usize,
    pub is_lh: usize,
    pub is_lhu: usize,
    pub is_lw: usize,
    pub is_sb: usize,
    pub is_sh: usize,
    pub is_sw: usize,
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

        let is_lb = take();
        let is_lbu = take();
        let is_lh = take();
        let is_lhu = take();
        let is_lw = take();
        let is_sb = take();
        let is_sh = take();
        let is_sw = take();
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

        debug_assert_eq!(next, 42);
        Self {
            cols: next,
            is_lb,
            is_lbu,
            is_lh,
            is_lhu,
            is_lw,
            is_sb,
            is_sh,
            is_sw,
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

        let opcode_u64 = cols.opcode[i] as u64;
        let funct3_u64 = cols.funct3[i] as u64;
        let is_load = opcode_u64 == 0x03;
        let is_store = opcode_u64 == 0x23;
        let flag = |on: bool| if on { F::ONE } else { F::ZERO };

        let is_lb = is_load && funct3_u64 == 0b000;
        let is_lh = is_load && funct3_u64 == 0b001;
        let is_lw = is_load && funct3_u64 == 0b010;
        let is_lbu = is_load && funct3_u64 == 0b100;
        let is_lhu = is_load && funct3_u64 == 0b101;
        let is_sb = is_store && funct3_u64 == 0b000;
        let is_sh = is_store && funct3_u64 == 0b001;
        let is_sw = is_store && funct3_u64 == 0b010;

        wit.cols[layout.is_lb][i] = flag(is_lb);
        wit.cols[layout.is_lbu][i] = flag(is_lbu);
        wit.cols[layout.is_lh][i] = flag(is_lh);
        wit.cols[layout.is_lhu][i] = flag(is_lhu);
        wit.cols[layout.is_lw][i] = flag(is_lw);
        wit.cols[layout.is_sb][i] = flag(is_sb);
        wit.cols[layout.is_sh][i] = flag(is_sh);
        wit.cols[layout.is_sw][i] = flag(is_sw);

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

pub fn build_rv32_width_sidecar_z(
    layout: &Rv32WidthSidecarLayout,
    wit: &Rv32WidthSidecarWitness,
    m: usize,
    m_in: usize,
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if x_prefix.len() != m_in {
        return Err(format!(
            "width sidecar: x_prefix.len()={} != m_in={m_in}",
            x_prefix.len()
        ));
    }
    if wit.cols.len() != layout.cols {
        return Err(format!(
            "width sidecar: witness width mismatch (got {}, expected {})",
            wit.cols.len(),
            layout.cols
        ));
    }
    if wit.t == 0 {
        return Err("width sidecar: t must be >= 1".into());
    }
    let sidecar_span = layout
        .cols
        .checked_mul(wit.t)
        .ok_or_else(|| "width sidecar: cols*t overflow".to_string())?;
    let end = m_in
        .checked_add(sidecar_span)
        .ok_or_else(|| "width sidecar: m_in + cols*t overflow".to_string())?;
    if end > m {
        return Err(format!(
            "width sidecar: matrix too small (need at least {end}, got {m})"
        ));
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);
    for col in 0..layout.cols {
        let col_start = m_in + col * wit.t;
        for row in 0..wit.t {
            z[col_start + row] = wit.cols[col][row];
        }
    }
    Ok(z)
}
