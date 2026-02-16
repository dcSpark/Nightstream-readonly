use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;

/// Deterministic decode sidecar identifier for RV32 Trace Track-A W2.
pub const RV32_TRACE_W2_DECODE_ID: u32 = 0x5256_3332;

#[derive(Clone, Debug)]
pub struct Rv32DecodeSidecarLayout {
    pub cols: usize,
    pub funct7: usize,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub op_lui: usize,
    pub op_auipc: usize,
    pub op_jal: usize,
    pub op_jalr: usize,
    pub op_branch: usize,
    pub op_load: usize,
    pub op_store: usize,
    pub op_alu_imm: usize,
    pub op_alu_reg: usize,
    pub op_misc_mem: usize,
    pub op_system: usize,
    pub op_amo: usize,
    pub op_lui_write: usize,
    pub op_auipc_write: usize,
    pub op_jal_write: usize,
    pub op_jalr_write: usize,
    pub op_alu_imm_write: usize,
    pub op_alu_reg_write: usize,
    pub is_lb_write: usize,
    pub is_lbu_write: usize,
    pub is_lh_write: usize,
    pub is_lhu_write: usize,
    pub is_lw_write: usize,
    pub funct3_is: [usize; 8],
    pub alu_reg_table_delta: usize,
    pub alu_imm_table_delta: usize,
    pub alu_imm_shift_rhs_delta: usize,
    pub imm_i: usize,
    pub imm_s: usize,
    pub imm_b: usize,
    pub imm_j: usize,
}

impl Rv32DecodeSidecarLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };
        let funct7 = take();
        let rd = take();
        let rs1 = take();
        let rs2 = take();
        let op_lui = take();
        let op_auipc = take();
        let op_jal = take();
        let op_jalr = take();
        let op_branch = take();
        let op_load = take();
        let op_store = take();
        let op_alu_imm = take();
        let op_alu_reg = take();
        let op_misc_mem = take();
        let op_system = take();
        let op_amo = take();
        let op_lui_write = take();
        let op_auipc_write = take();
        let op_jal_write = take();
        let op_jalr_write = take();
        let op_alu_imm_write = take();
        let op_alu_reg_write = take();
        let is_lb_write = take();
        let is_lbu_write = take();
        let is_lh_write = take();
        let is_lhu_write = take();
        let is_lw_write = take();
        let funct3_is_0 = take();
        let funct3_is_1 = take();
        let funct3_is_2 = take();
        let funct3_is_3 = take();
        let funct3_is_4 = take();
        let funct3_is_5 = take();
        let funct3_is_6 = take();
        let funct3_is_7 = take();
        let alu_reg_table_delta = take();
        let alu_imm_table_delta = take();
        let alu_imm_shift_rhs_delta = take();
        let imm_i = take();
        let imm_s = take();
        let imm_b = take();
        let imm_j = take();
        debug_assert_eq!(next, 42);
        Self {
            cols: next,
            funct7,
            rd,
            rs1,
            rs2,
            op_lui,
            op_auipc,
            op_jal,
            op_jalr,
            op_branch,
            op_load,
            op_store,
            op_alu_imm,
            op_alu_reg,
            op_misc_mem,
            op_system,
            op_amo,
            op_lui_write,
            op_auipc_write,
            op_jal_write,
            op_jalr_write,
            op_alu_imm_write,
            op_alu_reg_write,
            is_lb_write,
            is_lbu_write,
            is_lh_write,
            is_lhu_write,
            is_lw_write,
            funct3_is: [
                funct3_is_0,
                funct3_is_1,
                funct3_is_2,
                funct3_is_3,
                funct3_is_4,
                funct3_is_5,
                funct3_is_6,
                funct3_is_7,
            ],
            alu_reg_table_delta,
            alu_imm_table_delta,
            alu_imm_shift_rhs_delta,
            imm_i,
            imm_s,
            imm_b,
            imm_j,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32DecodeSidecarWitness {
    pub t: usize,
    pub cols: Vec<Vec<F>>,
}

impl Rv32DecodeSidecarWitness {
    pub fn new_zero(layout: &Rv32DecodeSidecarLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }
}

#[inline]
fn sign_extend_to_u32(value: u32, bits: u32) -> u32 {
    debug_assert!(bits > 0 && bits <= 32);
    let shift = 32 - bits;
    (((value << shift) as i32) >> shift) as u32
}

#[inline]
fn imm_i_from_word(instr_word: u32) -> u32 {
    sign_extend_to_u32((instr_word >> 20) & 0x0fff, 12)
}

#[inline]
fn imm_s_from_word(instr_word: u32) -> u32 {
    let imm = ((instr_word >> 7) & 0x1f) | (((instr_word >> 25) & 0x7f) << 5);
    sign_extend_to_u32(imm, 12)
}

#[inline]
fn imm_b_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 12)
        | (((instr_word >> 7) & 0x1) << 11)
        | (((instr_word >> 25) & 0x3f) << 5)
        | (((instr_word >> 8) & 0xf) << 1);
    sign_extend_to_u32(imm, 13)
}

#[inline]
fn imm_j_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 20)
        | (((instr_word >> 12) & 0xff) << 12)
        | (((instr_word >> 20) & 0x1) << 11)
        | (((instr_word >> 21) & 0x3ff) << 1);
    sign_extend_to_u32(imm, 21)
}

pub fn rv32_decode_sidecar_witness_from_exec_table(
    layout: &Rv32DecodeSidecarLayout,
    exec: &Rv32ExecTable,
) -> Rv32DecodeSidecarWitness {
    let cols = exec.to_columns();
    let t = cols.len();
    let mut wit = Rv32DecodeSidecarWitness::new_zero(layout, t);

    for i in 0..t {
        let instr_word = cols.instr_word[i];
        let opcode_u64 = cols.opcode[i] as u64;
        let funct3_u64 = cols.funct3[i] as u64;
        let funct7_u64 = cols.funct7[i] as u64;
        let rd_u64 = cols.rd[i] as u64;
        let rs1_u64 = cols.rs1[i] as u64;
        let rs2_u64 = cols.rs2[i] as u64;
        let active = cols.active[i];
        let rd_has_write = cols.rd_has_write[i];

        wit.cols[layout.funct7][i] = F::from_u64(funct7_u64);
        wit.cols[layout.rd][i] = F::from_u64(rd_u64);
        wit.cols[layout.rs1][i] = F::from_u64(rs1_u64);
        wit.cols[layout.rs2][i] = F::from_u64(rs2_u64);
        wit.cols[layout.imm_i][i] = F::from_u64(imm_i_from_word(instr_word) as u64);
        wit.cols[layout.imm_s][i] = F::from_u64(imm_s_from_word(instr_word) as u64);
        wit.cols[layout.imm_b][i] = F::from_u64(imm_b_from_word(instr_word) as u64);
        wit.cols[layout.imm_j][i] = F::from_u64(imm_j_from_word(instr_word) as u64);

        let is = |op: u64| if opcode_u64 == op { F::ONE } else { F::ZERO };
        wit.cols[layout.op_lui][i] = is(0x37);
        wit.cols[layout.op_auipc][i] = is(0x17);
        wit.cols[layout.op_jal][i] = is(0x6F);
        wit.cols[layout.op_jalr][i] = is(0x67);
        wit.cols[layout.op_branch][i] = is(0x63);
        wit.cols[layout.op_load][i] = is(0x03);
        wit.cols[layout.op_store][i] = is(0x23);
        wit.cols[layout.op_alu_imm][i] = is(0x13);
        wit.cols[layout.op_alu_reg][i] = is(0x33);
        wit.cols[layout.op_misc_mem][i] = is(0x0F);
        wit.cols[layout.op_system][i] = is(0x73);
        wit.cols[layout.op_amo][i] = is(0x2F);

        let rd_has_write_f = if rd_has_write { F::ONE } else { F::ZERO };
        wit.cols[layout.op_lui_write][i] = wit.cols[layout.op_lui][i] * rd_has_write_f;
        wit.cols[layout.op_auipc_write][i] = wit.cols[layout.op_auipc][i] * rd_has_write_f;
        wit.cols[layout.op_jal_write][i] = wit.cols[layout.op_jal][i] * rd_has_write_f;
        wit.cols[layout.op_jalr_write][i] = wit.cols[layout.op_jalr][i] * rd_has_write_f;
        wit.cols[layout.op_alu_imm_write][i] = wit.cols[layout.op_alu_imm][i] * rd_has_write_f;
        wit.cols[layout.op_alu_reg_write][i] = wit.cols[layout.op_alu_reg][i] * rd_has_write_f;

        let is_load = opcode_u64 == 0x03;
        let is_lb = is_load && funct3_u64 == 0b000;
        let is_lh = is_load && funct3_u64 == 0b001;
        let is_lw = is_load && funct3_u64 == 0b010;
        let is_lbu = is_load && funct3_u64 == 0b100;
        let is_lhu = is_load && funct3_u64 == 0b101;
        let flag = |on: bool| if on { F::ONE } else { F::ZERO };
        wit.cols[layout.is_lb_write][i] = flag(is_lb) * rd_has_write_f;
        wit.cols[layout.is_lbu_write][i] = flag(is_lbu) * rd_has_write_f;
        wit.cols[layout.is_lh_write][i] = flag(is_lh) * rd_has_write_f;
        wit.cols[layout.is_lhu_write][i] = flag(is_lhu) * rd_has_write_f;
        wit.cols[layout.is_lw_write][i] = flag(is_lw) * rd_has_write_f;

        for (k, &f3_col) in layout.funct3_is.iter().enumerate() {
            wit.cols[f3_col][i] = if active && funct3_u64 == k as u64 {
                F::ONE
            } else {
                F::ZERO
            };
        }

        let funct7_b5 = (funct7_u64 >> 5) & 1;
        let f3_is_0 = if active && funct3_u64 == 0 { 1 } else { 0 };
        let f3_is_5 = if active && funct3_u64 == 5 { 1 } else { 0 };
        wit.cols[layout.alu_reg_table_delta][i] = F::from_u64(funct7_b5 * (f3_is_0 + f3_is_5));
        wit.cols[layout.alu_imm_table_delta][i] = F::from_u64(funct7_b5 * f3_is_5);

        let shift_f3_sel = wit.cols[layout.funct3_is[1]][i] + wit.cols[layout.funct3_is[5]][i];
        wit.cols[layout.alu_imm_shift_rhs_delta][i] =
            shift_f3_sel * (F::from_u64(rs2_u64) - wit.cols[layout.imm_i][i]);
    }

    wit
}

pub fn build_rv32_decode_sidecar_z(
    layout: &Rv32DecodeSidecarLayout,
    wit: &Rv32DecodeSidecarWitness,
    m: usize,
    m_in: usize,
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if x_prefix.len() != m_in {
        return Err(format!(
            "decode sidecar: x_prefix.len()={} != m_in={m_in}",
            x_prefix.len()
        ));
    }
    if wit.cols.len() != layout.cols {
        return Err(format!(
            "decode sidecar: witness width mismatch (got {}, expected {})",
            wit.cols.len(),
            layout.cols
        ));
    }
    if wit.t == 0 {
        return Err("decode sidecar: t must be >= 1".into());
    }
    let decode_span = layout
        .cols
        .checked_mul(wit.t)
        .ok_or_else(|| "decode sidecar: cols*t overflow".to_string())?;
    let end = m_in
        .checked_add(decode_span)
        .ok_or_else(|| "decode sidecar: m_in + cols*t overflow".to_string())?;
    if end > m {
        return Err(format!(
            "decode sidecar: matrix too small (need at least {end}, got {m})"
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
