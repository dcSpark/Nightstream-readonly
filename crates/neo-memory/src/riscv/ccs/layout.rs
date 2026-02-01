use std::collections::HashMap;

use crate::cpu::bus_layout::BusLayout;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};

use super::config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};

/// Witness/column layout for the RV32 B1 step circuit.
#[derive(Clone, Debug)]
pub struct Rv32B1Layout {
    pub m_in: usize,
    pub m: usize,
    pub chunk_size: usize,
    pub const_one: usize,
    // Public I/O (single values per chunk).
    pub pc0: usize,
    pub pc_final: usize,
    pub halted_in: usize,
    pub halted_out: usize,
    pub is_active: usize,
    /// A dedicated all-zero CPU column (used to safely disable bus lanes).
    pub zero: usize,

    pub pc_in: usize,
    pub pc_out: usize,
    pub instr_word: usize,

    pub instr_bits_start: usize, // 32 bits

    pub opcode: usize,
    pub funct3: usize,
    pub funct7: usize,
    pub rd_field: usize,
    pub rs1_field: usize,
    pub rs2_field: usize,

    pub imm12_raw: usize,
    pub imm_i: usize,
    pub imm_s: usize,
    pub imm_u: usize,
    pub imm_b_raw: usize,
    pub imm_b: usize,
    pub imm_j_raw: usize,
    pub imm_j: usize,

    // One-hot instruction flags (sum == is_active).
    pub is_add: usize,
    pub is_sub: usize,
    pub is_sll: usize,
    pub is_slt: usize,
    pub is_sltu: usize,
    pub is_xor: usize,
    pub is_srl: usize,
    pub is_sra: usize,
    pub is_or: usize,
    pub is_and: usize,

    // RV32M (R-type, funct7=0b0000001).
    pub is_mul: usize,
    pub is_mulh: usize,
    pub is_mulhu: usize,
    pub is_mulhsu: usize,
    pub is_div: usize,
    pub is_divu: usize,
    pub is_rem: usize,
    pub is_remu: usize,

    pub is_addi: usize,
    pub is_slti: usize,
    pub is_sltiu: usize,
    pub is_xori: usize,
    pub is_ori: usize,
    pub is_andi: usize,
    pub is_slli: usize,
    pub is_srli: usize,
    pub is_srai: usize,

    pub is_lb: usize,
    pub is_lbu: usize,
    pub is_lh: usize,
    pub is_lhu: usize,
    pub is_lw: usize,
    pub is_sb: usize,
    pub is_sh: usize,
    pub is_sw: usize,
    // RV32A (atomics, word only).
    pub is_amoswap_w: usize,
    pub is_amoadd_w: usize,
    pub is_amoxor_w: usize,
    pub is_amoor_w: usize,
    pub is_amoand_w: usize,
    pub is_lui: usize,
    pub is_auipc: usize,
    pub is_beq: usize,
    pub is_bne: usize,
    pub is_blt: usize,
    pub is_bge: usize,
    pub is_bltu: usize,
    pub is_bgeu: usize,
    pub is_jal: usize,
    pub is_jalr: usize,
    pub is_fence: usize,
    pub is_halt: usize,

    pub br_taken: usize,
    pub br_not_taken: usize,

    pub rs1_val: usize,
    pub rs2_val: usize,

    pub alu_out: usize,
    pub mem_rv: usize,
    pub mem_rv_bits_start: usize, // 32
    pub eff_addr: usize,
    // RAM bus selectors/values (must be tied to instruction flags to avoid bypassing Twist).
    pub ram_has_read: usize,
    pub ram_has_write: usize,
    pub ram_wv: usize,
    pub rd_write_val: usize,
    pub rd_write_bits_start: usize, // 32

    pub add_has_lookup: usize,
    pub and_has_lookup: usize,
    pub xor_has_lookup: usize,
    pub or_has_lookup: usize,
    pub sll_has_lookup: usize,
    pub srl_has_lookup: usize,
    pub sra_has_lookup: usize,
    pub slt_has_lookup: usize,
    pub sltu_has_lookup: usize,
    pub lookup_key: usize,
    pub add_a0b0: usize,

    // In-circuit RV32M helpers (avoid requiring implicit Shout tables).
    // MUL* helpers: rs1_val * rs2_val = mul_lo + 2^32 * mul_hi
    pub mul_lo: usize,
    pub mul_hi: usize,
    pub mul_lo_bits_start: usize,   // 32
    pub mul_hi_bits_start: usize,   // 32
    pub mul_hi_prefix_start: usize, // 31
    pub mul_carry: usize,
    pub mul_carry_bits_start: usize, // 2

    // Signed helpers: rs1/rs2 bits + absolute values.
    pub rs1_bits_start: usize,        // 32
    pub rs2_bits_start: usize,        // 32
    pub rs2_zero_prefix_start: usize, // 31
    pub rs1_abs: usize,
    pub rs2_abs: usize,
    pub rs1_rs2_sign_and: usize,
    pub rs1_sign_rs2_val: usize,
    pub rs2_sign_rs1_val: usize,

    // DIV/REM helpers (unsigned + signed).
    pub div_quot: usize,
    pub div_rem: usize,
    pub div_quot_signed: usize,
    pub div_rem_signed: usize,
    pub div_quot_carry: usize,
    pub div_rem_carry: usize,
    pub div_prod: usize,
    pub div_divisor: usize,
    pub rs2_is_zero: usize,
    pub rs2_nonzero: usize,
    pub is_divu_or_remu: usize,
    pub divu_by_zero: usize,
    pub is_div_or_rem: usize,
    pub div_nonzero: usize,
    pub rem_nonzero: usize,
    pub div_by_zero: usize,
    pub rem_by_zero: usize,
    pub div_sign: usize,
    pub div_rem_check: usize,
    pub div_rem_check_signed: usize,
    // ECALL helpers (Jolt marker/print IDs).
    pub ecall_a0_bits_start: usize,      // 32
    pub ecall_cycle_prefix_start: usize, // 31
    pub ecall_is_cycle: usize,
    pub ecall_print_prefix_start: usize, // 31
    pub ecall_is_print: usize,
    pub ecall_halts: usize,
    pub halt_effective: usize,

    // Regfile-as-Twist glue.
    pub reg_has_write: usize,
    pub reg_rs2_addr: usize,
    pub rd_is_zero_01: usize,
    pub rd_is_zero_012: usize,
    pub rd_is_zero_0123: usize,
    pub rd_is_zero: usize,

    pub bus: BusLayout,
    pub mem_ids: Vec<u32>,
    pub table_ids: Vec<u32>,
    pub ram_twist_idx: usize,
    pub prog_twist_idx: usize,
    pub reg_twist_idx: usize,
}

impl Rv32B1Layout {
    #[inline]
    fn cpu_cell(&self, base: usize, j: usize) -> usize {
        debug_assert!(j < self.chunk_size, "cpu j out of range");
        base + j
    }

    #[inline]
    pub fn is_active(&self, j: usize) -> usize {
        self.cpu_cell(self.is_active, j)
    }

    #[inline]
    pub fn pc_in(&self, j: usize) -> usize {
        self.cpu_cell(self.pc_in, j)
    }

    #[inline]
    pub fn pc_out(&self, j: usize) -> usize {
        self.cpu_cell(self.pc_out, j)
    }

    #[inline]
    pub fn instr_word(&self, j: usize) -> usize {
        self.cpu_cell(self.instr_word, j)
    }

    #[inline]
    pub fn zero(&self, j: usize) -> usize {
        self.cpu_cell(self.zero, j)
    }

    pub fn instr_bit(&self, i: usize, j: usize) -> usize {
        assert!(i < 32);
        self.instr_bits_start + i * self.chunk_size + j
    }

    #[inline]
    pub fn reg_has_write(&self, j: usize) -> usize {
        self.cpu_cell(self.reg_has_write, j)
    }

    #[inline]
    pub fn reg_rs2_addr(&self, j: usize) -> usize {
        self.cpu_cell(self.reg_rs2_addr, j)
    }

    #[inline]
    pub fn rd_is_zero(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_is_zero, j)
    }

    #[inline]
    pub fn rd_is_zero_01(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_is_zero_01, j)
    }

    #[inline]
    pub fn rd_is_zero_012(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_is_zero_012, j)
    }

    #[inline]
    pub fn rd_is_zero_0123(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_is_zero_0123, j)
    }

    #[inline]
    pub fn rs1_val(&self, j: usize) -> usize {
        self.cpu_cell(self.rs1_val, j)
    }

    #[inline]
    pub fn rs2_val(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_val, j)
    }

    #[inline]
    pub fn alu_out(&self, j: usize) -> usize {
        self.cpu_cell(self.alu_out, j)
    }

    #[inline]
    pub fn mem_rv(&self, j: usize) -> usize {
        self.cpu_cell(self.mem_rv, j)
    }

    pub fn mem_rv_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.mem_rv_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn eff_addr(&self, j: usize) -> usize {
        self.cpu_cell(self.eff_addr, j)
    }

    #[inline]
    pub fn ram_has_read(&self, j: usize) -> usize {
        self.cpu_cell(self.ram_has_read, j)
    }

    #[inline]
    pub fn ram_has_write(&self, j: usize) -> usize {
        self.cpu_cell(self.ram_has_write, j)
    }

    #[inline]
    pub fn ram_wv(&self, j: usize) -> usize {
        self.cpu_cell(self.ram_wv, j)
    }

    #[inline]
    pub fn rd_write_val(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_write_val, j)
    }

    pub fn rd_write_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.rd_write_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn lookup_key(&self, j: usize) -> usize {
        self.cpu_cell(self.lookup_key, j)
    }

    #[inline]
    pub fn add_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.add_has_lookup, j)
    }

    #[inline]
    pub fn and_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.and_has_lookup, j)
    }

    #[inline]
    pub fn xor_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.xor_has_lookup, j)
    }

    #[inline]
    pub fn or_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.or_has_lookup, j)
    }

    #[inline]
    pub fn mul_lo(&self, j: usize) -> usize {
        self.cpu_cell(self.mul_lo, j)
    }

    pub fn mul_lo_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.mul_lo_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn mul_hi(&self, j: usize) -> usize {
        self.cpu_cell(self.mul_hi, j)
    }

    pub fn mul_hi_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.mul_hi_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn mul_hi_prefix(&self, k: usize, j: usize) -> usize {
        assert!(k < 31);
        self.mul_hi_prefix_start + k * self.chunk_size + j
    }

    #[inline]
    pub fn mul_carry(&self, j: usize) -> usize {
        self.cpu_cell(self.mul_carry, j)
    }

    pub fn mul_carry_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 2);
        self.mul_carry_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn div_quot(&self, j: usize) -> usize {
        self.cpu_cell(self.div_quot, j)
    }

    #[inline]
    pub fn div_rem(&self, j: usize) -> usize {
        self.cpu_cell(self.div_rem, j)
    }

    #[inline]
    pub fn div_quot_signed(&self, j: usize) -> usize {
        self.cpu_cell(self.div_quot_signed, j)
    }

    #[inline]
    pub fn div_rem_signed(&self, j: usize) -> usize {
        self.cpu_cell(self.div_rem_signed, j)
    }

    #[inline]
    pub fn div_quot_carry(&self, j: usize) -> usize {
        self.cpu_cell(self.div_quot_carry, j)
    }

    #[inline]
    pub fn div_rem_carry(&self, j: usize) -> usize {
        self.cpu_cell(self.div_rem_carry, j)
    }

    #[inline]
    pub fn div_prod(&self, j: usize) -> usize {
        self.cpu_cell(self.div_prod, j)
    }

    #[inline]
    pub fn div_divisor(&self, j: usize) -> usize {
        self.cpu_cell(self.div_divisor, j)
    }

    pub fn rs1_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.rs1_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn rs2_is_zero(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_is_zero, j)
    }

    pub fn rs2_bit(&self, bit: usize, j: usize) -> usize {
        assert!(bit < 32);
        self.rs2_bits_start + bit * self.chunk_size + j
    }

    pub fn rs2_zero_prefix(&self, idx: usize, j: usize) -> usize {
        assert!(idx < 31);
        self.rs2_zero_prefix_start + idx * self.chunk_size + j
    }

    #[inline]
    pub fn rs2_nonzero(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_nonzero, j)
    }

    #[inline]
    pub fn rs1_abs(&self, j: usize) -> usize {
        self.cpu_cell(self.rs1_abs, j)
    }

    #[inline]
    pub fn rs2_abs(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_abs, j)
    }

    #[inline]
    pub fn rs1_rs2_sign_and(&self, j: usize) -> usize {
        self.cpu_cell(self.rs1_rs2_sign_and, j)
    }

    #[inline]
    pub fn rs1_sign_rs2_val(&self, j: usize) -> usize {
        self.cpu_cell(self.rs1_sign_rs2_val, j)
    }

    #[inline]
    pub fn rs2_sign_rs1_val(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_sign_rs1_val, j)
    }

    #[inline]
    pub fn is_divu_or_remu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_divu_or_remu, j)
    }

    #[inline]
    pub fn divu_by_zero(&self, j: usize) -> usize {
        self.cpu_cell(self.divu_by_zero, j)
    }

    #[inline]
    pub fn is_div_or_rem(&self, j: usize) -> usize {
        self.cpu_cell(self.is_div_or_rem, j)
    }

    #[inline]
    pub fn div_nonzero(&self, j: usize) -> usize {
        self.cpu_cell(self.div_nonzero, j)
    }

    #[inline]
    pub fn rem_nonzero(&self, j: usize) -> usize {
        self.cpu_cell(self.rem_nonzero, j)
    }

    #[inline]
    pub fn div_by_zero(&self, j: usize) -> usize {
        self.cpu_cell(self.div_by_zero, j)
    }

    #[inline]
    pub fn rem_by_zero(&self, j: usize) -> usize {
        self.cpu_cell(self.rem_by_zero, j)
    }

    #[inline]
    pub fn div_sign(&self, j: usize) -> usize {
        self.cpu_cell(self.div_sign, j)
    }

    #[inline]
    pub fn div_rem_check(&self, j: usize) -> usize {
        self.cpu_cell(self.div_rem_check, j)
    }

    #[inline]
    pub fn div_rem_check_signed(&self, j: usize) -> usize {
        self.cpu_cell(self.div_rem_check_signed, j)
    }

    #[inline]
    pub fn ecall_a0_bit(&self, bit: usize, j: usize) -> usize {
        debug_assert!(bit < 32, "a0 bit out of range");
        self.ecall_a0_bits_start + bit * self.chunk_size + j
    }

    #[inline]
    pub fn ecall_cycle_prefix(&self, k: usize, j: usize) -> usize {
        debug_assert!(k < 31, "ecall_cycle_prefix k out of range");
        self.ecall_cycle_prefix_start + k * self.chunk_size + j
    }

    #[inline]
    pub fn ecall_is_cycle(&self, j: usize) -> usize {
        self.cpu_cell(self.ecall_is_cycle, j)
    }

    #[inline]
    pub fn ecall_print_prefix(&self, k: usize, j: usize) -> usize {
        debug_assert!(k < 31, "ecall_print_prefix k out of range");
        self.ecall_print_prefix_start + k * self.chunk_size + j
    }

    #[inline]
    pub fn ecall_is_print(&self, j: usize) -> usize {
        self.cpu_cell(self.ecall_is_print, j)
    }

    #[inline]
    pub fn ecall_halts(&self, j: usize) -> usize {
        self.cpu_cell(self.ecall_halts, j)
    }

    #[inline]
    pub fn halt_effective(&self, j: usize) -> usize {
        self.cpu_cell(self.halt_effective, j)
    }

    #[inline]
    pub fn sll_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.sll_has_lookup, j)
    }

    #[inline]
    pub fn srl_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.srl_has_lookup, j)
    }

    #[inline]
    pub fn sra_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.sra_has_lookup, j)
    }

    #[inline]
    pub fn slt_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.slt_has_lookup, j)
    }

    #[inline]
    pub fn sltu_has_lookup(&self, j: usize) -> usize {
        self.cpu_cell(self.sltu_has_lookup, j)
    }

    #[inline]
    pub fn add_a0b0(&self, j: usize) -> usize {
        self.cpu_cell(self.add_a0b0, j)
    }

    #[inline]
    pub fn imm12_raw(&self, j: usize) -> usize {
        self.cpu_cell(self.imm12_raw, j)
    }

    #[inline]
    pub fn imm_i(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_i, j)
    }

    #[inline]
    pub fn imm_s(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_s, j)
    }

    #[inline]
    pub fn imm_u(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_u, j)
    }

    #[inline]
    pub fn imm_b_raw(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_b_raw, j)
    }

    #[inline]
    pub fn imm_b(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_b, j)
    }

    #[inline]
    pub fn imm_j_raw(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_j_raw, j)
    }

    #[inline]
    pub fn imm_j(&self, j: usize) -> usize {
        self.cpu_cell(self.imm_j, j)
    }

    #[inline]
    pub fn shamt(&self, j: usize) -> usize {
        // Shift amount lives in the same 5-bit field as `rs2_field` (instr bits [24:20]).
        self.rs2_field(j)
    }

    #[inline]
    pub fn opcode(&self, j: usize) -> usize {
        self.cpu_cell(self.opcode, j)
    }

    #[inline]
    pub fn funct3(&self, j: usize) -> usize {
        self.cpu_cell(self.funct3, j)
    }

    #[inline]
    pub fn funct7(&self, j: usize) -> usize {
        self.cpu_cell(self.funct7, j)
    }

    #[inline]
    pub fn rd_field(&self, j: usize) -> usize {
        self.cpu_cell(self.rd_field, j)
    }

    #[inline]
    pub fn rs1_field(&self, j: usize) -> usize {
        self.cpu_cell(self.rs1_field, j)
    }

    #[inline]
    pub fn rs2_field(&self, j: usize) -> usize {
        self.cpu_cell(self.rs2_field, j)
    }

    #[inline]
    pub fn is_add(&self, j: usize) -> usize {
        self.cpu_cell(self.is_add, j)
    }

    #[inline]
    pub fn is_sub(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sub, j)
    }

    #[inline]
    pub fn is_sll(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sll, j)
    }

    #[inline]
    pub fn is_slt(&self, j: usize) -> usize {
        self.cpu_cell(self.is_slt, j)
    }

    #[inline]
    pub fn is_sltu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sltu, j)
    }

    #[inline]
    pub fn is_xor(&self, j: usize) -> usize {
        self.cpu_cell(self.is_xor, j)
    }

    #[inline]
    pub fn is_srl(&self, j: usize) -> usize {
        self.cpu_cell(self.is_srl, j)
    }

    #[inline]
    pub fn is_sra(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sra, j)
    }

    #[inline]
    pub fn is_or(&self, j: usize) -> usize {
        self.cpu_cell(self.is_or, j)
    }

    #[inline]
    pub fn is_and(&self, j: usize) -> usize {
        self.cpu_cell(self.is_and, j)
    }

    #[inline]
    pub fn is_mul(&self, j: usize) -> usize {
        self.cpu_cell(self.is_mul, j)
    }

    #[inline]
    pub fn is_mulh(&self, j: usize) -> usize {
        self.cpu_cell(self.is_mulh, j)
    }

    #[inline]
    pub fn is_mulhu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_mulhu, j)
    }

    #[inline]
    pub fn is_mulhsu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_mulhsu, j)
    }

    #[inline]
    pub fn is_div(&self, j: usize) -> usize {
        self.cpu_cell(self.is_div, j)
    }

    #[inline]
    pub fn is_divu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_divu, j)
    }

    #[inline]
    pub fn is_rem(&self, j: usize) -> usize {
        self.cpu_cell(self.is_rem, j)
    }

    #[inline]
    pub fn is_remu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_remu, j)
    }

    #[inline]
    pub fn is_addi(&self, j: usize) -> usize {
        self.cpu_cell(self.is_addi, j)
    }

    #[inline]
    pub fn is_slti(&self, j: usize) -> usize {
        self.cpu_cell(self.is_slti, j)
    }

    #[inline]
    pub fn is_sltiu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sltiu, j)
    }

    #[inline]
    pub fn is_xori(&self, j: usize) -> usize {
        self.cpu_cell(self.is_xori, j)
    }

    #[inline]
    pub fn is_ori(&self, j: usize) -> usize {
        self.cpu_cell(self.is_ori, j)
    }

    #[inline]
    pub fn is_andi(&self, j: usize) -> usize {
        self.cpu_cell(self.is_andi, j)
    }

    #[inline]
    pub fn is_slli(&self, j: usize) -> usize {
        self.cpu_cell(self.is_slli, j)
    }

    #[inline]
    pub fn is_srli(&self, j: usize) -> usize {
        self.cpu_cell(self.is_srli, j)
    }

    #[inline]
    pub fn is_srai(&self, j: usize) -> usize {
        self.cpu_cell(self.is_srai, j)
    }

    #[inline]
    pub fn is_lb(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lb, j)
    }

    #[inline]
    pub fn is_lbu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lbu, j)
    }

    #[inline]
    pub fn is_lh(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lh, j)
    }

    #[inline]
    pub fn is_lhu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lhu, j)
    }

    #[inline]
    pub fn is_lw(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lw, j)
    }

    #[inline]
    pub fn is_sb(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sb, j)
    }

    #[inline]
    pub fn is_sh(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sh, j)
    }

    #[inline]
    pub fn is_sw(&self, j: usize) -> usize {
        self.cpu_cell(self.is_sw, j)
    }

    #[inline]
    pub fn is_amoswap_w(&self, j: usize) -> usize {
        self.cpu_cell(self.is_amoswap_w, j)
    }

    #[inline]
    pub fn is_amoadd_w(&self, j: usize) -> usize {
        self.cpu_cell(self.is_amoadd_w, j)
    }

    #[inline]
    pub fn is_amoxor_w(&self, j: usize) -> usize {
        self.cpu_cell(self.is_amoxor_w, j)
    }

    #[inline]
    pub fn is_amoor_w(&self, j: usize) -> usize {
        self.cpu_cell(self.is_amoor_w, j)
    }

    #[inline]
    pub fn is_amoand_w(&self, j: usize) -> usize {
        self.cpu_cell(self.is_amoand_w, j)
    }

    #[inline]
    pub fn is_lui(&self, j: usize) -> usize {
        self.cpu_cell(self.is_lui, j)
    }

    #[inline]
    pub fn is_auipc(&self, j: usize) -> usize {
        self.cpu_cell(self.is_auipc, j)
    }

    #[inline]
    pub fn is_beq(&self, j: usize) -> usize {
        self.cpu_cell(self.is_beq, j)
    }

    #[inline]
    pub fn is_bne(&self, j: usize) -> usize {
        self.cpu_cell(self.is_bne, j)
    }

    #[inline]
    pub fn is_blt(&self, j: usize) -> usize {
        self.cpu_cell(self.is_blt, j)
    }

    #[inline]
    pub fn is_bge(&self, j: usize) -> usize {
        self.cpu_cell(self.is_bge, j)
    }

    #[inline]
    pub fn is_bltu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_bltu, j)
    }

    #[inline]
    pub fn is_bgeu(&self, j: usize) -> usize {
        self.cpu_cell(self.is_bgeu, j)
    }

    #[inline]
    pub fn is_jal(&self, j: usize) -> usize {
        self.cpu_cell(self.is_jal, j)
    }

    #[inline]
    pub fn is_jalr(&self, j: usize) -> usize {
        self.cpu_cell(self.is_jalr, j)
    }

    #[inline]
    pub fn is_fence(&self, j: usize) -> usize {
        self.cpu_cell(self.is_fence, j)
    }

    #[inline]
    pub fn is_halt(&self, j: usize) -> usize {
        self.cpu_cell(self.is_halt, j)
    }

    #[inline]
    pub fn br_taken(&self, j: usize) -> usize {
        self.cpu_cell(self.br_taken, j)
    }

    #[inline]
    pub fn br_not_taken(&self, j: usize) -> usize {
        self.cpu_cell(self.br_not_taken, j)
    }

    pub fn shout_idx(&self, table_id: u32) -> Result<usize, String> {
        self.table_ids
            .binary_search(&table_id)
            .map_err(|_| format!("RV32 B1: table_ids missing required table_id={table_id}"))
    }
}

pub(super) fn build_layout_with_m(
    m: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<Rv32B1Layout, String> {
    if chunk_size == 0 {
        return Err("RV32 B1 layout: chunk_size must be >= 1".into());
    }
    let const_one = 0usize;

    // Public inputs: boundary state for chunk chaining.
    // Layout: [const_one, pc0, pc_final, halted_in, halted_out]
    let pc0 = 1usize;
    let pc_final = pc0 + 1;
    let halted_in = pc_final + 1;
    let halted_out = halted_in + 1;
    let m_in = halted_out + 1;

    // Fixed CPU column allocation (CPU region only). All indices must be < bus.bus_base.
    let mut col = m_in;
    let alloc_scalar = |col: &mut usize| {
        let base = *col;
        *col += chunk_size;
        base
    };
    let alloc_array = |col: &mut usize, n: usize| {
        let base = *col;
        *col += n * chunk_size;
        base
    };

    let is_active = alloc_scalar(&mut col);
    let zero = alloc_scalar(&mut col);
    let pc_in = alloc_scalar(&mut col);
    let pc_out = alloc_scalar(&mut col);
    let instr_word = alloc_scalar(&mut col);

    // Regfile-as-Twist glue columns.
    let reg_has_write = alloc_scalar(&mut col);
    let reg_rs2_addr = alloc_scalar(&mut col);
    let rd_is_zero_01 = alloc_scalar(&mut col);
    let rd_is_zero_012 = alloc_scalar(&mut col);
    let rd_is_zero_0123 = alloc_scalar(&mut col);
    let rd_is_zero = alloc_scalar(&mut col);

    let instr_bits_start = alloc_array(&mut col, 32);

    let opcode = alloc_scalar(&mut col);
    let funct3 = alloc_scalar(&mut col);
    let funct7 = alloc_scalar(&mut col);
    let rd_field = alloc_scalar(&mut col);
    let rs1_field = alloc_scalar(&mut col);
    let rs2_field = alloc_scalar(&mut col);

    let imm12_raw = alloc_scalar(&mut col);
    let imm_i = alloc_scalar(&mut col);
    let imm_s = alloc_scalar(&mut col);
    let imm_u = alloc_scalar(&mut col);
    let imm_b_raw = alloc_scalar(&mut col);
    let imm_b = alloc_scalar(&mut col);
    let imm_j_raw = alloc_scalar(&mut col);
    let imm_j = alloc_scalar(&mut col);

    let is_add = alloc_scalar(&mut col);
    let is_sub = alloc_scalar(&mut col);
    let is_sll = alloc_scalar(&mut col);
    let is_slt = alloc_scalar(&mut col);
    let is_sltu = alloc_scalar(&mut col);
    let is_xor = alloc_scalar(&mut col);
    let is_srl = alloc_scalar(&mut col);
    let is_sra = alloc_scalar(&mut col);
    let is_or = alloc_scalar(&mut col);
    let is_and = alloc_scalar(&mut col);

    let is_mul = alloc_scalar(&mut col);
    let is_mulh = alloc_scalar(&mut col);
    let is_mulhu = alloc_scalar(&mut col);
    let is_mulhsu = alloc_scalar(&mut col);
    let is_div = alloc_scalar(&mut col);
    let is_divu = alloc_scalar(&mut col);
    let is_rem = alloc_scalar(&mut col);
    let is_remu = alloc_scalar(&mut col);

    let is_addi = alloc_scalar(&mut col);
    let is_slti = alloc_scalar(&mut col);
    let is_sltiu = alloc_scalar(&mut col);
    let is_xori = alloc_scalar(&mut col);
    let is_ori = alloc_scalar(&mut col);
    let is_andi = alloc_scalar(&mut col);
    let is_slli = alloc_scalar(&mut col);
    let is_srli = alloc_scalar(&mut col);
    let is_srai = alloc_scalar(&mut col);
    let is_lb = alloc_scalar(&mut col);
    let is_lbu = alloc_scalar(&mut col);
    let is_lh = alloc_scalar(&mut col);
    let is_lhu = alloc_scalar(&mut col);
    let is_lw = alloc_scalar(&mut col);
    let is_sb = alloc_scalar(&mut col);
    let is_sh = alloc_scalar(&mut col);
    let is_sw = alloc_scalar(&mut col);
    let is_amoswap_w = alloc_scalar(&mut col);
    let is_amoadd_w = alloc_scalar(&mut col);
    let is_amoxor_w = alloc_scalar(&mut col);
    let is_amoor_w = alloc_scalar(&mut col);
    let is_amoand_w = alloc_scalar(&mut col);
    let is_lui = alloc_scalar(&mut col);
    let is_auipc = alloc_scalar(&mut col);
    let is_beq = alloc_scalar(&mut col);
    let is_bne = alloc_scalar(&mut col);
    let is_blt = alloc_scalar(&mut col);
    let is_bge = alloc_scalar(&mut col);
    let is_bltu = alloc_scalar(&mut col);
    let is_bgeu = alloc_scalar(&mut col);
    let is_jal = alloc_scalar(&mut col);
    let is_jalr = alloc_scalar(&mut col);
    let is_fence = alloc_scalar(&mut col);
    let is_halt = alloc_scalar(&mut col);

    let br_taken = alloc_scalar(&mut col);
    let br_not_taken = alloc_scalar(&mut col);

    let rs1_val = alloc_scalar(&mut col);
    let rs2_val = alloc_scalar(&mut col);

    let alu_out = alloc_scalar(&mut col);
    let mem_rv = alloc_scalar(&mut col);
    let mem_rv_bits_start = alloc_array(&mut col, 32);
    let eff_addr = alloc_scalar(&mut col);
    let ram_has_read = alloc_scalar(&mut col);
    let ram_has_write = alloc_scalar(&mut col);
    let ram_wv = alloc_scalar(&mut col);
    let rd_write_val = alloc_scalar(&mut col);
    let rd_write_bits_start = alloc_array(&mut col, 32);

    let add_has_lookup = alloc_scalar(&mut col);
    let and_has_lookup = alloc_scalar(&mut col);
    let xor_has_lookup = alloc_scalar(&mut col);
    let or_has_lookup = alloc_scalar(&mut col);
    let sll_has_lookup = alloc_scalar(&mut col);
    let srl_has_lookup = alloc_scalar(&mut col);
    let sra_has_lookup = alloc_scalar(&mut col);
    let slt_has_lookup = alloc_scalar(&mut col);
    let sltu_has_lookup = alloc_scalar(&mut col);
    let lookup_key = alloc_scalar(&mut col);
    let add_a0b0 = alloc_scalar(&mut col);

    // In-circuit RV32M helpers.
    let mul_lo = alloc_scalar(&mut col);
    let mul_hi = alloc_scalar(&mut col);
    let mul_lo_bits_start = alloc_array(&mut col, 32);
    let mul_hi_bits_start = alloc_array(&mut col, 32);
    let mul_hi_prefix_start = alloc_array(&mut col, 31);
    let mul_carry = alloc_scalar(&mut col);
    let mul_carry_bits_start = alloc_array(&mut col, 2);

    let rs1_bits_start = alloc_array(&mut col, 32);
    let rs2_bits_start = alloc_array(&mut col, 32);
    let rs2_zero_prefix_start = alloc_array(&mut col, 31);
    let rs1_abs = alloc_scalar(&mut col);
    let rs2_abs = alloc_scalar(&mut col);
    let rs1_rs2_sign_and = alloc_scalar(&mut col);
    let rs1_sign_rs2_val = alloc_scalar(&mut col);
    let rs2_sign_rs1_val = alloc_scalar(&mut col);

    let div_quot = alloc_scalar(&mut col);
    let div_rem = alloc_scalar(&mut col);
    let div_quot_signed = alloc_scalar(&mut col);
    let div_rem_signed = alloc_scalar(&mut col);
    let div_quot_carry = alloc_scalar(&mut col);
    let div_rem_carry = alloc_scalar(&mut col);
    let div_prod = alloc_scalar(&mut col);
    let div_divisor = alloc_scalar(&mut col);
    let rs2_is_zero = alloc_scalar(&mut col);
    let rs2_nonzero = alloc_scalar(&mut col);
    let is_divu_or_remu = alloc_scalar(&mut col);
    let divu_by_zero = alloc_scalar(&mut col);
    let is_div_or_rem = alloc_scalar(&mut col);
    let div_nonzero = alloc_scalar(&mut col);
    let rem_nonzero = alloc_scalar(&mut col);
    let div_by_zero = alloc_scalar(&mut col);
    let rem_by_zero = alloc_scalar(&mut col);
    let div_sign = alloc_scalar(&mut col);
    let div_rem_check = alloc_scalar(&mut col);
    let div_rem_check_signed = alloc_scalar(&mut col);
    let ecall_a0_bits_start = alloc_array(&mut col, 32);
    let ecall_cycle_prefix_start = alloc_array(&mut col, 31);
    let ecall_is_cycle = alloc_scalar(&mut col);
    let ecall_print_prefix_start = alloc_array(&mut col, 31);
    let ecall_is_print = alloc_scalar(&mut col);
    let ecall_halts = alloc_scalar(&mut col);
    let halt_effective = alloc_scalar(&mut col);

    let cpu_cols_used = col;

    let (mem_ids, twist_ell_addrs) = derive_mem_ids_and_ell_addrs(mem_layouts)?;
    let (table_ids, shout_ell_addrs) = derive_shout_ids_and_ell_addrs(shout_table_ids)?;

    let twist_ell_addrs_and_lanes: Vec<(usize, usize)> = mem_ids
        .iter()
        .zip(twist_ell_addrs.iter())
        .map(|(mem_id, ell_addr)| {
            let lanes = mem_layouts
                .get(mem_id)
                .map(|l| l.lanes.max(1))
                .unwrap_or(1);
            (*ell_addr, lanes)
        })
        .collect();
    let bus = crate::cpu::bus_layout::build_bus_layout_for_instances_with_twist_lanes(
        m,
        m_in,
        chunk_size,
        shout_ell_addrs,
        twist_ell_addrs_and_lanes,
    )?;
    if cpu_cols_used > bus.bus_base {
        return Err(format!(
            "RV32 B1 layout: CPU columns end at {cpu_cols_used}, but bus_base={} (need more padding columns before bus tail)",
            bus.bus_base
        ));
    }

    // Determine which twist instance index corresponds to RAM/PROG in the sorted mem_ids order.
    let ram_id = RAM_ID.0;
    let prog_id = PROG_ID.0;
    let reg_id = REG_ID.0;
    let ram_twist_idx = mem_ids
        .iter()
        .position(|&id| id == ram_id)
        .ok_or_else(|| format!("mem_layouts missing RAM_ID={ram_id}"))?;
    let prog_twist_idx = mem_ids
        .iter()
        .position(|&id| id == prog_id)
        .ok_or_else(|| format!("mem_layouts missing PROG_ID={prog_id}"))?;
    let reg_twist_idx = mem_ids
        .iter()
        .position(|&id| id == reg_id)
        .ok_or_else(|| format!("mem_layouts missing REG_ID={reg_id}"))?;

    Ok(Rv32B1Layout {
        m_in,
        m,
        chunk_size,
        const_one,
        pc0,
        pc_final,
        halted_in,
        halted_out,
        is_active,
        zero,
        pc_in,
        pc_out,
        instr_word,
        instr_bits_start,
        opcode,
        funct3,
        funct7,
        rd_field,
        rs1_field,
        rs2_field,
        imm12_raw,
        imm_i,
        imm_s,
        imm_u,
        imm_b_raw,
        imm_b,
        imm_j_raw,
        imm_j,
        is_add,
        is_sub,
        is_sll,
        is_slt,
        is_sltu,
        is_xor,
        is_srl,
        is_sra,
        is_or,
        is_and,
        is_mul,
        is_mulh,
        is_mulhu,
        is_mulhsu,
        is_div,
        is_divu,
        is_rem,
        is_remu,
        is_addi,
        is_slti,
        is_sltiu,
        is_xori,
        is_ori,
        is_andi,
        is_slli,
        is_srli,
        is_srai,
        is_lb,
        is_lbu,
        is_lh,
        is_lhu,
        is_lw,
        is_sb,
        is_sh,
        is_sw,
        is_amoswap_w,
        is_amoadd_w,
        is_amoxor_w,
        is_amoor_w,
        is_amoand_w,
        is_lui,
        is_auipc,
        is_beq,
        is_bne,
        is_blt,
        is_bge,
        is_bltu,
        is_bgeu,
        is_jal,
        is_jalr,
        is_fence,
        is_halt,
        br_taken,
        br_not_taken,
        rs1_val,
        rs2_val,
        alu_out,
        mem_rv,
        mem_rv_bits_start,
        eff_addr,
        ram_has_read,
        ram_has_write,
        ram_wv,
        rd_write_val,
        rd_write_bits_start,
        add_has_lookup,
        and_has_lookup,
        xor_has_lookup,
        or_has_lookup,
        sll_has_lookup,
        srl_has_lookup,
        sra_has_lookup,
        slt_has_lookup,
        sltu_has_lookup,
        lookup_key,
        add_a0b0,
        mul_lo,
        mul_hi,
        mul_lo_bits_start,
        mul_hi_bits_start,
        mul_hi_prefix_start,
        mul_carry,
        mul_carry_bits_start,
        rs1_bits_start,
        rs2_bits_start,
        rs2_zero_prefix_start,
        rs1_abs,
        rs2_abs,
        rs1_rs2_sign_and,
        rs1_sign_rs2_val,
        rs2_sign_rs1_val,
        div_quot,
        div_rem,
        div_quot_signed,
        div_rem_signed,
        div_quot_carry,
        div_rem_carry,
        div_prod,
        div_divisor,
        rs2_is_zero,
        rs2_nonzero,
        is_divu_or_remu,
        divu_by_zero,
        is_div_or_rem,
        div_nonzero,
        rem_nonzero,
        div_by_zero,
        rem_by_zero,
        div_sign,
        div_rem_check,
        div_rem_check_signed,
        ecall_a0_bits_start,
        ecall_cycle_prefix_start,
        ecall_is_cycle,
        ecall_print_prefix_start,
        ecall_is_print,
        ecall_halts,
        halt_effective,
        reg_has_write,
        reg_rs2_addr,
        rd_is_zero_01,
        rd_is_zero_012,
        rd_is_zero_0123,
        rd_is_zero,
        bus,
        mem_ids,
        table_ids,
        ram_twist_idx,
        prog_twist_idx,
        reg_twist_idx,
    })
}
