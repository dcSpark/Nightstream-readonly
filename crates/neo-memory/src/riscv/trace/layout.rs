#[derive(Clone, Debug)]
pub struct Rv32TraceLayout {
    pub cols: usize,

    // Core control / fetch
    pub one: usize,
    pub active: usize,
    pub halted: usize,
    pub cycle: usize,
    pub pc_before: usize,
    pub pc_after: usize,
    pub instr_word: usize,

    // Decoded fields (scalars)
    pub opcode: usize,
    pub funct3: usize,
    pub funct7: usize,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,

    // Opcode-class one-hot (compact decode scaffold).
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

    // Program ROM view (PROG Twist)
    pub prog_addr: usize,
    pub prog_value: usize,

    // Regfile view (REG Twist)
    pub rs1_addr: usize,
    pub rs1_val: usize,
    pub rs2_addr: usize,
    pub rs2_val: usize,
    pub rd_has_write: usize,
    pub rd_addr: usize,
    pub rd_val: usize,

    // RAM view (RAM Twist, normalized to at most 1R + 1W per row)
    pub ram_has_read: usize,
    pub ram_has_write: usize,
    pub ram_addr: usize,
    pub ram_rv: usize,
    pub ram_wv: usize,

    // Shout view (single fixed-lane per row; output-only for now)
    pub shout_has_lookup: usize,
    pub shout_val: usize,
    pub shout_lhs: usize,
    pub shout_rhs: usize,
    pub shout_table_id: usize,

    // Load/store sub-op decode helpers.
    pub is_lb: usize,
    pub is_lbu: usize,
    pub is_lh: usize,
    pub is_lhu: usize,
    pub is_lw: usize,
    pub is_sb: usize,
    pub is_sh: usize,
    pub is_sw: usize,

    // Class+write helper gates for value-binding semantics.
    pub op_alu_imm_write: usize,
    pub op_alu_reg_write: usize,
    pub is_lb_write: usize,
    pub is_lbu_write: usize,
    pub is_lh_write: usize,
    pub is_lhu_write: usize,
    pub is_lw_write: usize,

    // Funct3 decode helpers used by ALU table-id mapping.
    pub funct3_is: [usize; 8],
    pub alu_reg_table_delta: usize,
    pub alu_imm_table_delta: usize,

    // Low-bit helpers for load/store subword semantics.
    pub ram_rv_q16: usize,
    pub rs2_q16: usize,
    pub ram_rv_low_bit: [usize; 16],
    pub rs2_low_bit: [usize; 16],

    // Small rd-bit plumbing (enables sound `rd_has_write => rd != 0`).
    pub rd_bit: [usize; 5],
    pub funct3_bit: [usize; 3],
    pub rs1_bit: [usize; 5],
    pub rs2_bit: [usize; 5],
    pub funct7_bit: [usize; 7],
    pub rd_is_zero_01: usize,
    pub rd_is_zero_012: usize,
    pub rd_is_zero_0123: usize,
    pub rd_is_zero: usize,

    // Immediate helpers (signed immediates represented as RV32 u32-in-u64).
    pub imm_i: usize,
    pub imm_s: usize,
    pub imm_b: usize,
    pub imm_j: usize,

    // Branch/JALR semantic helpers.
    pub branch_taken: usize,
    pub branch_invert_shout: usize,
    pub branch_taken_imm: usize,
    pub branch_f3b1_op: usize,
    pub branch_invert_shout_prod: usize,
    pub jalr_drop_bit: [usize; 2],
}

impl Rv32TraceLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };

        let one = take();
        let active = take();
        let halted = take();
        let cycle = take();
        let pc_before = take();
        let pc_after = take();
        let instr_word = take();

        let opcode = take();
        let funct3 = take();
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

        let prog_addr = take();
        let prog_value = take();

        let rs1_addr = take();
        let rs1_val = take();
        let rs2_addr = take();
        let rs2_val = take();
        let rd_has_write = take();
        let rd_addr = take();
        let rd_val = take();

        let ram_has_read = take();
        let ram_has_write = take();
        let ram_addr = take();
        let ram_rv = take();
        let ram_wv = take();

        let shout_has_lookup = take();
        let shout_val = take();
        let shout_lhs = take();
        let shout_rhs = take();
        let shout_table_id = take();
        let is_lb = take();
        let is_lbu = take();
        let is_lh = take();
        let is_lhu = take();
        let is_lw = take();
        let is_sb = take();
        let is_sh = take();
        let is_sw = take();
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

        let rd_b0 = take();
        let rd_b1 = take();
        let rd_b2 = take();
        let rd_b3 = take();
        let rd_b4 = take();
        let funct3_b0 = take();
        let funct3_b1 = take();
        let funct3_b2 = take();
        let rs1_b0 = take();
        let rs1_b1 = take();
        let rs1_b2 = take();
        let rs1_b3 = take();
        let rs1_b4 = take();
        let rs2_b0 = take();
        let rs2_b1 = take();
        let rs2_b2 = take();
        let rs2_b3 = take();
        let rs2_b4 = take();
        let funct7_b0 = take();
        let funct7_b1 = take();
        let funct7_b2 = take();
        let funct7_b3 = take();
        let funct7_b4 = take();
        let funct7_b5 = take();
        let funct7_b6 = take();
        let rd_is_zero_01 = take();
        let rd_is_zero_012 = take();
        let rd_is_zero_0123 = take();
        let rd_is_zero = take();
        let imm_i = take();
        let imm_s = take();
        let imm_b = take();
        let imm_j = take();
        let branch_taken = take();
        let branch_invert_shout = take();
        let branch_taken_imm = take();
        let branch_f3b1_op = take();
        let branch_invert_shout_prod = take();
        let jalr_drop_b0 = take();
        let jalr_drop_b1 = take();

        Self {
            cols: next,

            one,
            active,
            halted,
            cycle,
            pc_before,
            pc_after,
            instr_word,

            opcode,
            funct3,
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

            prog_addr,
            prog_value,

            rs1_addr,
            rs1_val,
            rs2_addr,
            rs2_val,
            rd_has_write,
            rd_addr,
            rd_val,

            ram_has_read,
            ram_has_write,
            ram_addr,
            ram_rv,
            ram_wv,

            shout_has_lookup,
            shout_val,
            shout_lhs,
            shout_rhs,
            shout_table_id,
            is_lb,
            is_lbu,
            is_lh,
            is_lhu,
            is_lw,
            is_sb,
            is_sh,
            is_sw,
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

            rd_bit: [rd_b0, rd_b1, rd_b2, rd_b3, rd_b4],
            funct3_bit: [funct3_b0, funct3_b1, funct3_b2],
            rs1_bit: [rs1_b0, rs1_b1, rs1_b2, rs1_b3, rs1_b4],
            rs2_bit: [rs2_b0, rs2_b1, rs2_b2, rs2_b3, rs2_b4],
            funct7_bit: [
                funct7_b0, funct7_b1, funct7_b2, funct7_b3, funct7_b4, funct7_b5, funct7_b6,
            ],
            rd_is_zero_01,
            rd_is_zero_012,
            rd_is_zero_0123,
            rd_is_zero,
            imm_i,
            imm_s,
            imm_b,
            imm_j,
            branch_taken,
            branch_invert_shout,
            branch_taken_imm,
            branch_f3b1_op,
            branch_invert_shout_prod,
            jalr_drop_bit: [jalr_drop_b0, jalr_drop_b1],
        }
    }
}
