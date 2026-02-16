#[derive(Clone, Debug)]
pub struct Rv32TraceLayout {
    pub cols: usize,

    // Core control / fetch.
    pub one: usize,
    pub active: usize,
    pub halted: usize,
    pub cycle: usize,
    pub pc_before: usize,
    pub pc_after: usize,
    pub instr_word: usize,

    // Retained decode scalars (transitional Track A surface).
    pub opcode: usize,
    pub funct3: usize,

    // Program ROM view (PROG Twist).
    pub prog_addr: usize,
    pub prog_value: usize,

    // Regfile view (REG Twist).
    pub rs1_addr: usize,
    pub rs1_val: usize,
    pub rs2_addr: usize,
    pub rs2_val: usize,
    pub rd_has_write: usize,
    pub rd_addr: usize,
    pub rd_val: usize,

    // RAM view (RAM Twist, normalized to at most 1R + 1W per row).
    pub ram_has_read: usize,
    pub ram_has_write: usize,
    pub ram_addr: usize,
    pub ram_rv: usize,
    pub ram_wv: usize,

    // Shout view (single fixed-lane per row; output-only for now).
    pub shout_has_lookup: usize,
    pub shout_val: usize,
    pub shout_lhs: usize,
    pub shout_rhs: usize,
    pub shout_table_id: usize,

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

        let branch_taken = take();
        let branch_invert_shout = take();
        let branch_taken_imm = take();
        let branch_f3b1_op = take();
        let branch_invert_shout_prod = take();
        let jalr_drop_b0 = take();
        let jalr_drop_b1 = take();

        debug_assert_eq!(next, 64, "RV32 trace width drift after W3 cutover");

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
            branch_taken,
            branch_invert_shout,
            branch_taken_imm,
            branch_f3b1_op,
            branch_invert_shout_prod,
            jalr_drop_bit: [jalr_drop_b0, jalr_drop_b1],
        }
    }
}
