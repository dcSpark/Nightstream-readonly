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

    // Small rd-bit plumbing (enables sound `rd_has_write => rd != 0`).
    pub rd_bit: [usize; 5],
    pub rd_is_zero_01: usize,
    pub rd_is_zero_012: usize,
    pub rd_is_zero_0123: usize,
    pub rd_is_zero: usize,
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

        let rd_b0 = take();
        let rd_b1 = take();
        let rd_b2 = take();
        let rd_b3 = take();
        let rd_b4 = take();
        let rd_is_zero_01 = take();
        let rd_is_zero_012 = take();
        let rd_is_zero_0123 = take();
        let rd_is_zero = take();

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

            rd_bit: [rd_b0, rd_b1, rd_b2, rd_b3, rd_b4],
            rd_is_zero_01,
            rd_is_zero_012,
            rd_is_zero_0123,
            rd_is_zero,
        }
    }
}

