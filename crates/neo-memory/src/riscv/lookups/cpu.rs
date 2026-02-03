use neo_vm_trace::{Shout, Twist};

use super::bits::interleave_bits;
use super::decode::decode_instruction;
use super::encode::encode_instruction;
use super::isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use super::tables::RiscvShoutTables;

/// A RISC-V CPU that can be traced using Neo's VmCpu trait.
///
/// Implements RV32I/RV64I base instruction set.
/// Based on Jolt's CPU implementation (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
pub struct RiscvCpu {
    /// Program counter.
    pub pc: u64,
    /// General-purpose registers (x0-x31, where x0 is always 0).
    pub regs: [u64; 32],
    /// Word size in bits (32 or 64).
    pub xlen: usize,
    /// Whether the CPU has halted.
    pub halted: bool,
    /// Program to execute (list of instructions).
    program: Vec<RiscvInstruction>,
    /// Base address of the program.
    program_base: u64,
}

impl RiscvCpu {
    /// Create a new CPU with the given word size.
    pub fn new(xlen: usize) -> Self {
        assert!(xlen == 32 || xlen == 64);
        Self {
            pc: 0,
            regs: [0; 32],
            xlen,
            halted: false,
            program: Vec::new(),
            program_base: 0,
        }
    }

    /// Load a program starting at the given base address.
    pub fn load_program(&mut self, base: u64, program: Vec<RiscvInstruction>) {
        self.program_base = base;
        self.program = program;
        self.pc = base;
    }

    /// Set a register value (x0 writes are ignored).
    pub fn set_reg(&mut self, reg: u8, value: u64) {
        if reg != 0 {
            self.regs[reg as usize] = self.mask_value(value);
        }
    }

    /// Get a register value.
    pub fn get_reg(&self, reg: u8) -> u64 {
        self.regs[reg as usize]
    }

    /// Mask a value to the word size.
    fn mask_value(&self, value: u64) -> u64 {
        if self.xlen == 32 {
            value as u32 as u64
        } else {
            value
        }
    }

    /// Sign-extend an immediate.
    fn sign_extend_imm(&self, imm: i32) -> u64 {
        if self.xlen == 32 {
            imm as u32 as u64
        } else {
            imm as i64 as u64
        }
    }

    /// Get the current instruction (if any).
    fn current_instruction(&self) -> Option<&RiscvInstruction> {
        let index = (self.pc - self.program_base) / 4;
        self.program.get(index as usize)
    }

    fn handle_ecall(&mut self) {
        self.halted = true;
    }

    fn write_reg<T: Twist<u64, u64>>(&mut self, twist: &mut T, reg: u8, value: u64) {
        if reg == 0 {
            return;
        }
        let masked = self.mask_value(value);
        twist.store_lane(super::REG_ID, reg as u64, masked, /*lane=*/ 0);
        self.regs[reg as usize] = masked;
    }
}

impl neo_vm_trace::VmCpu<u64, u64> for RiscvCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        self.regs.to_vec()
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<neo_vm_trace::StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        if (self.pc & 0b11) != 0 {
            return Err(format!("PC not 4-byte aligned (no compressed): pc={:#x}", self.pc));
        }

        let ram = super::RAM_ID;
        let prog = super::PROG_ID;
        let shout_tables = RiscvShoutTables::new(self.xlen);
        let add_shout_id = shout_tables.opcode_to_id(RiscvOpcode::Add);

        let instr_word = twist.load(prog, self.pc);
        let instr_word_u32 = u32::try_from(instr_word).map_err(|_| {
            format!(
                "Instruction word at PC {:#x} does not fit in 32 bits: {:#x}",
                self.pc, instr_word
            )
        })?;
        if (instr_word_u32 & 0b11) != 0b11 {
            return Err(format!(
                "Compressed instructions not supported (PC {:#x}, word {:#x})",
                self.pc, instr_word_u32
            ));
        }

        if let Some(expected) = self.current_instruction() {
            let expected_word = encode_instruction(expected);
            if expected_word != instr_word_u32 {
                return Err(format!(
                    "Program ROM mismatch at PC {:#x}: expected {:#x}, got {:#x}",
                    self.pc, expected_word, instr_word_u32
                ));
            }
        }

        let instr = decode_instruction(instr_word_u32).map_err(|e| {
            format!(
                "Failed to decode instruction at PC {:#x} (word {:#x}): {e}",
                self.pc, instr_word_u32
            )
        })?;

        // --------------------------------------------------------------------
        // Regfile-as-Twist (REG_ID): always emit two register reads per step.
        //
        // Lane assignment (RV32 B1 convention):
        // - lane 0: read rs1_field
        // - lane 1: read rs2_field
        // --------------------------------------------------------------------
        let reg = super::REG_ID;
        let rs1_field = ((instr_word_u32 >> 15) & 0x1f) as u64;
        let rs2_field = ((instr_word_u32 >> 20) & 0x1f) as u64;
        let rs2_addr = rs2_field;

        let rs1_val = self.mask_value(twist.load_lane(reg, rs1_field, /*lane=*/ 0));
        let rs2_val = self.mask_value(twist.load_lane(reg, rs2_addr, /*lane=*/ 1));

        // Keep the CPU's register snapshot mirror consistent with Twist state.
        self.regs[0] = 0;
        if rs1_field != 0 {
            self.regs[rs1_field as usize] = rs1_val;
        }
        if rs2_addr != 0 {
            self.regs[rs2_addr as usize] = rs2_val;
        }

        // Default: advance PC by 4
        let mut next_pc = self.pc.wrapping_add(4);
        let step_opcode: u32 = instr_word_u32;

        match instr {
            RiscvInstruction::RAlu { op, rd, rs1: _, rs2: _ } => {
                match op {
                    // RV32 B1 does not use Shout tables for RV32M semantics.
                    // (They are checked by the RV32M sidecar CCS; Shout is only used for the remainder-bound SLTU check.)
                    RiscvOpcode::Mul
                    | RiscvOpcode::Mulh
                    | RiscvOpcode::Mulhu
                    | RiscvOpcode::Mulhsu
                    | RiscvOpcode::Div
                    | RiscvOpcode::Divu
                    | RiscvOpcode::Rem
                    | RiscvOpcode::Remu
                        if self.xlen == 32 =>
                    {
                        let rs1_u32 = rs1_val as u32;
                        let rs2_u32 = rs2_val as u32;
                        let rs1_i32 = rs1_u32 as i32;
                        let rs2_i32 = rs2_u32 as i32;

                        match op {
                            RiscvOpcode::Mul => {
                                let result = rs1_u32.wrapping_mul(rs2_u32) as u64;
                                self.write_reg(twist, rd, result);
                            }
                            RiscvOpcode::Mulh => {
                                let product = (rs1_i32 as i64) * (rs2_i32 as i64);
                                let result = (product >> 32) as i32 as u32;
                                self.write_reg(twist, rd, result as u64);
                            }
                            RiscvOpcode::Mulhu => {
                                let product = (rs1_u32 as u64) * (rs2_u32 as u64);
                                let result = (product >> 32) as u32;
                                self.write_reg(twist, rd, result as u64);
                            }
                            RiscvOpcode::Mulhsu => {
                                let product = (rs1_i32 as i64) * (rs2_u32 as i64);
                                let result = (product >> 32) as i32 as u32;
                                self.write_reg(twist, rd, result as u64);
                            }
                            RiscvOpcode::Div | RiscvOpcode::Rem => {
                                let divisor_is_zero = rs2_u32 == 0;
                                let (quot_i32, rem_i32) = if divisor_is_zero {
                                    (-1i32, rs1_i32)
                                } else if rs1_i32 == i32::MIN && rs2_i32 == -1 {
                                    (rs1_i32, 0)
                                } else {
                                    (rs1_i32 / rs2_i32, rs1_i32 % rs2_i32)
                                };
                                let result = match op {
                                    RiscvOpcode::Div => quot_i32 as u32,
                                    RiscvOpcode::Rem => rem_i32 as u32,
                                    _ => unreachable!(),
                                };
                                self.write_reg(twist, rd, result as u64);

                                // Record a single Shout event for the remainder bound, only when divisor != 0.
                                if !divisor_is_zero {
                                    let rem_abs = (rem_i32 as i64).abs() as u64;
                                    let divisor_abs = (rs2_i32 as i64).abs() as u64;
                                    let sltu_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu);
                                    let index = interleave_bits(rem_abs, divisor_abs) as u64;
                                    let _ = shout.lookup(sltu_id, index);
                                }
                            }
                            RiscvOpcode::Divu | RiscvOpcode::Remu => {
                                let dividend = rs1_u32 as u64;
                                let divisor = rs2_u32 as u64;
                                let (quot, rem) = if divisor == 0 {
                                    (u32::MAX as u64, dividend)
                                } else {
                                    (dividend / divisor, dividend % divisor)
                                };
                                let result = match op {
                                    RiscvOpcode::Divu => quot,
                                    RiscvOpcode::Remu => rem,
                                    _ => unreachable!(),
                                };
                                self.write_reg(twist, rd, result);

                                // Record a single Shout event for the remainder bound, only when divisor != 0.
                                if divisor != 0 {
                                    let sltu_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu);
                                    let index = interleave_bits(rem, divisor) as u64;
                                    let _ = shout.lookup(sltu_id, index);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        // Use Shout for the ALU operation
                        let shout_id = shout_tables.opcode_to_id(op);
                        let index = interleave_bits(rs1_val, rs2_val) as u64;
                        let result = shout.lookup(shout_id, index);
                        self.write_reg(twist, rd, result);
                    }
                }
            }

            RiscvInstruction::IAlu { op, rd, rs1: _, imm } => {
                let imm_val = self.sign_extend_imm(imm);

                // Use Shout for the ALU operation
                let shout_id = shout_tables.opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::Load { op, rd, rs1: _, imm } => {
                let base = rs1_val;
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(base, imm_val) as u64;
                let addr = shout.lookup(add_shout_id, index);

                // Twist RAM semantics (RV32 B1 / MVP):
                // - Memory is byte-addressed, and `addr` is a byte address.
                // - Twist accesses are word-valued (XLEN bits), i.e. a `load/store` at `addr`
                //   reads/writes the little-endian word window starting at `addr`.
                // - Sub-word ops (LB/LH/LBU/LHU) read the word window at `addr` and then mask/sign-extend
                //   the low byte/halfword.
                //
                // NOTE: This matches the proof-layer convention in `riscv::ccs` and keeps the trace at
                // "≤ 1 Twist read per instruction".
                let raw_value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let value = raw_value & mask;

                // Sign-extend if needed
                let result = if op.is_sign_extend() {
                    match width {
                        1 => (value as u8) as i8 as i64 as u64,
                        2 => (value as u16) as i16 as i64 as u64,
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value,
                    }
                } else {
                    value
                };

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::Store {
                op,
                rs1: _,
                rs2: _,
                imm,
            } => {
                let base = rs1_val;
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(base, imm_val) as u64;
                let addr = shout.lookup(add_shout_id, index);
                let value = rs2_val;

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Twist RAM semantics: see the comment in the Load arm above.
                //
                // For SB/SH, implement a read-modify-write on the word window at `addr`,
                // updating only the low 8/16 bits.
                if width < (self.xlen / 8) {
                    let raw_value = twist.load(ram, addr);
                    let merged = (raw_value & !mask) | store_value;
                    twist.store(ram, addr, merged);
                } else {
                    twist.store(ram, addr, store_value);
                }
            }

            RiscvInstruction::Branch {
                cond,
                rs1: _,
                rs2: _,
                imm,
            } => {
                // Use Shout for the comparison
                let shout_id = shout_tables.opcode_to_id(cond.to_shout_opcode());
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let cmp = shout.lookup(shout_id, index);
                if cmp > 1 {
                    return Err(format!(
                        "branch compare lookup must be 0/1: cond={cond}, rs1={rs1_val:#x}, rs2={rs2_val:#x}, got={cmp}"
                    ));
                }

                let taken = match cond {
                    BranchCondition::Eq | BranchCondition::Ne | BranchCondition::Lt | BranchCondition::Ltu => cmp == 1,
                    BranchCondition::Ge | BranchCondition::Geu => cmp == 0,
                };
                if taken {
                    let imm_u = self.sign_extend_imm(imm);
                    next_pc = self.pc.wrapping_add(imm_u);
                }
            }

            RiscvInstruction::Jal { rd, imm } => {
                // rd = pc + 4 (return address)
                self.write_reg(twist, rd, self.pc.wrapping_add(4));
                // pc = pc + imm
                let imm_u = self.sign_extend_imm(imm);
                next_pc = self.pc.wrapping_add(imm_u);
            }

            RiscvInstruction::Jalr { rd, rs1: _, imm } => {
                let return_addr = self.pc.wrapping_add(4);

                // pc = (rs1 + imm) & !3 (MVP: no compressed instructions)
                // Use Shout ADD for modular semantics, then apply the JALR alignment mask.
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let sum = shout.lookup(add_shout_id, index);
                next_pc = sum & !3u64;

                // rd = return address
                self.write_reg(twist, rd, return_addr);
            }

            RiscvInstruction::Lui { rd, imm } => {
                // rd = imm << 12 (upper 20 bits)
                let value = (imm as i64 as u64) << 12;
                self.write_reg(twist, rd, value);
            }

            RiscvInstruction::Auipc { rd, imm } => {
                // rd = pc + (imm << 12) (via Shout ADD for modular RV32 semantics)
                let imm_u = self.mask_value((imm as i64 as u64) << 12);
                let index = interleave_bits(self.pc, imm_u) as u64;
                let value = shout.lookup(add_shout_id, index);
                self.write_reg(twist, rd, value);
            }

            RiscvInstruction::Halt => {
                // ECALL trap semantics: halt.
                self.handle_ecall();
            }

            RiscvInstruction::Nop => {}

            // === RV64 W-suffix Operations ===
            RiscvInstruction::RAluw { op, rd, rs1: _, rs2: _ } => {
                let shout_id = shout_tables.opcode_to_id(op);
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::IAluw { op, rd, rs1: _, imm } => {
                let imm_val = self.sign_extend_imm(imm);

                let shout_id = shout_tables.opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            // === A Extension: Atomics ===
            RiscvInstruction::LoadReserved { op, rd, rs1: _ } => {
                let addr = rs1_val;
                let value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let result = if op.is_sign_extend() {
                    match width {
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value & mask,
                    }
                } else {
                    value & mask
                };

                self.write_reg(twist, rd, result);
                // Note: In a real implementation, we'd reserve the address here
            }

            RiscvInstruction::StoreConditional { op, rd, rs1: _, rs2: _ } => {
                let addr = rs1_val;
                let value = rs2_val;

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Store the value
                twist.store(ram, addr, store_value);

                // SC returns 0 on success (assuming reservation is valid in single-threaded mode)
                self.write_reg(twist, rd, 0);
            }

            RiscvInstruction::Amo { op, rd, rs1: _, rs2: _ } => {
                let addr = rs1_val;
                let src = rs2_val;

                // Load original value
                let original = self.mask_value(twist.load(ram, addr));
                self.write_reg(twist, rd, original);

                // Compute new value based on AMO operation
                let new_val = match op {
                    RiscvMemOp::AmoswapW | RiscvMemOp::AmoswapD => src,
                    // Use Shout for modular semantics (and to emit a ShoutEvent for the prover).
                    RiscvMemOp::AmoaddW | RiscvMemOp::AmoaddD => {
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(add_shout_id, index)
                    }
                    RiscvMemOp::AmoxorW | RiscvMemOp::AmoxorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Xor);
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoandW | RiscvMemOp::AmoandD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::And);
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoorW | RiscvMemOp::AmoorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Or);
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmominW => {
                        if (original as i32) < (src as i32) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmominD => {
                        if (original as i64) < (src as i64) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxW => {
                        if (original as i32) > (src as i32) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxD => {
                        if (original as i64) > (src as i64) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmominuW | RiscvMemOp::AmominuD => {
                        if original < src {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxuW | RiscvMemOp::AmomaxuD => {
                        if original > src {
                            original
                        } else {
                            src
                        }
                    }
                    _ => src, // Fallback
                };

                // Store new value
                twist.store(ram, addr, self.mask_value(new_val));
            }

            // === System Instructions ===
            RiscvInstruction::Ecall => {
                // ECALL - environment call (syscall).
                self.handle_ecall();
            }

            RiscvInstruction::Ebreak => {
                // EBREAK - debugger breakpoint
                // For now, treat as halt
                self.halted = true;
            }

            RiscvInstruction::Fence { pred: _, succ: _ } => {
                // FENCE - memory ordering
                // No-op in single-threaded execution
            }

            RiscvInstruction::FenceI => {
                // FENCE.I - instruction fence
                // No-op in our implementation
            }
        }

        let next_pc_masked = self.mask_value(next_pc);
        if (next_pc_masked & 0b11) != 0 {
            return Err(format!(
                "control-flow target not 4-byte aligned (no compressed): next_pc={:#x}",
                next_pc_masked
            ));
        }
        self.pc = next_pc_masked;

        Ok(neo_vm_trace::StepMeta {
            pc_after: self.pc,
            opcode: step_opcode,
        })
    }
}
