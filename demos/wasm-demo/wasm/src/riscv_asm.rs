use std::collections::HashMap;

use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};

#[derive(Debug)]
enum PendingInstr {
    Resolved(RiscvInstruction),
    Beq { rs1: u8, rs2: u8, target: String },
    Jal { rd: u8, target: String },
}

#[derive(Debug)]
struct PendingLine {
    line_no: usize,
    instr: PendingInstr,
}

fn strip_comment(line: &str) -> &str {
    let hash = line.find('#');
    let slash = line.find("//");
    match (hash, slash) {
        (Some(h), Some(s)) => &line[..h.min(s)],
        (Some(h), None) => &line[..h],
        (None, Some(s)) => &line[..s],
        (None, None) => line,
    }
}

fn parse_reg(token: &str, line_no: usize) -> Result<u8, String> {
    let t = token.trim().trim_end_matches(',').to_ascii_lowercase();
    if t.is_empty() {
        return Err(format!("line {line_no}: expected register, got empty token"));
    }
    if let Some(num) = t.strip_prefix('x') {
        let idx: u8 = num
            .parse()
            .map_err(|_| format!("line {line_no}: invalid register '{token}'"))?;
        if idx > 31 {
            return Err(format!("line {line_no}: invalid register '{token}' (x0..x31)"));
        }
        return Ok(idx);
    }

    let reg = match t.as_str() {
        "zero" => 0,
        "ra" => 1,
        "sp" => 2,
        "gp" => 3,
        "tp" => 4,
        "t0" => 5,
        "t1" => 6,
        "t2" => 7,
        "s0" | "fp" => 8,
        "s1" => 9,
        "a0" => 10,
        "a1" => 11,
        "a2" => 12,
        "a3" => 13,
        "a4" => 14,
        "a5" => 15,
        "a6" => 16,
        "a7" => 17,
        "s2" => 18,
        "s3" => 19,
        "s4" => 20,
        "s5" => 21,
        "s6" => 22,
        "s7" => 23,
        "s8" => 24,
        "s9" => 25,
        "s10" => 26,
        "s11" => 27,
        "t3" => 28,
        "t4" => 29,
        "t5" => 30,
        "t6" => 31,
        _ => {
            return Err(format!(
                "line {line_no}: unknown register '{token}' (expected x0..x31 or ABI names like a0/t0)"
            ))
        }
    };
    Ok(reg)
}

fn parse_i32(token: &str, line_no: usize) -> Result<i32, String> {
    let s0 = token.trim().trim_end_matches(',').replace('_', "");
    if s0.is_empty() {
        return Err(format!("line {line_no}: expected immediate, got empty token"));
    }
    let (neg, s) = s0.strip_prefix('-').map(|rest| (true, rest)).unwrap_or((false, s0.as_str()));
    let val: i64 = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).map_err(|_| format!("line {line_no}: invalid hex immediate '{token}'"))?
    } else {
        s.parse::<i64>()
            .map_err(|_| format!("line {line_no}: invalid immediate '{token}'"))?
    };
    let val = if neg { -val } else { val };
    if val < i32::MIN as i64 || val > i32::MAX as i64 {
        return Err(format!("line {line_no}: immediate out of range for i32: '{token}'"));
    }
    Ok(val as i32)
}

fn parse_u32_word(line: &str) -> Option<u32> {
    let s0 = line.trim().replace('_', "");
    if s0.is_empty() {
        return None;
    }
    if s0.chars().any(|c| c.is_whitespace()) {
        return None;
    }
    let (s, radix) = if let Some(hex) = s0.strip_prefix("0x").or_else(|| s0.strip_prefix("0X")) {
        (hex, 16)
    } else {
        (s0.as_str(), 10)
    };
    u32::from_str_radix(s, radix).ok()
}

fn parse_mem_operand(token: &str, line_no: usize) -> Result<(i32, u8), String> {
    let t = token.trim().trim_end_matches(',');
    let open = t
        .find('(')
        .ok_or_else(|| format!("line {line_no}: expected mem operand like '0(x1)', got '{token}'"))?;
    let close = t
        .rfind(')')
        .ok_or_else(|| format!("line {line_no}: expected mem operand like '0(x1)', got '{token}'"))?;
    if close <= open {
        return Err(format!("line {line_no}: invalid mem operand '{token}'"));
    }
    let off_str = t[..open].trim();
    let base_str = t[open + 1..close].trim();
    let off = if off_str.is_empty() { 0 } else { parse_i32(off_str, line_no)? };
    let base = parse_reg(base_str, line_no)?;
    Ok((off, base))
}

fn parse_operands(rest: &str) -> Vec<&str> {
    rest.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_line(
    line_no: usize,
    line: &str,
    labels: &mut HashMap<String, usize>,
    out: &mut Vec<PendingLine>,
) -> Result<(), String> {
    let raw = strip_comment(line).trim();
    if raw.is_empty() {
        return Ok(());
    }
    if raw.starts_with('.') {
        return Ok(());
    }

    let mut rest = raw;
    if let Some((label, after)) = raw.split_once(':') {
        let name = label.trim();
        if !name.is_empty() {
            let idx = out.len();
            labels.insert(name.to_string(), idx);
        }
        rest = after.trim();
        if rest.is_empty() {
            return Ok(());
        }
    }

    if let Some(word) = parse_u32_word(rest) {
        let decoded = neo_memory::riscv::lookups::decode_instruction(word)
            .map_err(|e| format!("line {line_no}: invalid RV32 word '{rest}': {e}"))?;
        // Ensure our round-trip matches the original encoding.
        let expected = word.to_le_bytes();
        let re_bytes = encode_program(&[decoded.clone()]);
        if re_bytes.len() != 4 || re_bytes.as_slice() != &expected {
            return Err(format!(
                "line {line_no}: unsupported raw word '{rest}' (decode/encode mismatch)"
            ));
        }
        out.push(PendingLine {
            line_no,
            instr: PendingInstr::Resolved(decoded),
        });
        return Ok(());
    }

    let mut iter = rest.split_whitespace();
    let op = iter
        .next()
        .ok_or_else(|| format!("line {line_no}: expected instruction"))?
        .to_ascii_lowercase();
    let operands_str = iter.collect::<Vec<_>>().join(" ");
    let args = parse_operands(&operands_str);

    let pending = match op.as_str() {
        "addi" => {
            if args.len() != 3 {
                return Err(format!("line {line_no}: addi expects 3 operands: addi rd, rs1, imm"));
            }
            let rd = parse_reg(args[0], line_no)?;
            let rs1 = parse_reg(args[1], line_no)?;
            let imm = parse_i32(args[2], line_no)?;
            PendingInstr::Resolved(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1,
                imm,
            })
        }
        "add" => {
            if args.len() != 3 {
                return Err(format!("line {line_no}: add expects 3 operands: add rd, rs1, rs2"));
            }
            let rd = parse_reg(args[0], line_no)?;
            let rs1 = parse_reg(args[1], line_no)?;
            let rs2 = parse_reg(args[2], line_no)?;
            PendingInstr::Resolved(RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1,
                rs2,
            })
        }
        "lw" => {
            if args.len() != 2 {
                return Err(format!("line {line_no}: lw expects 2 operands: lw rd, off(rs1)"));
            }
            let rd = parse_reg(args[0], line_no)?;
            let (imm, rs1) = parse_mem_operand(args[1], line_no)?;
            PendingInstr::Resolved(RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd,
                rs1,
                imm,
            })
        }
        "sw" => {
            if args.len() != 2 {
                return Err(format!("line {line_no}: sw expects 2 operands: sw rs2, off(rs1)"));
            }
            let rs2 = parse_reg(args[0], line_no)?;
            let (imm, rs1) = parse_mem_operand(args[1], line_no)?;
            PendingInstr::Resolved(RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1,
                rs2,
                imm,
            })
        }
        "beq" => {
            if args.len() != 3 {
                return Err(format!(
                    "line {line_no}: beq expects 3 operands: beq rs1, rs2, label|imm"
                ));
            }
            let rs1 = parse_reg(args[0], line_no)?;
            let rs2 = parse_reg(args[1], line_no)?;
            match parse_i32(args[2], line_no) {
                Ok(imm) => PendingInstr::Resolved(RiscvInstruction::Branch {
                    cond: BranchCondition::Eq,
                    rs1,
                    rs2,
                    imm,
                }),
                Err(_) => PendingInstr::Beq {
                    rs1,
                    rs2,
                    target: args[2].to_string(),
                },
            }
        }
        "jal" => {
            if args.len() == 1 {
                // pseudo: jal label  (rd=ra)
                let rd = 1u8;
                let target = args[0].to_string();
                PendingInstr::Jal { rd, target }
            } else if args.len() == 2 {
                let rd = parse_reg(args[0], line_no)?;
                match parse_i32(args[1], line_no) {
                    Ok(imm) => PendingInstr::Resolved(RiscvInstruction::Jal { rd, imm }),
                    Err(_) => PendingInstr::Jal {
                        rd,
                        target: args[1].to_string(),
                    },
                }
            } else {
                return Err(format!(
                    "line {line_no}: jal expects 1 or 2 operands: jal label  OR  jal rd, label|imm"
                ));
            }
        }
        "j" => {
            if args.len() != 1 {
                return Err(format!("line {line_no}: j expects 1 operand: j label"));
            }
            PendingInstr::Jal {
                rd: 0u8,
                target: args[0].to_string(),
            }
        }
        "li" => {
            if args.len() != 2 {
                return Err(format!("line {line_no}: li expects 2 operands: li rd, imm"));
            }
            let rd = parse_reg(args[0], line_no)?;
            let imm = parse_i32(args[1], line_no)?;
            if !(-2048..=2047).contains(&imm) {
                return Err(format!(
                    "line {line_no}: li immediate out of range for MVP (-2048..2047). Use 'lui'/'addi' or paste raw words."
                ));
            }
            PendingInstr::Resolved(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1: 0,
                imm,
            })
        }
        "mv" => {
            if args.len() != 2 {
                return Err(format!("line {line_no}: mv expects 2 operands: mv rd, rs"));
            }
            let rd = parse_reg(args[0], line_no)?;
            let rs = parse_reg(args[1], line_no)?;
            PendingInstr::Resolved(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1: rs,
                imm: 0,
            })
        }
        "ecall" | "halt" => {
            if !args.is_empty() {
                return Err(format!("line {line_no}: {op} takes no operands"));
            }
            PendingInstr::Resolved(RiscvInstruction::Halt)
        }
        "nop" => {
            if !args.is_empty() {
                return Err(format!("line {line_no}: nop takes no operands"));
            }
            PendingInstr::Resolved(RiscvInstruction::Nop)
        }
        _ => {
            return Err(format!(
                "line {line_no}: unsupported opcode '{op}' (supported: addi, add, lw, sw, beq, jal, j, li, mv, ecall, nop)"
            ))
        }
    };

    out.push(PendingLine {
        line_no,
        instr: pending,
    });
    Ok(())
}

pub fn assemble_rv32_mini_asm(text: &str) -> Result<Vec<u8>, String> {
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut pending: Vec<PendingLine> = Vec::new();

    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        parse_line(line_no, line, &mut labels, &mut pending)?;
    }

    let mut program: Vec<RiscvInstruction> = Vec::with_capacity(pending.len());
    for (i, item) in pending.into_iter().enumerate() {
        let pc_from = (i as i32) * 4;
        let resolved = match item.instr {
            PendingInstr::Resolved(instr) => instr,
            PendingInstr::Beq { rs1, rs2, target } => {
                let idx = labels
                    .get(&target)
                    .copied()
                    .ok_or_else(|| format!("line {}: unknown label '{target}'", item.line_no))?;
                let pc_to = (idx as i32) * 4;
                let imm = pc_to - pc_from;
                RiscvInstruction::Branch {
                    cond: BranchCondition::Eq,
                    rs1,
                    rs2,
                    imm,
                }
            }
            PendingInstr::Jal { rd, target } => {
                let idx = labels
                    .get(&target)
                    .copied()
                    .ok_or_else(|| format!("line {}: unknown label '{target}'", item.line_no))?;
                let pc_to = (idx as i32) * 4;
                let imm = pc_to - pc_from;
                RiscvInstruction::Jal { rd, imm }
            }
        };
        program.push(resolved);
    }

    Ok(encode_program(&program))
}
