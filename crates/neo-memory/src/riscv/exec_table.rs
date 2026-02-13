use neo_vm_trace::{ShoutEvent, StepTrace, TwistEvent, TwistOpKind, VmTrace};

use crate::riscv::lookups::{
    compute_op, decode_instruction, interleave_bits, uninterleave_bits, RiscvInstruction, RiscvOpcode,
    RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rv32InstrFields {
    pub opcode: u32,
    pub rd: u8,
    pub funct3: u32,
    pub rs1: u8,
    pub rs2: u8,
    pub funct7: u32,
}

impl Rv32InstrFields {
    pub fn from_word(instr_word: u32) -> Self {
        Self {
            opcode: instr_word & 0x7f,
            rd: ((instr_word >> 7) & 0x1f) as u8,
            funct3: (instr_word >> 12) & 0x7,
            rs1: ((instr_word >> 15) & 0x1f) as u8,
            rs2: ((instr_word >> 20) & 0x1f) as u8,
            funct7: (instr_word >> 25) & 0x7f,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32RegLaneIo {
    pub addr: u64,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32ExecRow {
    /// True for real trace rows; false for padded/inactive rows.
    pub active: bool,

    pub cycle: u64,
    pub pc_before: u64,
    pub pc_after: u64,
    pub instr_word: u32,
    pub fields: Rv32InstrFields,
    pub halted: bool,

    /// Decoded instruction (for semantic context; derived from `instr_word`).
    pub decoded: Option<crate::riscv::lookups::RiscvInstruction>,

    /// PROG ROM fetch (`PROG_ID`) for this step.
    pub prog_read: Option<TwistEvent<u64, u64>>,

    /// REG lane 0 read (`REG_ID`, lane=0): rs1_field → rs1_val.
    pub reg_read_lane0: Option<Rv32RegLaneIo>,

    /// REG lane 1 read (`REG_ID`, lane=1): rs2_field → rs2_val.
    pub reg_read_lane1: Option<Rv32RegLaneIo>,

    /// Optional REG lane 0 write (`REG_ID`, lane=0): rd_field → rd_write_val.
    pub reg_write_lane0: Option<Rv32RegLaneIo>,

    /// RAM twist events (`RAM_ID`) for this step.
    pub ram_events: Vec<TwistEvent<u64, u64>>,

    /// Shout events for this step.
    pub shout_events: Vec<ShoutEvent<u64>>,
}

#[derive(Clone, Debug)]
pub struct Rv32ExecColumns {
    pub active: Vec<bool>,
    pub cycle: Vec<u64>,
    pub pc_before: Vec<u64>,
    pub pc_after: Vec<u64>,
    pub instr_word: Vec<u32>,
    pub opcode: Vec<u32>,
    pub rd: Vec<u8>,
    pub funct3: Vec<u32>,
    pub rs1: Vec<u8>,
    pub rs2: Vec<u8>,
    pub funct7: Vec<u32>,
    pub halted: Vec<bool>,
    pub prog_addr: Vec<u64>,
    pub prog_value: Vec<u64>,
    pub rs1_addr: Vec<u64>,
    pub rs1_val: Vec<u64>,
    pub rs2_addr: Vec<u64>,
    pub rs2_val: Vec<u64>,
    pub rd_has_write: Vec<bool>,
    pub rd_addr: Vec<u64>,
    pub rd_val: Vec<u64>,
}

impl Rv32ExecColumns {
    pub fn len(&self) -> usize {
        self.cycle.len()
    }
}

#[derive(Clone, Debug)]
pub struct Rv32ExecTable {
    pub rows: Vec<Rv32ExecRow>,
}

impl Rv32ExecTable {
    pub fn from_trace(trace: &VmTrace<u64, u64>) -> Result<Self, String> {
        let mut rows = Vec::with_capacity(trace.steps.len());
        for step in &trace.steps {
            rows.push(Rv32ExecRow::from_step(step)?);
        }
        Ok(Self { rows })
    }

    pub fn from_trace_padded(trace: &VmTrace<u64, u64>, padded_len: usize) -> Result<Self, String> {
        if padded_len < trace.steps.len() {
            return Err(format!(
                "padded_len must be >= trace length (padded_len={} trace_len={})",
                padded_len,
                trace.steps.len()
            ));
        }

        let mut rows = Vec::with_capacity(padded_len);
        for step in &trace.steps {
            rows.push(Rv32ExecRow::from_step(step)?);
        }
        if rows.is_empty() {
            if padded_len == 0 {
                return Ok(Self { rows });
            }
            return Err("cannot pad empty trace without an initial pc".into());
        }

        let last = rows.last().expect("rows non-empty");
        let mut cycle = last.cycle;
        let pad_pc = last.pc_after;
        let pad_halted = last.halted;

        while rows.len() < padded_len {
            cycle = cycle
                .checked_add(1)
                .ok_or_else(|| "cycle overflow while padding".to_string())?;
            rows.push(Rv32ExecRow::inactive(cycle, pad_pc, pad_halted));
        }

        Ok(Self { rows })
    }

    pub fn from_trace_padded_pow2(trace: &VmTrace<u64, u64>, min_len: usize) -> Result<Self, String> {
        let steps = trace.steps.len();
        let target = steps.max(min_len).next_power_of_two();
        Self::from_trace_padded(trace, target)
    }

    pub fn validate_pc_chain(&self) -> Result<(), String> {
        for w in self.rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if a.pc_after != b.pc_before {
                return Err(format!(
                    "pc chain mismatch: cycle {} pc_after={:#x} != cycle {} pc_before={:#x}",
                    a.cycle, a.pc_after, b.cycle, b.pc_before
                ));
            }
        }
        Ok(())
    }

    /// Validate that cycles are consecutive (`cycle[t+1] = cycle[t] + 1`).
    pub fn validate_cycle_chain(&self) -> Result<(), String> {
        for w in self.rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if b.cycle != a.cycle + 1 {
                return Err(format!(
                    "cycle chain mismatch: cycle {} then {} (expected {})",
                    a.cycle,
                    b.cycle,
                    a.cycle + 1
                ));
            }
        }
        Ok(())
    }

    /// Validate that inactive rows contain no events and no decoded instruction.
    pub fn validate_inactive_rows_are_empty(&self) -> Result<(), String> {
        for r in &self.rows {
            if r.active {
                continue;
            }
            if r.decoded.is_some()
                || r.prog_read.is_some()
                || r.reg_read_lane0.is_some()
                || r.reg_read_lane1.is_some()
                || r.reg_write_lane0.is_some()
                || !r.ram_events.is_empty()
                || !r.shout_events.is_empty()
            {
                return Err(format!("inactive row has events/decoded at cycle {}", r.cycle));
            }
        }
        Ok(())
    }

    /// Validate that once `halted` becomes true, it stays true and the PC stops changing.
    pub fn validate_halted_tail(&self) -> Result<(), String> {
        let mut saw_halt = false;
        let mut halt_pc: Option<u64> = None;
        for r in &self.rows {
            if !saw_halt {
                if r.halted {
                    saw_halt = true;
                    // In our trace semantics, the HALT row itself can advance the PC (default +4),
                    // but after that the machine is halted and PC should stop changing.
                    halt_pc = Some(r.pc_after);
                }
                continue;
            }

            if !r.halted {
                return Err(format!(
                    "halted tail violated: halted dropped to false at cycle {} (pc_before={:#x})",
                    r.cycle, r.pc_before
                ));
            }

            let pc0 = halt_pc.expect("halt_pc set");
            if r.pc_before != pc0 || r.pc_after != pc0 {
                return Err(format!(
                    "halted tail violated: pc changed after halt at cycle {} (pc_before={:#x} pc_after={:#x}, expected {:#x})",
                    r.cycle, r.pc_before, r.pc_after, pc0
                ));
            }
        }
        Ok(())
    }

    /// Validate REG lane semantics by replaying the register file from an initial state.
    ///
    /// - `init_regs` maps `reg_idx (0..31)` → value (u32 stored in u64).
    /// - Unspecified registers default to 0.
    /// - Reads happen before the optional lane0 write in each cycle.
    pub fn validate_regfile_semantics(&self, init_regs: &HashMap<u64, u64>) -> Result<(), String> {
        let mut regs = [0u64; 32];
        for (&addr, &value) in init_regs {
            if addr >= 32 {
                return Err(format!("reg init addr out of range: addr={addr}"));
            }
            if addr == 0 && value != 0 {
                return Err("reg init must keep x0 == 0".into());
            }
            regs[addr as usize] = value;
        }

        for r in &self.rows {
            if !r.active {
                continue;
            }

            let Some(rs1) = &r.reg_read_lane0 else {
                return Err(format!("missing REG lane0 read at cycle {}", r.cycle));
            };
            let Some(rs2) = &r.reg_read_lane1 else {
                return Err(format!("missing REG lane1 read at cycle {}", r.cycle));
            };
            if rs1.addr >= 32 || rs2.addr >= 32 {
                return Err(format!(
                    "REG read addr out of range at cycle {}: lane0={} lane1={}",
                    r.cycle, rs1.addr, rs2.addr
                ));
            }

            let exp_rs1 = regs[rs1.addr as usize];
            let exp_rs2 = regs[rs2.addr as usize];
            if rs1.value != exp_rs1 {
                return Err(format!(
                    "REG lane0 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs1.addr, rs1.value, exp_rs1
                ));
            }
            if rs2.value != exp_rs2 {
                return Err(format!(
                    "REG lane1 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs2.addr, rs2.value, exp_rs2
                ));
            }

            if let Some(w) = &r.reg_write_lane0 {
                if w.addr >= 32 {
                    return Err(format!(
                        "REG write addr out of range at cycle {}: addr={}",
                        r.cycle, w.addr
                    ));
                }
                if w.addr == 0 {
                    return Err(format!(
                        "unexpected x0 write at cycle {} pc={:#x}",
                        r.cycle, r.pc_before
                    ));
                }
                regs[w.addr as usize] = w.value;
            }

            // x0 is always 0.
            regs[0] = 0;
        }

        Ok(())
    }

    /// Validate RAM twist semantics by replaying the RAM state from an initial state.
    ///
    /// - `init_ram` maps `byte_addr` → word value (u32 stored in u64) under the RV32 B1 convention.
    /// - Unspecified addresses default to 0.
    /// - Multiple RAM events in a cycle are applied in trace order (e.g. SB/SH read-modify-write).
    pub fn validate_ram_semantics(&self, init_ram: &HashMap<u64, u64>) -> Result<(), String> {
        let mut mem: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_ram {
            if value == 0 {
                continue;
            }
            mem.insert(addr, value);
        }

        for r in &self.rows {
            if !r.active {
                continue;
            }

            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        let exp = mem.get(&e.addr).copied().unwrap_or(0);
                        if e.value != exp {
                            return Err(format!(
                                "RAM read value mismatch at cycle {} pc={:#x}: addr={:#x} got={:#x} expected={:#x}",
                                r.cycle, r.pc_before, e.addr, e.value, exp
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        if e.value == 0 {
                            mem.remove(&e.addr);
                        } else {
                            mem.insert(e.addr, e.value);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn to_columns(&self) -> Rv32ExecColumns {
        let n = self.rows.len();

        let mut out = Rv32ExecColumns {
            active: Vec::with_capacity(n),
            cycle: Vec::with_capacity(n),
            pc_before: Vec::with_capacity(n),
            pc_after: Vec::with_capacity(n),
            instr_word: Vec::with_capacity(n),
            opcode: Vec::with_capacity(n),
            rd: Vec::with_capacity(n),
            funct3: Vec::with_capacity(n),
            rs1: Vec::with_capacity(n),
            rs2: Vec::with_capacity(n),
            funct7: Vec::with_capacity(n),
            halted: Vec::with_capacity(n),
            prog_addr: Vec::with_capacity(n),
            prog_value: Vec::with_capacity(n),
            rs1_addr: Vec::with_capacity(n),
            rs1_val: Vec::with_capacity(n),
            rs2_addr: Vec::with_capacity(n),
            rs2_val: Vec::with_capacity(n),
            rd_has_write: Vec::with_capacity(n),
            rd_addr: Vec::with_capacity(n),
            rd_val: Vec::with_capacity(n),
        };

        for r in &self.rows {
            out.active.push(r.active);
            out.cycle.push(r.cycle);
            out.pc_before.push(r.pc_before);
            out.pc_after.push(r.pc_after);
            out.instr_word.push(r.instr_word);
            out.opcode.push(r.fields.opcode);
            out.rd.push(r.fields.rd);
            out.funct3.push(r.fields.funct3);
            out.rs1.push(r.fields.rs1);
            out.rs2.push(r.fields.rs2);
            out.funct7.push(r.fields.funct7);
            out.halted.push(r.halted);

            match &r.prog_read {
                Some(e) => {
                    out.prog_addr.push(e.addr);
                    out.prog_value.push(e.value);
                }
                None => {
                    out.prog_addr.push(0);
                    out.prog_value.push(0);
                }
            }

            match &r.reg_read_lane0 {
                Some(io) => {
                    out.rs1_addr.push(io.addr);
                    out.rs1_val.push(io.value);
                }
                None => {
                    out.rs1_addr.push(0);
                    out.rs1_val.push(0);
                }
            }

            match &r.reg_read_lane1 {
                Some(io) => {
                    out.rs2_addr.push(io.addr);
                    out.rs2_val.push(io.value);
                }
                None => {
                    out.rs2_addr.push(0);
                    out.rs2_val.push(0);
                }
            }

            match &r.reg_write_lane0 {
                Some(io) => {
                    out.rd_has_write.push(true);
                    out.rd_addr.push(io.addr);
                    out.rd_val.push(io.value);
                }
                None => {
                    out.rd_has_write.push(false);
                    out.rd_addr.push(0);
                    out.rd_val.push(0);
                }
            }
        }

        out
    }
}

impl Rv32ExecRow {
    pub fn from_step(step: &StepTrace<u64, u64>) -> Result<Self, String> {
        let instr_word = step.opcode;
        let fields = Rv32InstrFields::from_word(instr_word);
        let decoded = decode_instruction(instr_word).map_err(|e| {
            format!(
                "decode_instruction failed at cycle {} pc={:#x} word={:#x}: {e}",
                step.cycle, step.pc_before, instr_word
            )
        })?;

        // PROG fetch
        let prog_read = {
            let mut reads = step
                .twist_events
                .iter()
                .filter(|e| e.twist_id == PROG_ID && matches!(e.kind, TwistOpKind::Read))
                .cloned();
            let first = reads.next().ok_or_else(|| {
                format!(
                    "missing PROG_ID read event at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                )
            })?;
            if reads.next().is_some() {
                return Err(format!(
                    "expected exactly 1 PROG_ID read event at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                ));
            }
            first
        };
        if prog_read.addr != step.pc_before {
            return Err(format!(
                "PROG_ID read addr mismatch at cycle {}: got={:#x} expected pc_before={:#x}",
                step.cycle, prog_read.addr, step.pc_before
            ));
        }
        if prog_read.value != instr_word as u64 {
            return Err(format!(
                "PROG_ID read value mismatch at cycle {} pc={:#x}: got={:#x} expected instr_word={:#x}",
                step.cycle, step.pc_before, prog_read.value, instr_word
            ));
        }
        if prog_read.lane.is_some() {
            return Err(format!(
                "unexpected PROG_ID lane hint at cycle {} pc={:#x}: lane={:?}",
                step.cycle, step.pc_before, prog_read.lane
            ));
        }

        // REG reads (lane 0 and lane 1)
        let mut reg_read_lane0: Option<Rv32RegLaneIo> = None;
        let mut reg_read_lane1: Option<Rv32RegLaneIo> = None;
        let mut reg_write_lane0: Option<Rv32RegLaneIo> = None;
        for e in step.twist_events.iter().filter(|e| e.twist_id == REG_ID) {
            match e.kind {
                TwistOpKind::Read => match e.lane {
                    Some(0) => {
                        if reg_read_lane0.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 0 read at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_read_lane0 = Some(Rv32RegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    Some(1) => {
                        if reg_read_lane1.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 1 read at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_read_lane1 = Some(Rv32RegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    other => {
                        return Err(format!(
                            "unexpected REG_ID read lane {:?} at cycle {} pc={:#x}",
                            other, step.cycle, step.pc_before
                        ));
                    }
                },
                TwistOpKind::Write => match e.lane {
                    Some(0) => {
                        if reg_write_lane0.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 0 write at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_write_lane0 = Some(Rv32RegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    other => {
                        return Err(format!(
                            "unexpected REG_ID write lane {:?} at cycle {} pc={:#x}",
                            other, step.cycle, step.pc_before
                        ));
                    }
                },
            }
        }
        let reg_read_lane0 = reg_read_lane0.ok_or_else(|| {
            format!(
                "missing REG_ID lane 0 read at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            )
        })?;
        let reg_read_lane1 = reg_read_lane1.ok_or_else(|| {
            format!(
                "missing REG_ID lane 1 read at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            )
        })?;
        if let Some(w) = &reg_write_lane0 {
            if fields.rd == 0 {
                return Err(format!(
                    "unexpected REG_ID lane 0 write to x0 at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                ));
            }
            if w.addr != fields.rd as u64 {
                return Err(format!(
                    "REG lane0 write addr mismatch at cycle {} pc={:#x}: got={} expected rd_field={}",
                    step.cycle, step.pc_before, w.addr, fields.rd
                ));
            }
        }

        // Light sanity check: make sure the trace's lane policy matches Rv32 B1's convention.
        //
        // - lane0 reads rs1_field always
        // - lane1 reads rs2_field
        let rs2_expected = fields.rs2 as u64;
        if reg_read_lane0.addr != fields.rs1 as u64 {
            return Err(format!(
                "REG lane0 read addr mismatch at cycle {} pc={:#x}: got={} expected rs1_field={}",
                step.cycle, step.pc_before, reg_read_lane0.addr, fields.rs1
            ));
        }
        if reg_read_lane1.addr != rs2_expected {
            return Err(format!(
                "REG lane1 read addr mismatch at cycle {} pc={:#x}: got={} expected={}",
                step.cycle, step.pc_before, reg_read_lane1.addr, rs2_expected
            ));
        }

        // RAM events
        let ram_events: Vec<TwistEvent<u64, u64>> = step
            .twist_events
            .iter()
            .filter(|e| e.twist_id == RAM_ID)
            .cloned()
            .collect();

        // Shout events
        let mut shout_events = step.shout_events.clone();
        if shout_events.is_empty() {
            // Backfill RV32M shout events for trace/event-table consumers.
            //
            // Some trace builders currently omit explicit Shout events for RV32M rows even when
            // the operation is semantically Shout-backed. Reconstruct the canonical event from the
            // decoded op and the architectural operands.
            if let RiscvInstruction::RAlu { op, .. } = &decoded {
                let is_rv32m = matches!(
                    op,
                    RiscvOpcode::Mul
                        | RiscvOpcode::Mulh
                        | RiscvOpcode::Mulhu
                        | RiscvOpcode::Mulhsu
                        | RiscvOpcode::Div
                        | RiscvOpcode::Divu
                        | RiscvOpcode::Rem
                        | RiscvOpcode::Remu
                );
                if is_rv32m {
                    let rs1_val = reg_read_lane0.value;
                    let rs2_val = reg_read_lane1.value;
                    let shout_id = RiscvShoutTables::new(/*xlen=*/ 32).opcode_to_id(*op);
                    let key = interleave_bits(rs1_val, rs2_val) as u64;
                    let value = compute_op(*op, rs1_val, rs2_val, /*xlen=*/ 32);
                    shout_events.push(ShoutEvent {
                        shout_id,
                        key,
                        value,
                    });
                }
            }
        }

        Ok(Self {
            active: true,
            cycle: step.cycle,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            instr_word,
            fields,
            halted: step.halted,
            decoded: Some(decoded),
            prog_read: Some(prog_read),
            reg_read_lane0: Some(reg_read_lane0),
            reg_read_lane1: Some(reg_read_lane1),
            reg_write_lane0,
            ram_events,
            shout_events,
        })
    }

    pub fn inactive(cycle: u64, pc: u64, halted: bool) -> Self {
        Self {
            active: false,
            cycle,
            pc_before: pc,
            pc_after: pc,
            instr_word: 0,
            fields: Rv32InstrFields::from_word(0),
            halted,
            decoded: None,
            prog_read: None,
            reg_read_lane0: None,
            reg_read_lane1: None,
            reg_write_lane0: None,
            ram_events: Vec::new(),
            shout_events: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32MEventRow {
    pub cycle: u64,
    pub pc: u64,
    pub opcode: RiscvOpcode,
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
    pub rs1_val: u64,
    pub rs2_val: u64,
    pub rd_write_val: Option<u64>,
    pub expected_rd_val: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32MEventTable {
    pub rows: Vec<Rv32MEventRow>,
}

impl Rv32MEventTable {
    pub fn from_exec_table(exec: &Rv32ExecTable) -> Result<Self, String> {
        let mut rows = Vec::new();

        for r in &exec.rows {
            if !r.active {
                continue;
            }
            let Some(decoded) = &r.decoded else {
                continue;
            };
            let (op, rd, rs1, rs2) = match decoded {
                RiscvInstruction::RAlu { op, rd, rs1, rs2 } => (*op, *rd, *rs1, *rs2),
                _ => continue,
            };

            let is_rv32m = matches!(
                op,
                RiscvOpcode::Mul
                    | RiscvOpcode::Mulh
                    | RiscvOpcode::Mulhu
                    | RiscvOpcode::Mulhsu
                    | RiscvOpcode::Div
                    | RiscvOpcode::Divu
                    | RiscvOpcode::Rem
                    | RiscvOpcode::Remu
            );
            if !is_rv32m {
                continue;
            }

            let rs1_val = r
                .reg_read_lane0
                .as_ref()
                .ok_or_else(|| format!("missing REG lane0 read on RV32M row at cycle {}", r.cycle))?
                .value;
            let rs2_val = r
                .reg_read_lane1
                .as_ref()
                .ok_or_else(|| format!("missing REG lane1 read on RV32M row at cycle {}", r.cycle))?
                .value;
            let expected = compute_op(op, rs1_val, rs2_val, /*xlen=*/ 32);
            let rd_write_val = r.reg_write_lane0.as_ref().map(|w| w.value);

            // The trace should not write to x0; keep the event row but require no write event.
            if rd == 0 && rd_write_val.is_some() {
                return Err(format!(
                    "unexpected x0 write event on RV32M row at cycle {} pc={:#x}",
                    r.cycle, r.pc_before
                ));
            }
            if rd != 0 && rd_write_val.is_none() {
                return Err(format!(
                    "missing rd write event on RV32M row at cycle {} pc={:#x} (rd={rd})",
                    r.cycle, r.pc_before
                ));
            }

            rows.push(Rv32MEventRow {
                cycle: r.cycle,
                pc: r.pc_before,
                opcode: op,
                rs1,
                rs2,
                rd,
                rs1_val,
                rs2_val,
                rd_write_val,
                expected_rd_val: expected,
            });
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Debug)]
pub struct Rv32ShoutEventRow {
    /// Row index within the padded exec table (0..t).
    pub row_idx: usize,
    pub cycle: u64,
    pub pc: u64,
    pub shout_id: u32,
    pub opcode: Option<RiscvOpcode>,
    /// Canonicalized key: for shift ops, `rhs` is masked to 5 bits.
    pub key: u64,
    pub lhs: u64,
    pub rhs: u64,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32ShoutEventTable {
    pub rows: Vec<Rv32ShoutEventRow>,
}

impl Rv32ShoutEventTable {
    pub fn from_exec_table(exec: &Rv32ExecTable) -> Result<Self, String> {
        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 32);
        let mut rows = Vec::new();

        for (row_idx, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            for ev in r.shout_events.iter() {
                let opcode = shout_tables.id_to_opcode(ev.shout_id);
                let (lhs, rhs_raw) = uninterleave_bits(ev.key as u128);
                let rhs = if matches!(opcode, Some(RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra)) {
                    rhs_raw & 0x1F
                } else {
                    rhs_raw
                };
                let key = if rhs != rhs_raw {
                    interleave_bits(lhs, rhs) as u64
                } else {
                    ev.key
                };

                rows.push(Rv32ShoutEventRow {
                    row_idx,
                    cycle: r.cycle,
                    pc: r.pc_before,
                    shout_id: ev.shout_id.0,
                    opcode,
                    key,
                    lhs,
                    rhs,
                    value: ev.value,
                });
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rv32RegEventKind {
    ReadLane0,
    ReadLane1,
    WriteLane0,
}

#[derive(Clone, Debug)]
pub struct Rv32RegEventRow {
    pub cycle: u64,
    pub pc: u64,
    pub kind: Rv32RegEventKind,
    pub addr: u8,
    pub prev_val: u64,
    pub next_val: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32RegEventTable {
    pub rows: Vec<Rv32RegEventRow>,
}

impl Rv32RegEventTable {
    pub fn from_exec_table(exec: &Rv32ExecTable, init_regs: &HashMap<u64, u64>) -> Result<Self, String> {
        let mut regs = [0u64; 32];
        for (&addr, &value) in init_regs {
            if addr >= 32 {
                return Err(format!("reg init addr out of range: addr={addr}"));
            }
            if addr == 0 && value != 0 {
                return Err("reg init must keep x0 == 0".into());
            }
            regs[addr as usize] = value;
        }

        let mut rows: Vec<Rv32RegEventRow> = Vec::new();
        for r in &exec.rows {
            if !r.active {
                continue;
            }

            let Some(rs1) = &r.reg_read_lane0 else {
                return Err(format!("missing REG lane0 read at cycle {}", r.cycle));
            };
            let Some(rs2) = &r.reg_read_lane1 else {
                return Err(format!("missing REG lane1 read at cycle {}", r.cycle));
            };
            if rs1.addr >= 32 || rs2.addr >= 32 {
                return Err(format!(
                    "REG read addr out of range at cycle {}: lane0={} lane1={}",
                    r.cycle, rs1.addr, rs2.addr
                ));
            }

            // Reads happen before the optional write.
            let rs1_prev = regs[rs1.addr as usize];
            let rs2_prev = regs[rs2.addr as usize];
            if rs1.value != rs1_prev {
                return Err(format!(
                    "REG lane0 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs1.addr, rs1.value, rs1_prev
                ));
            }
            if rs2.value != rs2_prev {
                return Err(format!(
                    "REG lane1 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs2.addr, rs2.value, rs2_prev
                ));
            }

            rows.push(Rv32RegEventRow {
                cycle: r.cycle,
                pc: r.pc_before,
                kind: Rv32RegEventKind::ReadLane0,
                addr: rs1.addr as u8,
                prev_val: rs1_prev,
                next_val: rs1_prev,
            });
            rows.push(Rv32RegEventRow {
                cycle: r.cycle,
                pc: r.pc_before,
                kind: Rv32RegEventKind::ReadLane1,
                addr: rs2.addr as u8,
                prev_val: rs2_prev,
                next_val: rs2_prev,
            });

            if let Some(w) = &r.reg_write_lane0 {
                if w.addr >= 32 {
                    return Err(format!(
                        "REG write addr out of range at cycle {}: addr={}",
                        r.cycle, w.addr
                    ));
                }
                if w.addr == 0 {
                    return Err(format!(
                        "unexpected x0 write at cycle {} pc={:#x}",
                        r.cycle, r.pc_before
                    ));
                }

                let prev = regs[w.addr as usize];
                let next = w.value;
                regs[w.addr as usize] = next;
                regs[0] = 0;

                rows.push(Rv32RegEventRow {
                    cycle: r.cycle,
                    pc: r.pc_before,
                    kind: Rv32RegEventKind::WriteLane0,
                    addr: w.addr as u8,
                    prev_val: prev,
                    next_val: next,
                });
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rv32RamEventKind {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub struct Rv32RamEventRow {
    pub cycle: u64,
    pub pc: u64,
    pub kind: Rv32RamEventKind,
    pub addr: u64,
    pub prev_val: u64,
    pub next_val: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32RamEventTable {
    pub rows: Vec<Rv32RamEventRow>,
}

impl Rv32RamEventTable {
    pub fn from_exec_table(exec: &Rv32ExecTable, init_ram: &HashMap<u64, u64>) -> Result<Self, String> {
        let mut mem: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_ram {
            if value == 0 {
                continue;
            }
            mem.insert(addr, value);
        }

        let mut rows: Vec<Rv32RamEventRow> = Vec::new();
        for r in &exec.rows {
            if !r.active {
                continue;
            }

            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        let prev = mem.get(&e.addr).copied().unwrap_or(0);
                        let next = prev;
                        if e.value != prev {
                            return Err(format!(
                                "RAM read value mismatch at cycle {} pc={:#x}: addr={:#x} got={:#x} expected={:#x}",
                                r.cycle, r.pc_before, e.addr, e.value, prev
                            ));
                        }
                        rows.push(Rv32RamEventRow {
                            cycle: r.cycle,
                            pc: r.pc_before,
                            kind: Rv32RamEventKind::Read,
                            addr: e.addr,
                            prev_val: prev,
                            next_val: next,
                        });
                    }
                    TwistOpKind::Write => {
                        let prev = mem.get(&e.addr).copied().unwrap_or(0);
                        let next = e.value;
                        if next == 0 {
                            mem.remove(&e.addr);
                        } else {
                            mem.insert(e.addr, next);
                        }
                        rows.push(Rv32RamEventRow {
                            cycle: r.cycle,
                            pc: r.pc_before,
                            kind: Rv32RamEventKind::Write,
                            addr: e.addr,
                            prev_val: prev,
                            next_val: next,
                        });
                    }
                }
            }
        }

        Ok(Self { rows })
    }
}
