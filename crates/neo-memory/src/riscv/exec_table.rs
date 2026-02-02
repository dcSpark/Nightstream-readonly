use neo_vm_trace::{ShoutEvent, StepTrace, TwistEvent, TwistOpKind, VmTrace};

use crate::riscv::lookups::{decode_instruction, PROG_ID, RAM_ID, REG_ID};

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
    pub cycle: u64,
    pub pc_before: u64,
    pub pc_after: u64,
    pub instr_word: u32,
    pub fields: Rv32InstrFields,
    pub halted: bool,

    /// Decoded instruction (for semantic context; derived from `instr_word`).
    pub decoded: crate::riscv::lookups::RiscvInstruction,

    /// PROG ROM fetch (`PROG_ID`) for this step.
    pub prog_read: TwistEvent<u64, u64>,

    /// REG lane 0 read (`REG_ID`, lane=0): rs1_field → rs1_val.
    pub reg_read_lane0: Rv32RegLaneIo,

    /// REG lane 1 read (`REG_ID`, lane=1): rs2_field → rs2_val.
    pub reg_read_lane1: Rv32RegLaneIo,

    /// Optional REG lane 0 write (`REG_ID`, lane=0): rd_field → rd_write_val.
    pub reg_write_lane0: Option<Rv32RegLaneIo>,

    /// RAM twist events (`RAM_ID`) for this step.
    pub ram_events: Vec<TwistEvent<u64, u64>>,

    /// Shout events for this step.
    pub shout_events: Vec<ShoutEvent<u64>>,
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
        let shout_events = step.shout_events.clone();

        Ok(Self {
            cycle: step.cycle,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            instr_word,
            fields,
            halted: step.halted,
            decoded,
            prog_read,
            reg_read_lane0,
            reg_read_lane1,
            reg_write_lane0,
            ram_events,
            shout_events,
        })
    }
}
