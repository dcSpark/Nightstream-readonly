use neo_memory::riscv::lookups::{RiscvOpcode, RiscvShoutTables};
use neo_memory::riscv::trace::{rv32_decode_lookup_backed_row_from_instr_word, Rv32DecodeSidecarLayout};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn encode_r_type(funct7: u32, funct3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0110011
}

fn base_alu_opcode_for_funct3(funct3: u32) -> RiscvOpcode {
    match funct3 {
        0 => RiscvOpcode::Add,
        1 => RiscvOpcode::Sll,
        2 => RiscvOpcode::Slt,
        3 => RiscvOpcode::Sltu,
        4 => RiscvOpcode::Xor,
        5 => RiscvOpcode::Srl,
        6 => RiscvOpcode::Or,
        7 => RiscvOpcode::And,
        _ => unreachable!("funct3 must be in 0..8"),
    }
}

fn rv32m_opcode_for_funct3(funct3: u32) -> RiscvOpcode {
    match funct3 {
        0 => RiscvOpcode::Mul,
        1 => RiscvOpcode::Mulh,
        2 => RiscvOpcode::Mulhsu,
        3 => RiscvOpcode::Mulhu,
        4 => RiscvOpcode::Div,
        5 => RiscvOpcode::Divu,
        6 => RiscvOpcode::Rem,
        7 => RiscvOpcode::Remu,
        _ => unreachable!("funct3 must be in 0..8"),
    }
}

#[test]
fn rv32_decode_lookup_rv32m_alu_delta_matches_canonical_table_id_mapping() {
    let layout = Rv32DecodeSidecarLayout::new();
    let shout = RiscvShoutTables::new(32);

    for funct3 in 0u32..8u32 {
        // OP (R-type), funct7=1 selects RV32M.
        let instr_word = encode_r_type(
            /*funct7=*/ 0b0000001, funct3, /*rd=*/ 1, /*rs1=*/ 2, /*rs2=*/ 3,
        );
        let row = rv32_decode_lookup_backed_row_from_instr_word(&layout, instr_word, /*active=*/ true);

        let base_id = shout.opcode_to_id(base_alu_opcode_for_funct3(funct3)).0 as u64;
        let rv32m_id = shout.opcode_to_id(rv32m_opcode_for_funct3(funct3)).0 as u64;
        let expected_delta = rv32m_id - base_id;

        assert_eq!(
            row[layout.shout_table_id],
            F::from_u64(rv32m_id),
            "decode lookup shout table id mismatch for funct3={funct3}"
        );
        assert_eq!(
            row[layout.alu_reg_table_delta],
            F::from_u64(expected_delta),
            "decode lookup alu_reg_table_delta mismatch for funct3={funct3}"
        );
    }
}
