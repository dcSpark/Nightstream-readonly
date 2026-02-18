use neo_memory::riscv::lookups::{decode_instruction, encode_instruction, RiscvInstruction, RiscvOpcode};

#[test]
fn rv32_shift_imm_decode_roundtrip() {
    let slli = RiscvInstruction::IAlu {
        op: RiscvOpcode::Sll,
        rd: 1,
        rs1: 2,
        imm: 31,
    };
    let encoded = encode_instruction(&slli);
    let decoded = decode_instruction(encoded).expect("decode slli");
    match decoded {
        RiscvInstruction::IAlu { op, rd, rs1, imm } => {
            assert_eq!(op, RiscvOpcode::Sll);
            assert_eq!(rd, 1);
            assert_eq!(rs1, 2);
            assert_eq!(imm, 31);
        }
        _ => panic!("unexpected decode for SLLI"),
    }

    let srli = RiscvInstruction::IAlu {
        op: RiscvOpcode::Srl,
        rd: 3,
        rs1: 4,
        imm: 7,
    };
    let encoded = encode_instruction(&srli);
    let decoded = decode_instruction(encoded).expect("decode srli");
    match decoded {
        RiscvInstruction::IAlu { op, rd, rs1, imm } => {
            assert_eq!(op, RiscvOpcode::Srl);
            assert_eq!(rd, 3);
            assert_eq!(rs1, 4);
            assert_eq!(imm, 7);
        }
        _ => panic!("unexpected decode for SRLI"),
    }

    let srai = RiscvInstruction::IAlu {
        op: RiscvOpcode::Sra,
        rd: 5,
        rs1: 6,
        imm: 7,
    };
    let encoded = encode_instruction(&srai);
    let decoded = decode_instruction(encoded).expect("decode srai");
    match decoded {
        RiscvInstruction::IAlu { op, rd, rs1, imm } => {
            assert_eq!(op, RiscvOpcode::Sra);
            assert_eq!(rd, 5);
            assert_eq!(rs1, 6);
            assert_eq!(imm, 7);
        }
        _ => panic!("unexpected decode for SRAI"),
    }
}

#[test]
fn rv32_shift_imm_rejects_invalid_funct7() {
    let slli = RiscvInstruction::IAlu {
        op: RiscvOpcode::Sll,
        rd: 1,
        rs1: 2,
        imm: 1,
    };
    let encoded = encode_instruction(&slli);
    let invalid = encoded | (0b0100000u32 << 25);
    assert!(decode_instruction(invalid).is_err());

    let srli = RiscvInstruction::IAlu {
        op: RiscvOpcode::Srl,
        rd: 1,
        rs1: 2,
        imm: 1,
    };
    let encoded = encode_instruction(&srli);
    let invalid = (encoded & !(0x7fu32 << 25)) | (0b0010000u32 << 25);
    assert!(decode_instruction(invalid).is_err());
}

#[test]
fn rv32_system_decode_matches_jolt_behavior() {
    let ecall = 0x0000_0073u32;
    let decoded = decode_instruction(ecall).expect("decode ecall");
    match decoded {
        RiscvInstruction::Halt => {}
        _ => panic!("expected ECALL to decode to Halt"),
    }

    let ebreak = 0x0010_0073u32;
    assert!(decode_instruction(ebreak).is_err());
}

#[test]
fn rv32_fence_decode_accepts_fence_and_fence_i() {
    let fence = RiscvInstruction::Fence { pred: 0xf, succ: 0x0 };
    let encoded = encode_instruction(&fence);
    let decoded = decode_instruction(encoded).expect("decode fence");
    match decoded {
        RiscvInstruction::Fence { pred, succ } => {
            assert_eq!(pred, 0xf);
            assert_eq!(succ, 0x0);
        }
        _ => panic!("expected FENCE decode"),
    }

    let fence_i = RiscvInstruction::FenceI;
    let encoded_fence_i = encode_instruction(&fence_i);
    let decoded_fence_i = decode_instruction(encoded_fence_i).expect("decode fence.i");
    match decoded_fence_i {
        RiscvInstruction::FenceI => {}
        _ => panic!("expected FENCE.I decode"),
    }
}

#[test]
fn rv32_fence_i_decode_rejects_invalid_reserved_fields() {
    // FENCE.I requires imm[11:0]==0 and rd/rs1==x0.
    let invalid_nonzero_imm = 0x0010_100fu32;
    assert!(decode_instruction(invalid_nonzero_imm).is_err());

    // FENCE.I with rd != 0
    let invalid_nonzero_rd = encode_instruction(&RiscvInstruction::FenceI) | (1u32 << 7);
    assert!(decode_instruction(invalid_nonzero_rd).is_err());

    let fence_i = 0x0000_100fu32;
    assert!(matches!(
        decode_instruction(fence_i).expect("valid canonical fence.i"),
        RiscvInstruction::FenceI
    ));
}
