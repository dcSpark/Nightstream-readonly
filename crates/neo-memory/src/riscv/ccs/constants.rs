pub(super) const RV32_XLEN: usize = 32;

// Canonical RV32 Shout table IDs (must match `RiscvShoutTables::opcode_to_id`).
pub(super) const AND_TABLE_ID: u32 = 0; // `RiscvOpcode::And`
pub(super) const XOR_TABLE_ID: u32 = 1; // `RiscvOpcode::Xor`
pub(super) const OR_TABLE_ID: u32 = 2; // `RiscvOpcode::Or`
pub(super) const ADD_TABLE_ID: u32 = 3; // `RiscvOpcode::Add`
pub(super) const SUB_TABLE_ID: u32 = 4; // `RiscvOpcode::Sub`
pub(super) const SLT_TABLE_ID: u32 = 5; // `RiscvOpcode::Slt`
pub(super) const SLTU_TABLE_ID: u32 = 6; // `RiscvOpcode::Sltu`
pub(super) const SLL_TABLE_ID: u32 = 7; // `RiscvOpcode::Sll`
pub(super) const SRL_TABLE_ID: u32 = 8; // `RiscvOpcode::Srl`
pub(super) const SRA_TABLE_ID: u32 = 9; // `RiscvOpcode::Sra`
pub(super) const EQ_TABLE_ID: u32 = 10; // `RiscvOpcode::Eq`
pub(super) const NEQ_TABLE_ID: u32 = 11; // `RiscvOpcode::Neq`

