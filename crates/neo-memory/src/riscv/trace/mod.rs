pub mod air;
pub mod decode_lookup;
pub mod layout;
pub mod sidecar_extract;
pub mod width_sidecar;
pub mod witness;

pub use air::Rv32TraceAir;
pub use decode_lookup::{
    rv32_decode_lookup_addr_group_for_table_id, rv32_decode_lookup_backed_cols,
    rv32_decode_lookup_backed_row_from_instr_word, rv32_decode_lookup_table_id_for_col, rv32_is_decode_lookup_table_id,
    Rv32DecodeSidecarLayout, RV32_TRACE_DECODE_LOOKUP_TABLE_BASE,
};
pub use layout::Rv32TraceLayout;
pub use sidecar_extract::{
    extract_shout_lanes_over_time, extract_twist_lanes_over_time, ShoutLaneOverTime, TraceTwistLanesOverTime,
    TwistLaneOverTime,
};
pub use width_sidecar::{
    rv32_is_width_lookup_table_id, rv32_width_lookup_addr_group_for_table_id, rv32_width_lookup_backed_cols,
    rv32_width_lookup_table_id_for_col, rv32_width_sidecar_witness_from_exec_table, Rv32WidthSidecarLayout,
    Rv32WidthSidecarWitness, RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE,
};
pub use witness::Rv32TraceWitness;

/// Shared-address group id for canonical RV32 opcode Shout tables (table_id 0..=19).
///
/// These families all use the same interleaved `(lhs,rhs)` key width (`ell_addr=64`),
/// so in RV32 trace shared-bus mode they can share one addr-bit range.
pub const RV32_TRACE_OPCODE_ADDR_GROUP: u32 = 0x5256_4100;
/// Shared selector-group id for decode lookup families (table_id range at `RV32_TRACE_DECODE_LOOKUP_TABLE_BASE`).
pub const RV32_TRACE_DECODE_SELECTOR_GROUP: u32 = 0x5256_4B00;
/// Shared selector-group id for width lookup families (table_id range at `RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE`).
pub const RV32_TRACE_WIDTH_SELECTOR_GROUP: u32 = 0x5256_5B00;

#[inline]
pub fn rv32_trace_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    if table_id <= 19 {
        Some(RV32_TRACE_OPCODE_ADDR_GROUP)
    } else {
        rv32_decode_lookup_addr_group_for_table_id(table_id)
            .or_else(|| rv32_width_lookup_addr_group_for_table_id(table_id))
    }
}

/// Shape-aware address-group hint for shared-bus Shout lanes.
///
/// This guards against accidental grouping when callers use low numeric `table_id`s for
/// non-RV32 opcode tables (common in generic tests/fixtures). RV32 opcode tables (id 0..=19)
/// are grouped only when their key shape matches the canonical interleaved width.
#[inline]
pub fn rv32_trace_lookup_addr_group_for_table_shape(table_id: u32, ell_addr: usize) -> Option<u32> {
    let group = rv32_trace_lookup_addr_group_for_table_id(table_id)?;
    if table_id <= 19 && ell_addr != 64 {
        return None;
    }
    Some(group)
}

#[inline]
pub fn rv32_trace_lookup_selector_group_for_table_id(table_id: u32) -> Option<u32> {
    if rv32_is_decode_lookup_table_id(table_id) {
        Some(RV32_TRACE_DECODE_SELECTOR_GROUP)
    } else if rv32_is_width_lookup_table_id(table_id) {
        Some(RV32_TRACE_WIDTH_SELECTOR_GROUP)
    } else {
        None
    }
}
