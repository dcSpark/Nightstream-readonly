pub mod joint_lane;
pub mod manifest;
pub mod me_adapter;
pub mod reduction;

/// Decomposition base for Stage-8 time-column commitments/openings.
///
/// We intentionally use a base with representability high enough for full Goldilocks
/// field values while keeping balanced digits small.
pub const STAGE8_TIME_DECOMP_BASE: u32 = 3;
