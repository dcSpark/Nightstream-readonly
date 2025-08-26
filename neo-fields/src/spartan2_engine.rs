// This keeps the path neo_fields::spartan2_engine::GoldilocksEngine valid in all builds.
// In NARK/default builds it's an empty marker type.
// When you later wire Spartan2 for real, give this type a proper impl of Spartan2's `Engine`.
pub struct GoldilocksEngine;
