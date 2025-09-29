//! Test harness utilities for deterministic, low-noise runs.

use neo_spartan_bridge::me_to_r1cs::clear_snark_caches;

pub fn test_setup() {
    // Deterministic randomness for reproducible tests
    std::env::set_var("NEO_DETERMINISTIC", "1");
    // Avoid accidental identity RLC unless explicitly requested
    std::env::remove_var("NEO_TEST_RLC_IDENTITY");
    // Clear cached PK/VK to prevent cross-test interference
    clear_snark_caches();
    #[allow(deprecated)]
    {
        // Safe in tests (function exposed behind test/feature gates)
        neo_spartan_bridge::clear_vk_registry();
    }
}
