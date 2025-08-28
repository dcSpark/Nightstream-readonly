use neo_challenge::DEFAULT_STRONGSET;

#[test]
fn config_parameters_are_correct() {
    let cfg = DEFAULT_STRONGSET;
    assert_eq!(cfg.eta, 81);
    assert_eq!(cfg.d, 54);
    assert_eq!(cfg.coeff_bound, 2);
    assert_eq!(cfg.max_resamples, 16);
}

#[test]
fn expansion_bound_calculation() {
    let cfg = DEFAULT_STRONGSET;
    // φ(81) = 54, H = 2, so T = 2 * 54 * 2 = 216
    assert_eq!(cfg.phi_eta(), 54);
    assert_eq!(cfg.expansion_upper_bound(), 216);
}

// Note: Full integration tests with challenguer APIs require more complex setup
// For now we verify the configuration and mathematical properties are correct
