use crate::fs::{
    derive_all_sumcheck_challenges, derive_alpha_shared, derive_beta_a_shared, derive_beta_m_nc, derive_beta_r_fe,
    derive_gamma_shared, FsChannel,
};
use crate::k_field::KRepr;
use crate::statement::{compute_step_bundle_digest_v2, digest32_to_u128_limbs_le};

#[derive(Clone, Copy, Debug)]
pub struct StepBundleStatementV2 {
    pub step_idx: u32,
    pub params_digest32: [u8; 32],
    pub ccs_digest32: [u8; 32],
    pub initial_acc_digest32: [u8; 32],
    pub final_acc_digest32: [u8; 32],
}

impl StepBundleStatementV2 {
    pub fn bundle_digest32(&self) -> [u8; 32] {
        compute_step_bundle_digest_v2(
            self.step_idx,
            self.params_digest32,
            self.ccs_digest32,
            self.initial_acc_digest32,
            self.final_acc_digest32,
        )
    }

    pub fn bundle_digest_u128_limbs_le(&self) -> [u128; 2] {
        digest32_to_u128_limbs_le(self.bundle_digest32())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BundleVerifierError {
    BundleDigestMismatch {
        expected: [u128; 2],
        got: [u128; 2],
    },
    SumcheckChallengesLenMismatch {
        channel: FsChannel,
        expected: usize,
        got: usize,
    },
    SumcheckChallengeMismatch {
        channel: FsChannel,
        round: usize,
        expected: KRepr,
        got: KRepr,
    },
    GammaMismatch {
        expected: KRepr,
        got: KRepr,
    },
    AlphaLenMismatch {
        expected: usize,
        got: usize,
    },
    AlphaMismatch {
        index: usize,
        expected: KRepr,
        got: KRepr,
    },
    BetaALenMismatch {
        expected: usize,
        got: usize,
    },
    BetaAMismatch {
        index: usize,
        expected: KRepr,
        got: KRepr,
    },
    BetaRLenMismatch {
        expected: usize,
        got: usize,
    },
    BetaRMismatch {
        index: usize,
        expected: KRepr,
        got: KRepr,
    },
    BetaMLenMismatch {
        expected: usize,
        got: usize,
    },
    BetaMMismatch {
        index: usize,
        expected: KRepr,
        got: KRepr,
    },
}

pub fn verify_bundle_digest_v2(
    statement: &StepBundleStatementV2,
    claimed_bundle_digest: [u128; 2],
) -> Result<[u8; 32], BundleVerifierError> {
    let expected = statement.bundle_digest_u128_limbs_le();
    if expected != claimed_bundle_digest {
        return Err(BundleVerifierError::BundleDigestMismatch {
            expected,
            got: claimed_bundle_digest,
        });
    }
    Ok(statement.bundle_digest32())
}

pub fn verify_sumcheck_challenges_from_rounds(
    bundle_digest32: [u8; 32],
    channel: FsChannel,
    rounds: &[Vec<KRepr>],
    claimed_challenges: &[KRepr],
) -> Result<Vec<KRepr>, BundleVerifierError> {
    let expected = derive_all_sumcheck_challenges(bundle_digest32, channel, rounds);
    if expected.len() != claimed_challenges.len() {
        return Err(BundleVerifierError::SumcheckChallengesLenMismatch {
            channel,
            expected: expected.len(),
            got: claimed_challenges.len(),
        });
    }
    for (round, (exp, got)) in expected
        .iter()
        .copied()
        .zip(claimed_challenges.iter().copied())
        .enumerate()
    {
        if exp != got {
            return Err(BundleVerifierError::SumcheckChallengeMismatch {
                channel,
                round,
                expected: exp,
                got,
            });
        }
    }
    Ok(expected)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedStepFsValues {
    pub fe_sumcheck_challenges: Vec<KRepr>,
    pub nc_sumcheck_challenges: Vec<KRepr>,
    pub gamma: KRepr,
    pub alpha: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_r: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
}

pub fn derive_step_fs_values(
    bundle_digest32: [u8; 32],
    fe_rounds: &[Vec<KRepr>],
    nc_rounds: &[Vec<KRepr>],
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
) -> DerivedStepFsValues {
    let fe_sumcheck_challenges = derive_all_sumcheck_challenges(bundle_digest32, FsChannel::Fe, fe_rounds);
    let nc_sumcheck_challenges = derive_all_sumcheck_challenges(bundle_digest32, FsChannel::Nc, nc_rounds);
    let gamma = derive_gamma_shared(bundle_digest32, &fe_sumcheck_challenges, &nc_sumcheck_challenges);
    let alpha = derive_alpha_shared(bundle_digest32, &fe_sumcheck_challenges, &nc_sumcheck_challenges, ell_d);
    let beta_a = derive_beta_a_shared(bundle_digest32, &fe_sumcheck_challenges, &nc_sumcheck_challenges, ell_d);
    let beta_r = derive_beta_r_fe(bundle_digest32, &fe_sumcheck_challenges, ell_n);
    let beta_m = derive_beta_m_nc(bundle_digest32, &nc_sumcheck_challenges, ell_m);

    DerivedStepFsValues {
        fe_sumcheck_challenges,
        nc_sumcheck_challenges,
        gamma,
        alpha,
        beta_a,
        beta_r,
        beta_m,
    }
}

pub fn verify_step_fs_values(
    derived: &DerivedStepFsValues,
    claimed_gamma: KRepr,
    claimed_alpha: &[KRepr],
    claimed_beta_a: &[KRepr],
    claimed_beta_r: &[KRepr],
    claimed_beta_m: &[KRepr],
) -> Result<(), BundleVerifierError> {
    if derived.gamma != claimed_gamma {
        return Err(BundleVerifierError::GammaMismatch {
            expected: derived.gamma,
            got: claimed_gamma,
        });
    }

    if derived.alpha.len() != claimed_alpha.len() {
        return Err(BundleVerifierError::AlphaLenMismatch {
            expected: derived.alpha.len(),
            got: claimed_alpha.len(),
        });
    }
    for (i, (exp, got)) in derived
        .alpha
        .iter()
        .copied()
        .zip(claimed_alpha.iter().copied())
        .enumerate()
    {
        if exp != got {
            return Err(BundleVerifierError::AlphaMismatch {
                index: i,
                expected: exp,
                got,
            });
        }
    }

    if derived.beta_a.len() != claimed_beta_a.len() {
        return Err(BundleVerifierError::BetaALenMismatch {
            expected: derived.beta_a.len(),
            got: claimed_beta_a.len(),
        });
    }
    for (i, (exp, got)) in derived
        .beta_a
        .iter()
        .copied()
        .zip(claimed_beta_a.iter().copied())
        .enumerate()
    {
        if exp != got {
            return Err(BundleVerifierError::BetaAMismatch {
                index: i,
                expected: exp,
                got,
            });
        }
    }

    if derived.beta_r.len() != claimed_beta_r.len() {
        return Err(BundleVerifierError::BetaRLenMismatch {
            expected: derived.beta_r.len(),
            got: claimed_beta_r.len(),
        });
    }
    for (i, (exp, got)) in derived
        .beta_r
        .iter()
        .copied()
        .zip(claimed_beta_r.iter().copied())
        .enumerate()
    {
        if exp != got {
            return Err(BundleVerifierError::BetaRMismatch {
                index: i,
                expected: exp,
                got,
            });
        }
    }

    if derived.beta_m.len() != claimed_beta_m.len() {
        return Err(BundleVerifierError::BetaMLenMismatch {
            expected: derived.beta_m.len(),
            got: claimed_beta_m.len(),
        });
    }
    for (i, (exp, got)) in derived
        .beta_m
        .iter()
        .copied()
        .zip(claimed_beta_m.iter().copied())
        .enumerate()
    {
        if exp != got {
            return Err(BundleVerifierError::BetaMMismatch {
                index: i,
                expected: exp,
                got,
            });
        }
    }

    Ok(())
}
