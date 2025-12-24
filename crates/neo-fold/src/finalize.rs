use crate::shard_proof_types::ShardObligations;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinalizeReport {
    pub did_finalize_main: bool,
    pub did_finalize_val: bool,
}

pub trait ObligationFinalizer<Cmt, F, K> {
    type Error;

    /// Finalize all shard obligations.
    ///
    /// Implementations must handle both `obligations.main` and `obligations.val` (Twist `r_val` lane).
    /// Prefer iterating `obligations.iter_all()` unless you intentionally treat lanes differently.
    fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error>;
}
