use crate::matrix::Mat;

/// Minimal interface needed from Ajtai (or any S-module homomorphic) commitment.
///
/// We intentionally keep this trait tiny; `neo-ajtai` will implement it.
/// - `commit(Z)` returns the commitment `c`.
/// - `project_x(Z, min)` returns the first `min` columns (i.e., `X = L_x(Z)`).
pub trait SModuleHomomorphism<F, C> {
    /// Commit to a `d × m` matrix `Z`.
    fn commit(&self, z: &Mat<F>) -> C;

    /// Commit to many `d × m` matrices.
    ///
    /// Default implementation commits one-by-one. Backends can override to batch
    /// work when many matrices share the same commitment parameters.
    fn commit_many(&self, zs: &[&Mat<F>]) -> Vec<C> {
        zs.iter().map(|z| self.commit(z)).collect()
    }

    /// Project the first `min` columns of `Z`.
    fn project_x(&self, z: &Mat<F>, min: usize) -> Mat<F>;
}
