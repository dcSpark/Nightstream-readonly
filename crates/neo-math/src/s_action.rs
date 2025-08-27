//! S-action implementation: "left multiplication by a in R_q" as rot(a).
//! Definitionally correct: j-th column is cf(a * X^j mod Phi_81).

use crate::{Rq, Fq, D};
use crate::ring::{cf, cf_inv};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

#[derive(Clone, Debug)]
pub struct SAction { a: Rq }

impl SAction {
    /// Create S-action from ring element a: rot(a) matrix via column-wise definition
    pub fn from_ring(a: Rq) -> Self { Self { a } }

    /// Build the full d×d rotation matrix definitionally: column j = cf(a * X^j mod Phi)
    pub fn to_matrix(&self) -> DenseMatrix<Fq> {
        let mut values = vec![Fq::ZERO; D * D];
        let mut x_power = Rq::one();  // Start with X^0 = 1
        
        for j in 0..D {
            let col = cf(self.a.mul(&x_power));
            for i in 0..D {
                values[i * D + j] = col[i];  // Column j, row i
            }
            x_power = x_power.mul_by_monomial(1);  // X^j -> X^(j+1)
        }
        
        DenseMatrix::new(values, D)
    }

    /// Left action on v ∈ F_q^d.
    #[inline] pub fn apply_vec(&self, v: &[Fq; D]) -> [Fq; D] {
        let prod = self.a.mul(&cf_inv(*v));
        cf(prod)
    }

    /// Left action on a d×m matrix (columns are vectors).
    #[inline] pub fn apply_matrix(&self, z: &DenseMatrix<Fq>) -> DenseMatrix<Fq> {
        // The heavy lifting is in ring::rot_apply_matrix which validates shape.
        crate::ring::rot_apply_matrix(&self.a, z).expect("shape checked")
    }

    /// Compose S-actions (rot(a) ∘ rot(b) = rot(a*b)).
    #[inline] pub fn compose(&self, other: &SAction) -> SAction {
        SAction { a: self.a.mul(&other.a) }
    }
}
