use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

/// Public parameters for Ajtai: M ∈ R_q^{κ×m}, stored row-major.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PP<RqEl> {
    pub kappa: usize,
    pub m: usize,
    pub d: usize,
    /// Ajtai matrix rows; each row is a vector of ring elements of length m.
    pub m_rows: Vec<Vec<RqEl>>,
}

/// Commitment c ∈ F_q^{d×κ}, stored as column-major flat matrix (κ columns, each length d).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Commitment {
    pub d: usize,
    pub kappa: usize,
    /// data[c * d + i] = i-th row of column c
    pub data: Vec<Fq>,
}

impl Commitment {
    #[inline]
    fn validate_shape(d: usize, kappa: usize, data_len: usize) -> Result<(), String> {
        let expected_d = neo_math::ring::D;
        if d != expected_d {
            return Err(format!("invalid Commitment.d: expected {expected_d}, got {d}"));
        }

        let expected_len = d
            .checked_mul(kappa)
            .ok_or_else(|| format!("invalid Commitment shape: d*kappa overflow (d={d}, kappa={kappa})"))?;
        if data_len != expected_len {
            return Err(format!(
                "invalid Commitment shape: data.len()={} but d*kappa={expected_len}",
                data_len
            ));
        }

        Ok(())
    }

    pub fn zeros(d: usize, kappa: usize) -> Self {
        Self {
            d,
            kappa,
            data: vec![Fq::ZERO; d * kappa],
        }
    }

    #[inline]
    pub fn col(&self, c: usize) -> &[Fq] {
        &self.data[c * self.d..(c + 1) * self.d]
    }

    #[inline]
    pub fn col_mut(&mut self, c: usize) -> &mut [Fq] {
        &mut self.data[c * self.d..(c + 1) * self.d]
    }

    pub fn add_inplace(&mut self, rhs: &Commitment) {
        debug_assert_eq!(self.d, rhs.d);
        debug_assert_eq!(self.kappa, rhs.kappa);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += *b;
        }
    }
}

impl<'de> Deserialize<'de> for Commitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CommitmentWire {
            d: usize,
            kappa: usize,
            data: Vec<Fq>,
        }

        let wire = CommitmentWire::deserialize(deserializer)?;
        Commitment::validate_shape(wire.d, wire.kappa, wire.data.len()).map_err(D::Error::custom)?;

        Ok(Self {
            d: wire.d,
            kappa: wire.kappa,
            data: wire.data,
        })
    }
}
