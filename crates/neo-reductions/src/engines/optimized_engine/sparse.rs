//! Compressed Sparse Column (CSC) matrix format for efficient sparse operations.
//!
//! This module provides CSC:
//! - CSC: Efficient for column-wise operations (y = A^T·x) - better cache locality

#![allow(non_snake_case)]

use neo_ccs::Mat;
use p3_field::{Field, PrimeCharacteristicRing};

/// Compressed Sparse Column format (column-major iteration).
/// Better for A^T·x operations with dense x (contiguous accumulation into y).
#[derive(Clone, Debug)]
pub struct CscMat<Ff> {
    pub ncols: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub vals: Vec<Ff>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy> CscMat<Ff> {
    /// Build CSC from a dense row-major Mat<Ff>, skipping exact zeros.
    pub fn from_dense_row_major(a: &Mat<Ff>) -> Self {
        let (nrows, ncols) = (a.rows(), a.cols());

        // Count nnz per column
        let mut col_counts = vec![0usize; ncols];
        for r in 0..nrows {
            for c in 0..ncols {
                if a[(r, c)] != Ff::ZERO {
                    col_counts[c] += 1;
                }
            }
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for c in 0..ncols {
            col_ptr.push(col_ptr[c] + col_counts[c]);
        }

        let nnz = col_ptr[ncols];
        let mut row_idx = vec![0usize; nnz];
        let mut vals = vec![Ff::ZERO; nnz];

        // Temp write heads
        let mut next = col_ptr.clone();
        for r in 0..nrows {
            for c in 0..ncols {
                let v = a[(r, c)];
                if v != Ff::ZERO {
                    let k = next[c];
                    row_idx[k] = r;
                    vals[k] = v;
                    next[c] += 1;
                }
            }
        }

        CscMat {
            ncols,
            col_ptr,
            row_idx,
            vals,
        }
    }
}
