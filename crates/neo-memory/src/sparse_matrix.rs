//! Sparse matrix representations for multilinear polynomials.
//!
//! This module is a small, self-contained primitive inspired by Jolt's `read_write_matrix`
//! data structures (`external/jolt/.../read_write_matrix`). The core idea is:
//!
//! - A matrix `M(row, col)` over Boolean hypercubes is represented by its non-zero entries.
//! - Binding (folding) one variable corresponds to combining pairs of indices (2k,2k+1) with
//!   weights `(1-r, r)`, exactly like multilinear extension folding.
//!
//! The implementation here is intentionally simple (O(nnz) per bind) and is meant as a
//! correctness-first building block for future "Jolt-like" sparse arguments.

use p3_field::PrimeCharacteristicRing;

/// A sparse matrix entry at `(row, col)` with non-zero coefficient `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatEntry<R> {
    pub row: u64,
    pub col: u64,
    pub value: R,
}

/// Sparse matrix over a `2^ell_row × 2^ell_col` Boolean grid.
///
/// Invariant:
/// - dimensions are tracked by `(ell_row, ell_col)` (no `1 << ell` allocation)
/// - `entries` are strictly increasing by `(row, col)`
/// - all stored values are non-zero
#[derive(Clone, Debug)]
pub struct SparseMat<R> {
    ell_row: usize,
    ell_col: usize,
    entries: Vec<SparseMatEntry<R>>,
}

impl<R> SparseMat<R>
where
    R: PrimeCharacteristicRing + Copy + PartialEq,
{
    pub fn new(ell_row: usize, ell_col: usize) -> Self {
        Self {
            ell_row,
            ell_col,
            entries: Vec::new(),
        }
    }

    pub fn ell_row(&self) -> usize {
        self.ell_row
    }

    pub fn ell_col(&self) -> usize {
        self.ell_col
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[SparseMatEntry<R>] {
        &self.entries
    }

    /// Build a sparse matrix from arbitrary entries.
    ///
    /// Entries are normalized by:
    /// - dropping zeros,
    /// - sorting by `(row, col)`,
    /// - combining duplicates by summation.
    pub fn from_entries(ell_row: usize, ell_col: usize, mut entries: Vec<SparseMatEntry<R>>) -> Self {
        entries.retain(|e| e.value != R::ZERO);
        entries.sort_by_key(|e| (e.row, e.col));

        let mut out: Vec<SparseMatEntry<R>> = Vec::with_capacity(entries.len());
        for e in entries {
            debug_assert!(ell_row >= 64 || (e.row >> ell_row) == 0, "entry row out of range");
            debug_assert!(ell_col >= 64 || (e.col >> ell_col) == 0, "entry col out of range");
            if let Some(last) = out.last_mut() {
                if last.row == e.row && last.col == e.col {
                    last.value += e.value;
                    if last.value == R::ZERO {
                        out.pop();
                    }
                    continue;
                }
            }
            out.push(e);
        }

        #[cfg(debug_assertions)]
        for w in out.windows(2) {
            let a = &w[0];
            let b = &w[1];
            debug_assert!(
                (a.row, a.col) < (b.row, b.col),
                "SparseMat entries must be strictly increasing"
            );
            debug_assert!(a.value != R::ZERO && b.value != R::ZERO, "SparseMat stores zeros");
        }

        Self {
            ell_row,
            ell_col,
            entries: out,
        }
    }

    /// Get the coefficient at `(row, col)`, returning `0` if not present.
    pub fn get(&self, row: u64, col: u64) -> R {
        if (self.ell_row < 64 && (row >> self.ell_row) != 0) || (self.ell_col < 64 && (col >> self.ell_col) != 0) {
            return R::ZERO;
        }
        match self
            .entries
            .binary_search_by_key(&(row, col), |e| (e.row, e.col))
        {
            Ok(pos) => self.entries[pos].value,
            Err(_) => R::ZERO,
        }
    }

    /// One multilinear folding round on the least-significant row bit.
    ///
    /// For each parent row `p` and column `c`:
    /// `out[p, c] = in[2p, c] * (1-r) + in[2p+1, c] * r`
    pub fn fold_row_round_in_place(&mut self, r: R) {
        debug_assert!(self.ell_row > 0, "cannot fold when ell_row == 0");

        let one_minus_r = R::ONE - r;
        let mut out: Vec<SparseMatEntry<R>> = Vec::with_capacity(self.entries.len());

        for e in self.entries.iter().copied() {
            let parent_row = e.row >> 1;
            let scaled = if (e.row & 1) == 0 {
                e.value * one_minus_r
            } else {
                e.value * r
            };
            if scaled == R::ZERO {
                continue;
            }

            if let Some(last) = out.last_mut() {
                if last.row == parent_row && last.col == e.col {
                    last.value += scaled;
                    if last.value == R::ZERO {
                        out.pop();
                    }
                    continue;
                }
            }

            out.push(SparseMatEntry {
                row: parent_row,
                col: e.col,
                value: scaled,
            });
        }

        self.entries = out;
        self.ell_row -= 1;
    }

    /// One multilinear folding round on the least-significant column bit.
    ///
    /// For each row `r` and parent column `p`:
    /// `out[r, p] = in[r, 2p] * (1-r) + in[r, 2p+1] * r`
    pub fn fold_col_round_in_place(&mut self, r: R) {
        debug_assert!(self.ell_col > 0, "cannot fold when ell_col == 0");

        let one_minus_r = R::ONE - r;
        let mut out: Vec<SparseMatEntry<R>> = Vec::with_capacity(self.entries.len());

        // The input is strictly sorted by `(row, col)`. Since `parent_col = col >> 1` is
        // non-decreasing in `col`, the folded entries remain sorted by `(row, parent_col)`.
        // Duplicates can occur only when both `2p` and `2p+1` are present; those are adjacent
        // in the input order, so we can merge on the fly without a global re-sort.
        for e in self.entries.iter().copied() {
            let parent_col = e.col >> 1;
            let scaled = if (e.col & 1) == 0 {
                e.value * one_minus_r
            } else {
                e.value * r
            };
            if scaled == R::ZERO {
                continue;
            }

            if let Some(last) = out.last_mut() {
                if last.row == e.row && last.col == parent_col {
                    last.value += scaled;
                    if last.value == R::ZERO {
                        out.pop();
                    }
                    continue;
                }
            }

            out.push(SparseMatEntry {
                row: e.row,
                col: parent_col,
                value: scaled,
            });
        }

        self.entries = out;
        self.ell_col -= 1;
    }

    /// Evaluate the multilinear extension of the sparse matrix at `(r_row, r_col)`
    /// by repeated folding.
    pub fn mle_eval_by_folding(&self, r_row: &[R], r_col: &[R]) -> Result<R, String> {
        if self.ell_row != r_row.len() {
            return Err(format!(
                "SparseMat: ell_row={} does not match r_row.len()={}",
                self.ell_row,
                r_row.len()
            ));
        }
        if self.ell_col != r_col.len() {
            return Err(format!(
                "SparseMat: ell_col={} does not match r_col.len()={}",
                self.ell_col,
                r_col.len()
            ));
        }

        let mut cur = self.clone();
        for &r in r_row {
            cur.fold_row_round_in_place(r);
        }
        for &r in r_col {
            cur.fold_col_round_in_place(r);
        }
        if cur.ell_row != 0 || cur.ell_col != 0 {
            return Err("SparseMat: folding did not reach ell_row=0, ell_col=0".into());
        }
        Ok(cur.entries.first().map(|e| e.value).unwrap_or(R::ZERO))
    }

    /// Evaluate the multilinear extension of the sparse matrix at `(r_row, r_col)` by direct sparse summation.
    ///
    /// This computes:
    /// `Σ_{(i,j)∈supp} M[i,j] · χ_{r_row}(i) · χ_{r_col}(j)`.
    pub fn mle_eval_direct(&self, r_row: &[R], r_col: &[R]) -> Result<R, String> {
        if self.ell_row != r_row.len() {
            return Err(format!(
                "SparseMat: ell_row={} does not match r_row.len()={}",
                self.ell_row,
                r_row.len()
            ));
        }
        if self.ell_col != r_col.len() {
            return Err(format!(
                "SparseMat: ell_col={} does not match r_col.len()={}",
                self.ell_col,
                r_col.len()
            ));
        }

        #[inline]
        fn chi_at_u64_index<R: PrimeCharacteristicRing + Copy>(r: &[R], idx: u64) -> R {
            let mut acc = R::ONE;
            for (b, &rb) in r.iter().enumerate() {
                let bit = if b < 64 { (idx >> b) & 1 } else { 0 };
                acc *= if bit == 1 { rb } else { R::ONE - rb };
            }
            acc
        }

        let mut acc = R::ZERO;
        for e in self.entries.iter().copied() {
            if self.ell_row < 64 && (e.row >> self.ell_row) != 0 {
                return Err("SparseMat: entry row out of range for ell_row".into());
            }
            if self.ell_col < 64 && (e.col >> self.ell_col) != 0 {
                return Err("SparseMat: entry col out of range for ell_col".into());
            }

            let chi_row = chi_at_u64_index(r_row, e.row);
            if chi_row == R::ZERO {
                continue;
            }
            let chi_col = chi_at_u64_index(r_col, e.col);
            if chi_col == R::ZERO {
                continue;
            }
            acc += e.value * chi_row * chi_col;
        }
        Ok(acc)
    }
}
