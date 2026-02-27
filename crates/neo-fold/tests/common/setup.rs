#![allow(dead_code)]

use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::CcsStructure;
use neo_ccs::sparse::{CcsMatrix, CscMat};
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::{D, F};
use neo_memory::ajtai::commit_cols_for_ccs_m;
use neo_memory::cpu::{build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape};
use neo_memory::riscv::trace::rv32_trace_lookup_n_vals_for_table_id;
use neo_memory::witness::{StepWitnessBundle, TimeColumns};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

pub fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let m_commit = commit_cols_for_ccs_m(m);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m_commit).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

pub fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

pub fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for c in cs.iter().skip(1) {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, c);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

/// Test-only helper: widen CCS column count by appending all-zero columns to every matrix.
/// This is used by legacy no-shared packed-bus tests when bus tails need more per-step columns
/// than the optimized RV32 trace core now commits.
pub fn widen_ccs_cols_for_test(ccs: &mut CcsStructure<F>, target_m: usize) {
    if target_m <= ccs.m {
        return;
    }
    for mat in &mut ccs.matrices {
        match mat {
            CcsMatrix::Identity { n } => {
                let nrows = *n;
                let diag = nrows.min(target_m);
                let mut col_ptr = Vec::with_capacity(target_m + 1);
                for c in 0..=target_m {
                    col_ptr.push(c.min(diag));
                }
                let row_idx: Vec<usize> = (0..diag).collect();
                let vals = vec![F::ONE; diag];
                *mat = CcsMatrix::Csc(CscMat {
                    nrows,
                    ncols: target_m,
                    col_ptr,
                    row_idx,
                    vals,
                });
            }
            CcsMatrix::Csc(csc) => {
                if csc.ncols > target_m {
                    continue;
                }
                let nnz = *csc.col_ptr.last().unwrap_or(&0);
                csc.col_ptr.resize(target_m + 1, nnz);
                csc.ncols = target_m;
            }
        }
    }
    ccs.m = target_m;
}

fn infer_test_chunk_size<Cmt, K>(step: &StepWitnessBundle<Cmt, F, K>) -> usize {
    let mut chunk = 0usize;
    for (inst, _) in &step.lut_instances {
        chunk = chunk.max(inst.steps);
    }
    for (inst, _) in &step.mem_instances {
        chunk = chunk.max(inst.steps);
    }
    chunk.max(1)
}

fn derive_time_columns_from_step_witness<Cmt, K>(step: &StepWitnessBundle<Cmt, F, K>) -> TimeColumns<F> {
    let m_in = step.mcs.0.m_in;
    let mut z = Vec::with_capacity(step.mcs.0.x.len().saturating_add(step.mcs.1.w.len()));
    z.extend_from_slice(&step.mcs.0.x);
    z.extend_from_slice(&step.mcs.1.w);
    let m = z.len();
    assert!(
        m >= m_in,
        "derive_time_columns_from_step_witness: malformed witness (m={m} < m_in={m_in})"
    );

    let chunk_size = infer_test_chunk_size(step);
    let shout_shapes = step
        .lut_instances
        .iter()
        .map(|(inst, _)| ShoutInstanceShape {
            ell_addr: inst.d * inst.ell,
            lanes: inst.lanes,
            n_vals: rv32_trace_lookup_n_vals_for_table_id(inst.table_id),
            addr_group: inst.addr_group,
            selector_group: inst.selector_group,
        });
    let twist_shapes = step
        .mem_instances
        .iter()
        .map(|(inst, _)| (inst.d * inst.ell, inst.lanes));
    let bus = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m,
        m_in,
        chunk_size,
        shout_shapes,
        twist_shapes,
    )
    .expect("derive_time_columns_from_step_witness: failed to infer bus layout");
    assert!(
        bus.bus_base >= m_in && bus.bus_base + bus.bus_cols <= m,
        "derive_time_columns_from_step_witness: invalid bus layout for witness width (m={m}, m_in={m_in}, bus_base={}, bus_cols={})",
        bus.bus_base,
        bus.bus_cols
    );

    let cpu_flat = bus.bus_base - m_in;
    let bus_cols = bus.bus_cols;
    let flattened = chunk_size > 0
        && (m - m_in) % chunk_size == 0
        && ((m - m_in) / chunk_size) >= bus_cols
        && cpu_flat % chunk_size == 0;

    if flattened {
        let t = chunk_size;
        let total_cols = (m - m_in) / t;
        let cpu_cols_len = total_cols - bus_cols;
        let mut cpu_cols = Vec::with_capacity(cpu_cols_len);
        for col in 0..cpu_cols_len {
            let start = m_in + col * t;
            cpu_cols.push(z[start..start + t].to_vec());
        }
        let mem_start = m_in + cpu_cols_len * t;
        let mut mem_cols = Vec::with_capacity(bus_cols);
        for col in 0..bus_cols {
            let start = mem_start + col * t;
            mem_cols.push(z[start..start + t].to_vec());
        }
        let mut col_ids = Vec::with_capacity(cpu_cols_len + bus_cols);
        col_ids.extend(0..(cpu_cols_len + bus_cols));
        TimeColumns {
            t,
            cpu_cols,
            mem_cols,
            active_col: vec![F::ONE; t],
            col_ids,
        }
    } else {
        let t = 1usize;
        let cpu_cols_len = cpu_flat;
        let mut cpu_cols = Vec::with_capacity(cpu_cols_len);
        for col in 0..cpu_cols_len {
            cpu_cols.push(vec![z[m_in + col]]);
        }
        let mut mem_cols = Vec::with_capacity(bus_cols);
        for col in 0..bus_cols {
            mem_cols.push(vec![z[bus.bus_base + col]]);
        }
        let mut col_ids = Vec::with_capacity(cpu_cols_len + bus_cols);
        col_ids.extend(0..(cpu_cols_len + bus_cols));
        TimeColumns {
            t,
            cpu_cols,
            mem_cols,
            active_col: vec![F::ONE],
            col_ids,
        }
    }
}

pub fn canonicalize_step_time_columns<Cmt, K>(mut step: StepWitnessBundle<Cmt, F, K>) -> StepWitnessBundle<Cmt, F, K> {
    if step.time_columns.t == 0 {
        step.time_columns = derive_time_columns_from_step_witness(&step);
    }
    step
}

pub fn empty_time_columns() -> TimeColumns<F> {
    TimeColumns {
        t: 0,
        cpu_cols: Vec::new(),
        mem_cols: Vec::new(),
        active_col: Vec::new(),
        col_ids: Vec::new(),
    }
}
