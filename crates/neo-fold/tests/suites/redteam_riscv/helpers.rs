use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::K;
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::StepWitnessBundle;
use neo_params::NeoParams;
use p3_goldilocks::Goldilocks as F;

pub type StepWit = StepWitnessBundle<Cmt, F, K>;

pub fn collect_mcs(instances: &[StepWit]) -> (Vec<McsInstance<Cmt, F>>, Vec<McsWitness<F>>) {
    let mut insts = Vec::with_capacity(instances.len());
    let mut wits = Vec::with_capacity(instances.len());
    for step in instances {
        let (inst, wit) = &step.mcs;
        insts.push(inst.clone());
        wits.push(wit.clone());
    }
    (insts, wits)
}

pub fn mcs_recommit_step_after_private_tamper(
    params: &NeoParams,
    committer: &AjtaiSModule,
    mcs_inst: &mut McsInstance<Cmt, F>,
    mcs_wit: &mut McsWitness<F>,
    idx_to_tamper: usize,
    delta: F,
) {
    let m_in = mcs_inst.m_in;
    assert!(
        idx_to_tamper >= m_in,
        "expected idx_to_tamper to be in the private witness region (idx={idx_to_tamper}, m_in={m_in})"
    );

    let mut z = Vec::with_capacity(m_in + mcs_wit.w.len());
    z.extend_from_slice(&mcs_inst.x);
    z.extend_from_slice(&mcs_wit.w);
    assert!(
        idx_to_tamper < z.len(),
        "idx_to_tamper out of range: idx={idx_to_tamper} len={}",
        z.len()
    );

    let x_before = mcs_inst.x.clone();
    z[idx_to_tamper] += delta;
    assert_eq!(
        &z[..m_in],
        &x_before[..],
        "tamper helper must not modify public x region"
    );

    mcs_wit.w = z[m_in..].to_vec();
    mcs_wit.Z = encode_vector_balanced_to_mat(params, &z);
    mcs_inst.c = committer.commit(&mcs_wit.Z);
}

pub fn step_bundle_recommit_after_private_tamper(
    params: &NeoParams,
    committer: &AjtaiSModule,
    step: &mut StepWit,
    idx_to_tamper: usize,
    delta: F,
) {
    let (ref mut inst, ref mut wit) = step.mcs;
    mcs_recommit_step_after_private_tamper(params, committer, inst, wit, idx_to_tamper, delta);
}

pub fn assert_prove_or_verify_fails(res: Result<bool, neo_fold::PiCcsError>, label: &str) {
    match res {
        Ok(true) => panic!("{label}: unexpectedly verified"),
        Ok(false) | Err(_) => {}
    }
}
