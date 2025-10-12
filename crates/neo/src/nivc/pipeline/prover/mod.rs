//! NIVC proving pipeline

use crate::F;
use crate::ivc::{IvcStepInput, prove_ivc_step_chained};
use super::super::api::{NivcState, NivcStepProof};
use super::super::internal::{IndexExtractor, lanes_root_fields};

mod helpers;
use helpers::{build_prev_acc_lane, make_app_inputs};

/// Execute one NIVC step for lane `which` with given step IO and witness.
/// Returns the step proof and updates internal state.
pub fn step(
    state: &mut NivcState,
    which: usize,
    step_io: &[F],
    step_witness: &[F],
) -> anyhow::Result<NivcStepProof> {
    if which >= state.program.len() { 
        anyhow::bail!("which_type out of bounds"); 
    }
    let spec = &state.program.steps[which];

    // Build a lane‑scoped Accumulator view for the existing IVC prover
    let lane = &state.acc.lanes[which];
    let prev_acc_lane = build_prev_acc_lane(lane, &state.acc.global_y, state.acc.step);

    // Public input: bind which_type and lanes_root to the FS transcript via step_x
    let lanes_root = lanes_root_fields(&state.acc);
    let app_inputs = make_app_inputs(which, step_io, &lanes_root);

    // Extract y_step from witness using binding spec offsets
    let extractor = IndexExtractor { 
        indices: spec.binding.y_step_offsets.clone() 
    };
    let y_step = extractor.extract_y_step(step_witness);

    // Thread the running ME for this lane (if any)
    let prev_me = state.acc.lanes[which].me.clone();
    let prev_wit = state.acc.lanes[which].wit.clone();
    let prev_mcs = state.acc.lanes[which].lhs_mcs.clone()
        .zip(state.acc.lanes[which].lhs_mcs_wit.clone());

    // Prove the step using the existing chained IVC helper
    let input = IvcStepInput {
        params: &state.params,
        step_ccs: &spec.ccs,
        step_witness,
        prev_accumulator: &prev_acc_lane,
        step: state.acc.step,
        public_input: Some(&app_inputs),
        y_step: &y_step,
        binding_spec: &spec.binding,
        transcript_only_app_inputs: true,
        prev_augmented_x: state.prev_aug_x_by_lane[which].as_deref(),
    };
    let (res, me_out, wit_out, lhs_next) = prove_ivc_step_chained(input, prev_me, prev_wit, prev_mcs)
        .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;

    // Update lane state: carry ME forward and refresh commitment coords/digest
    let lane_mut = &mut state.acc.lanes[which];
    lane_mut.me = Some(me_out);
    lane_mut.wit = Some(wit_out);
    lane_mut.c_coords = res.proof.next_accumulator.c_coords.clone();
    lane_mut.c_digest = res.proof.next_accumulator.c_z_digest;
    lane_mut.lhs_mcs = Some(lhs_next.0);
    lane_mut.lhs_mcs_wit = Some(lhs_next.1);

    // Update global state
    state.acc.global_y = res.proof.next_accumulator.y_compact.clone();
    state.acc.step += 1;

    // Update lane-local previous augmented X for linking next time this lane is used
    state.prev_aug_x_by_lane[which] = Some(res.proof.step_augmented_public_input.clone());

    let sp = NivcStepProof { 
        which_type: which, 
        step_io: step_io.to_vec(), 
        inner: res.proof 
    };
    state.steps.push(sp.clone());
    Ok(sp)
}

