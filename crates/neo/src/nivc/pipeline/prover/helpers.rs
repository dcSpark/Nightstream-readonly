//! Helper functions for NIVC proving

use crate::F;
use crate::ivc::Accumulator;
use super::super::super::api::LaneRunningState;
use p3_field::PrimeCharacteristicRing;

/// Build a lane‑scoped Accumulator view for the existing IVC prover
pub fn build_prev_acc_lane(lane: &LaneRunningState, global_y: &[F], step: u64) -> Accumulator {
    Accumulator {
        c_z_digest: lane.c_digest,
        c_coords: lane.c_coords.clone(),
        y_compact: global_y.to_vec(),
        step,
    }
}

/// Build application inputs for the step: [which_type || step_io || lanes_root]
pub fn make_app_inputs(which: usize, step_io: &[F], lanes_root: &[F]) -> Vec<F> {
    let mut app_inputs = Vec::with_capacity(1 + step_io.len() + lanes_root.len());
    app_inputs.push(F::from_u64(which as u64));
    app_inputs.extend_from_slice(step_io);
    app_inputs.extend_from_slice(lanes_root);
    app_inputs
}

