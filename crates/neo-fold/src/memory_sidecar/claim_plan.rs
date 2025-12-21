use core::ops::Range;

use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_memory::witness::StepWitnessBundle;

use crate::PiCcsError;

#[derive(Clone, Debug)]
pub struct ShoutTimeClaimIdx {
    pub value: usize,
    pub adapter: usize,
    pub bitness_addr_bits: Range<usize>,
    pub bitness_has_lookup: usize,
    pub ell_addr: usize,
}

#[derive(Clone, Debug)]
pub struct TwistTimeClaimIdx {
    pub read_check: usize,
    pub write_check: usize,
    pub bitness_ra_bits: Range<usize>,
    pub bitness_wa_bits: Range<usize>,
    pub bitness_has_read: usize,
    pub bitness_has_write: usize,
    pub ell_addr: usize,
}

/// Deterministic claim schedule for Route A batched time claims (memory sidecar only).
///
/// This is a single source of truth for how indices into `batched_claimed_sums` /
/// `batched_final_values` map to each Shout/Twist instance.
#[derive(Clone, Debug)]
pub struct RouteATimeClaimPlan {
    pub claim_idx_start: usize,
    pub claim_idx_end: usize,
    pub shout: Vec<ShoutTimeClaimIdx>,
    pub twist: Vec<TwistTimeClaimIdx>,
}

impl RouteATimeClaimPlan {
    pub fn build(
        step: &StepWitnessBundle<Cmt, F, K>,
        claim_idx_start: usize,
    ) -> Result<RouteATimeClaimPlan, PiCcsError> {
        let mut idx = claim_idx_start;
        let mut shout = Vec::with_capacity(step.lut_instances.len());
        let mut twist = Vec::with_capacity(step.mem_instances.len());

        for (lut_inst, _lut_wit) in &step.lut_instances {
            let ell_addr = lut_inst.d * lut_inst.ell;
            let value = idx;
            idx += 1;
            let adapter = idx;
            idx += 1;
            let bitness_addr_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_has_lookup = idx;
            idx += 1;

            shout.push(ShoutTimeClaimIdx {
                value,
                adapter,
                bitness_addr_bits,
                bitness_has_lookup,
                ell_addr,
            });
        }

        for (mem_inst, _mem_wit) in &step.mem_instances {
            let ell_addr = mem_inst.d * mem_inst.ell;
            let read_check = idx;
            idx += 1;
            let write_check = idx;
            idx += 1;

            let bitness_ra_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_wa_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_has_read = idx;
            idx += 1;
            let bitness_has_write = idx;
            idx += 1;

            twist.push(TwistTimeClaimIdx {
                read_check,
                write_check,
                bitness_ra_bits,
                bitness_wa_bits,
                bitness_has_read,
                bitness_has_write,
                ell_addr,
            });
        }

        if idx < claim_idx_start {
            return Err(PiCcsError::ProtocolError("RouteATimeClaimPlan index underflow".into()));
        }

        Ok(RouteATimeClaimPlan {
            claim_idx_start,
            claim_idx_end: idx,
            shout,
            twist,
        })
    }
}
