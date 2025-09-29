use anyhow::Result;
use spartan2::traits::circuit::SpartanCircuit;

use neo_ccs::{MEInstance, MEWitness};

use crate::me_to_r1cs::MeCircuit;

/// Cheap guard: encoder bytes vs circuit public_values bytes parity.
/// Intended for debug/CI; call right before proving.
pub fn assert_public_io_parity(
    me: &MEInstance,
    wit: &MEWitness,
    ev: Option<&crate::me_to_r1cs::IvcEvEmbed>,
    pp: Option<std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>>,
) -> Result<()> {
    // 1) Canonical encoder (must match MeCircuit::public_values)
    let enc = crate::encode_bridge_io_header_with_ev(me, ev);

    // 2) Circuit’s own public_values
    let circ = MeCircuit::new(me.clone(), wit.clone(), pp, me.header_digest).with_ev(ev.cloned());
    let scalars = circ.public_values()
        .map_err(|e| anyhow::anyhow!("public_values() failed: {e}"))?;
    let mut via_circuit = Vec::with_capacity(scalars.len() * 8);
    for s in scalars { via_circuit.extend_from_slice(&s.to_canonical_u64().to_le_bytes()); }

    anyhow::ensure!(enc == via_circuit, "Public IO parity failed: encoder vs circuit bytes differ");
    Ok(())
}
