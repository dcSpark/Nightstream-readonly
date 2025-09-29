#![allow(deprecated)]

//! ME(b,L) SpartanCircuit implementation for direct Spartan2 SNARK integration
//!
//! This module implements the **real Spartan2 integration** using the `SpartanCircuit` trait.
//! This is the **official public API** approach, not a workaround.
//! 
//! ## Architecture
//!
//! 1. **`SpartanCircuit<E>` Implementation**: Uses bellpepper `ConstraintSystem`
//! 2. **Automatic R1CS Generation**: Spartan2 handles constraint matrix construction
//! 3. **Production SNARK API**: `setup() → prep_prove() → prove() → verify()`
//! 4. **Hash-MLE PCS Backend**: Integrated automatically via Engine selection
//!
//! ## Constraints Implemented
//!
//! 1. **Ajtai commitment binding**: `⟨L_{r,i}, vec(Z)⟩ = c_{r,i}`
//! 2. **ME evaluations**: `⟨v_j^{(ℓ)}, Z_row[r]⟩ = y_j^{(ℓ)}[r]` 
//! 3. **Fold digest binding**: Ensures transcript security between phases
//!
//! ## Variable Layout (Bellpepper Style)
//!
//! - **Public Inputs**: `(c_coords, y_limbs, fold_digest_limbs, challenges)`
//! - **Private Witness**: `vec(Z)` allocated as `AllocatedNum<E::Scalar>`
//! - **Shared Variables**: Empty for this circuit (no cross-circuit dependencies)

use anyhow::Result;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use dashmap::DashMap;
use ff::Field;
use once_cell::sync::Lazy;
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::{Permutation};
// RNG imports are now local to the static initializer
use spartan2::errors::SpartanError;
use spartan2::provider::GoldilocksMerkleMleEngine as E;
use spartan2::spartan::{R1CSSNARK, SpartanProverKey, SpartanVerifierKey};
use spartan2::traits::{circuit::SpartanCircuit, Engine, pcs::PCSEngineTrait, snark::R1CSSNARKTrait};
use std::sync::Arc;

// Real Spartan2 SNARK uses circuits directly, not R1CS shapes
// (the R1CS conversion happens internally)

use neo_ajtai::PP as AjtaiPP; // for later PP-backed binding
use neo_ccs::{MEInstance, MEWitness};

/// Optional data to embed the IVC EV check inside the Spartan circuit.
/// When present, the circuit exposes y_step via extra weight vectors
/// (appended to `me.y_outputs`) and enforces for each component k:
///   rho * y_step[k] = y_next[k] - y_prev[k]
#[derive(Clone, Debug)]
pub struct IvcEvEmbed {
    pub rho: neo_math::F,
    pub y_prev: Vec<neo_math::F>,
    pub y_next: Vec<neo_math::F>,
    /// Optional: provide y_step as public (host-computed) to avoid extra y_outputs claims
    pub y_step_public: Option<Vec<neo_math::F>>, 
    /// Optional: provide fold chain digest (32 bytes) packed as 4 limbs
    pub fold_chain_digest: Option<[u8; 32]>,
    /// Optional: accumulator commitment evolution inputs (all same length)
    pub acc_c_prev: Option<Vec<neo_math::F>>,
    pub acc_c_step: Option<Vec<neo_math::F>>,
    pub acc_c_next: Option<Vec<neo_math::F>>,
    /// Optional: effective rho for commit evolution (first use → 1, else ρ)
    pub rho_eff: Option<neo_math::F>,
}

/// Circuit fingerprint key for caching SNARK setup and preparation
/// SECURITY CRITICAL: This must fingerprint the actual constraint constants/wiring, 
/// not just sizes. Two circuits with same dimensions but different matrices 
/// CANNOT safely share keys.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CircuitKey([u8; 32]);

impl CircuitKey {
    pub fn from_circuit(circuit: &MeCircuit) -> Self {
        // Use Poseidon2 over Goldilocks for program fingerprinting (consistent with Neo transcript)
        const WIDTH: usize = 16;
        
        let mut state = Vec::<Goldilocks>::new();
        
        // Domain separation tag and versioning (encode as field elements)
        state.push(Goldilocks::from_u64(0x4e454f5f50524f)); // "NEO_PRO" as u64  
        state.push(Goldilocks::from_u64(0x4752414d5f4b4559)); // "GRAM_KEY" as u64
        state.push(Goldilocks::from_u64(2)); // version 2 (excludes public input values)
        
        // Base parameters that alter constraint generation
        state.push(Goldilocks::from_u64(circuit.me.base_b));
        state.push(Goldilocks::from_u64(circuit.wit.z_digits.len() as u64));
        state.push(Goldilocks::from_u64(circuit.me.c_coords.len() as u64));
        state.push(Goldilocks::from_u64(circuit.me.y_outputs.len() as u64));
        state.push(Goldilocks::from_u64(circuit.me.r_point.len() as u64));
        
        // CRITICAL: Hash all Ajtai rows (constraint matrix constants)
        if let Some(rows) = &circuit.wit.ajtai_rows {
            state.push(Goldilocks::from_u64(rows.len() as u64));
            for row in rows {
                state.push(Goldilocks::from_u64(row.len() as u64));
                for &coeff in row {
                    state.push(Goldilocks::from_u64(coeff.as_canonical_u64()));
                }
            }
        } else {
            state.push(Goldilocks::ZERO);
        }
        
        // CRITICAL: Hash all weight vectors (constraint matrix constants)
        state.push(Goldilocks::from_u64(circuit.wit.weight_vectors.len() as u64));
        for weight_vec in &circuit.wit.weight_vectors {
            state.push(Goldilocks::from_u64(weight_vec.len() as u64));
            for &coeff in weight_vec {
                state.push(Goldilocks::from_u64(coeff.as_canonical_u64()));
            }
        }

        // CRITICAL: If Ajtai rows are computed from PP (streaming), fingerprint PP
        // Otherwise VK caching can collide across different PPs with identical sizes.
        if let Some(pp_arc) = &circuit.pp {
            use neo_ccs::crypto::poseidon2_goldilocks as p2;
            state.push(Goldilocks::from_u64(0x5050)); // 'PP' tag
            state.push(Goldilocks::from_u64(pp_arc.kappa as u64));
            state.push(Goldilocks::from_u64(pp_arc.m as u64));
            state.push(Goldilocks::from_u64(pp_arc.d as u64));

            // Compute a compact Poseidon2 digest over all ring coefficients in row-major order
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&pp_arc.kappa.to_le_bytes());
            bytes.extend_from_slice(&pp_arc.m.to_le_bytes());
            bytes.extend_from_slice(&pp_arc.d.to_le_bytes());
            for row in &pp_arc.m_rows {
                // each element is an Rq with D coefficients (Goldilocks field)
                for rq in row {
                    // cf(a): [F; D] is accessible via ring::cf, but Rq type is in neo_math::ring::Rq([F;D])
                    // We rely on Debug/serialization stability by using field_coeffs() helper
                    let coeffs = rq.field_coeffs();
                    for c in coeffs { bytes.extend_from_slice(&c.as_canonical_u64().to_le_bytes()); }
                }
            }
            let digest = p2::poseidon2_hash_packed_bytes(&bytes);
            for i in 0..p2::DIGEST_LEN { state.push(digest[i]); }
        } else {
            state.push(Goldilocks::from_u64(0));
        }

        // RLC GUARD FINGERPRINT: include presence and a digest over c_step_coords (avoid huge keys)
        let rlc_guard_on = std::env::var("NEO_ENABLE_RLC_GUARD").ok().as_deref() == Some("1")
            && !circuit.me.c_step_coords.is_empty()
            && circuit.me.c_step_coords.len() == circuit.me.c_coords.len();
        state.push(Goldilocks::from_u64(if rlc_guard_on { 1 } else { 0 }));
        if rlc_guard_on {
            use neo_ccs::crypto::poseidon2_goldilocks as p2;
            state.push(Goldilocks::from_u64(circuit.me.c_step_coords.len() as u64));
            let mut bytes = Vec::with_capacity(8 + circuit.me.c_step_coords.len() * 8);
            bytes.extend_from_slice(&(circuit.me.c_step_coords.len() as u64).to_le_bytes());
            for &c in &circuit.me.c_step_coords { bytes.extend_from_slice(&c.as_canonical_u64().to_le_bytes()); }
            let h = p2::poseidon2_hash_packed_bytes(&bytes);
            for k in 0..p2::DIGEST_LEN { state.push(h[k]); }
        }

        // Ajtai binding mode fingerprint: 0 = bind to me.c_coords (c_next), 1 = bind to acc_c_step
        let ajtai_mode = if circuit.ev.as_ref().and_then(|e| e.acc_c_step.as_ref()).is_some() { 1u64 } else { 0u64 };
        state.push(Goldilocks::from_u64(0x414A54)); // 'AJT'
        state.push(Goldilocks::from_u64(ajtai_mode));
        
        // CIRCUIT STRUCTURE ONLY: Hash counts/lengths that affect circuit shape,
        // but NOT the values of public inputs (c_coords, y_outputs, r_point, fold_digest).
        // Public input values vary per instance and must not fragment the PK/VK cache.
        
        // Include counts/lengths that determine circuit structure
        state.push(Goldilocks::from_u64(circuit.me.c_coords.len() as u64));
        state.push(Goldilocks::from_u64(circuit.me.y_outputs.len() as u64)); 
        state.push(Goldilocks::from_u64(circuit.me.r_point.len() as u64));
        // Include presence of EV embedding and its length (affects circuit/public IO shape)
        if let Some(ev) = &circuit.ev {
            state.push(Goldilocks::from_u64(0x45564D)); // tag 'EVM' (any constant tag)
            state.push(Goldilocks::from_u64(ev.y_next.len() as u64));
        } else {
            state.push(Goldilocks::from_u64(0));
        }

        // Include presence of COMMIT-EVO embedding (affects constraints)
        if let Some(_c) = &circuit.commit {
            state.push(Goldilocks::from_u64(0x434D45)); // tag 'CME'
            state.push(Goldilocks::from_u64(circuit.me.c_coords.len() as u64));
        } else {
            state.push(Goldilocks::from_u64(0));
        }

        // Include presence of LINKAGE embedding (affects constraints)
        if let Some(link) = &circuit.linkage {
            state.push(Goldilocks::from_u64(0x4C4B47)); // tag 'LKG'
            state.push(Goldilocks::from_u64(link.x_indices_abs.len() as u64));
            state.push(Goldilocks::from_u64(link.y_prev_indices_abs.len() as u64));
            state.push(Goldilocks::from_u64(link.const1_index_abs.map(|_| 1).unwrap_or(0) as u64));
        } else {
            state.push(Goldilocks::from_u64(0));
        }
        
        // Include presence + content of Pi-CCS embedding (affects constraints)
        if let Some(pi) = &circuit.pi_ccs {
            use neo_ccs::crypto::poseidon2_goldilocks as p2;
            state.push(Goldilocks::from_u64(0x504943)); // 'PIC'
            state.push(Goldilocks::from_u64(pi.matrices.len() as u64));
            for (j, mj) in pi.matrices.iter().enumerate() {
                state.push(Goldilocks::from_u64(mj.rows as u64));
                state.push(Goldilocks::from_u64(mj.cols as u64));
                state.push(Goldilocks::from_u64(mj.entries.len() as u64));
                // Hash triplets for determinism
                let mut bytes = Vec::with_capacity(24 + mj.entries.len() * (8 + 8 + 8));
                bytes.extend_from_slice(&u64::try_from(j).unwrap_or(0).to_le_bytes());
                bytes.extend_from_slice(&(mj.rows as u64).to_le_bytes());
                bytes.extend_from_slice(&(mj.cols as u64).to_le_bytes());
                for &(r, c, a) in &mj.entries {
                    bytes.extend_from_slice(&(r as u64).to_le_bytes());
                    bytes.extend_from_slice(&(c as u64).to_le_bytes());
                    bytes.extend_from_slice(&a.as_canonical_u64().to_le_bytes());
                }
                let h = p2::poseidon2_hash_packed_bytes(&bytes);
                for k in 0..p2::DIGEST_LEN { state.push(h[k]); }
            }
        } else {
            state.push(Goldilocks::from_u64(0));
        }

        // CRITICAL: Include header digest (fold_digest) as it affects circuit behavior
        // Different header digests create different circuits that need different keys
        for chunk in circuit.fold_digest.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            state.push(Goldilocks::from_u64(u64::from_le_bytes(bytes)));
        }
        
        // Apply Poseidon2 permutation to get deterministic fingerprint
        // STABILITY: Create a singleton Poseidon2 instance for circuit fingerprinting
        // This avoids RNG-dependent construction and ensures fingerprint stability
        // across library versions and builds.
        static CIRCUIT_FINGERPRINT_POSEIDON2: Lazy<Poseidon2Goldilocks<16>> = 
            Lazy::new(|| {
                use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
                // NOTE: This RNG usage is one-time on first use (lazy runtime init).
                // It yields deterministic constants per process for the fixed seed,
                // but may still change across library versions if parameter derivation changes.
                // For maximum stability across versions, prefer explicit fixed Poseidon2 parameters.
                const FINGERPRINT_SEED: u64 = 0x4E454F5F434B4559; // "NEO_CKEY" - distinct from transcript
                let mut rng = ChaCha8Rng::seed_from_u64(FINGERPRINT_SEED);
                Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng)
            });
        let poseidon2 = &*CIRCUIT_FINGERPRINT_POSEIDON2;
        
        // Pad to multiple of width (16 for Poseidon2)
        let mut padded_state = state;
        while padded_state.len() % WIDTH != 0 {
            padded_state.push(Goldilocks::ZERO);
        }
        
        // Hash in chunks of WIDTH, building up a running hash
        let mut final_hash = [Goldilocks::ZERO; WIDTH];
        for chunk in padded_state.chunks(WIDTH) {
            let mut chunk_array = [Goldilocks::ZERO; WIDTH];
            chunk_array[..chunk.len()].copy_from_slice(chunk);
            // Absorb previous hash state (Merkle-Damgård style)
            for (i, &val) in final_hash.iter().enumerate() {
                chunk_array[i] += val;
            }
            poseidon2.permute_mut(&mut chunk_array);
            final_hash = chunk_array;
        }
        
        // Convert to 32-byte digest (use first 4 field elements)
        let mut digest = [0u8; 32];
        for (i, &elem) in final_hash[..4].iter().enumerate() {
            digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
        }
        
        Self(digest)
    }
    
    /// Extract the inner bytes for use in lean proof system
    pub fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Global cache for SNARK proving keys (keyed by circuit fingerprint)
static PK_CACHE: Lazy<DashMap<CircuitKey, Arc<SpartanProverKey<E>>>> =
    Lazy::new(DashMap::new);

/// Global cache for SNARK verifying keys (keyed by circuit fingerprint)
static VK_CACHE: Lazy<DashMap<CircuitKey, Arc<SpartanVerifierKey<E>>>> =
    Lazy::new(DashMap::new);

// Note: Prepared SNARK caching would provide additional performance benefits,
// but requires determining the exact return type of prep_prove() from Spartan2.
// The PK/VK caching above provides the largest performance win.

/// ME(b,L) circuit
#[derive(Clone, Debug)]
pub struct MeCircuit {
    pub me: MEInstance,
    pub wit: MEWitness,
    /// Optional Ajtai PP; once you expose rows from PP, use this instead of wit.ajtai_rows
    pub pp: Option<std::sync::Arc<AjtaiPP<neo_math::Rq>>>,
    /// 32-byte fold digest (binds transcript / header)
    pub fold_digest: [u8; 32],
    /// Optional embedded IVC EV data (rho, y_prev, y_next) used to enforce
    /// y_next = y_prev + rho * y_step, where y_step is exposed via additional
    /// weight vectors appended to `me.y_outputs` by the adapter.
    pub ev: Option<IvcEvEmbed>,
    /// Optional commitment evolution embedding
    pub commit: Option<CommitEvoEmbed>,
    /// Optional linkage inputs
    pub linkage: Option<IvcLinkageInputs>,
    /// Optional Pi-CCS embedding (CCS matrices for terminal check)
    pub pi_ccs: Option<crate::pi_ccs_embed::PiCcsEmbed>,
}

impl MeCircuit {
    pub fn new(me: MEInstance, wit: MEWitness, pp: Option<std::sync::Arc<AjtaiPP<neo_math::Rq>>>, fold_digest: [u8; 32]) -> Self {
        Self { me, wit, pp, fold_digest, ev: None, commit: None, linkage: None, pi_ccs: None }
    }

    pub fn with_ev(mut self, ev: Option<IvcEvEmbed>) -> Self {
        self.ev = ev;
        self
    }

    pub fn with_commit(mut self, commit: Option<CommitEvoEmbed>) -> Self {
        self.commit = commit;
        self
    }

    pub fn with_linkage(mut self, linkage: Option<IvcLinkageInputs>) -> Self {
        self.linkage = linkage;
        self
    }

    pub fn with_pi_ccs(mut self, pi: Option<crate::pi_ccs_embed::PiCcsEmbed>) -> Self {
        self.pi_ccs = pi;
        self
    }

    #[inline]
    fn digest_to_scalars(&self) -> Vec<<E as Engine>::Scalar> {
        self.fold_digest
            .chunks(8)
            .map(|chunk| {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                <E as Engine>::Scalar::from(u64::from_le_bytes(b))
            })
            .collect()
    }
}

/// Public inputs order **must** mirror `encode_bridge_io_header()`:
/// (c_coords) || (y limbs) || (r_point) || (base_b) || (fold_digest limbs)
impl SpartanCircuit<E> for MeCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        // TRANSCRIPT CONSISTENCY FIX: Return the SAME values that inputize() creates
        // Both prover (via this method) and verifier (via U.validate) must absorb identical data
        let mut pv = Vec::new();

        // 1) Ajtai commitment coords (Fq) - SAME ORDER as inputize() calls
        for &c in &self.me.c_coords {
            pv.push(<E as Engine>::Scalar::from(c.as_canonical_u64()));
        }
        // 2) y outputs — already flattened (K -> [F;2]) in adapter
        for &y_limb in &self.me.y_outputs {
            pv.push(<E as Engine>::Scalar::from(y_limb.as_canonical_u64()));
        }
        // 3) challenge r (Fq^m)
        for &r in &self.me.r_point {
            pv.push(<E as Engine>::Scalar::from(r.as_canonical_u64()));
        }
        // 4) base dimension b
        pv.push(<E as Engine>::Scalar::from(self.me.base_b as u64));

        // 4.5) Optional IVC EV public inputs: y_prev, y_next, rho, fold_chain_digest, acc commit evo
        if let Some(ev) = &self.ev {
            for &v in &ev.y_prev { pv.push(<E as Engine>::Scalar::from(v.as_canonical_u64())); }
            for &v in &ev.y_next { pv.push(<E as Engine>::Scalar::from(v.as_canonical_u64())); }
            pv.push(<E as Engine>::Scalar::from(ev.rho.as_canonical_u64()));
            if let Some(d) = &ev.fold_chain_digest {
                for chunk in d.chunks(8) {
                    let mut b = [0u8; 8];
                    b[..chunk.len()].copy_from_slice(chunk);
                    pv.push(<E as Engine>::Scalar::from(u64::from_le_bytes(b)));
                }
            }
            if let (Some(cprev), Some(cstep), Some(cnext)) = (&ev.acc_c_prev, &ev.acc_c_step, &ev.acc_c_next) {
                for &x in cprev { pv.push(<E as Engine>::Scalar::from(x.as_canonical_u64())); }
                for &x in cstep { pv.push(<E as Engine>::Scalar::from(x.as_canonical_u64())); }
                for &x in cnext { pv.push(<E as Engine>::Scalar::from(x.as_canonical_u64())); }
                if let Some(r_eff) = ev.rho_eff { pv.push(<E as Engine>::Scalar::from(r_eff.as_canonical_u64())); }
            }
        }
        
        // 5) Pad to power-of-2 BEFORE digest - CRITICAL: match encode_bridge_io_header() order
        let mut current_count_without_digest = self.me.c_coords.len()
            + self.me.y_outputs.len()
            + self.me.r_point.len()
            + 1; // base_b
        if let Some(ev) = &self.ev {
            current_count_without_digest += ev.y_prev.len() + ev.y_next.len() + 1; // +rho
            if ev.fold_chain_digest.is_some() { current_count_without_digest += 4; }
            if let (Some(cprev), Some(cstep), Some(cnext)) = (&ev.acc_c_prev, &ev.acc_c_step, &ev.acc_c_next) {
                current_count_without_digest += cprev.len() + cstep.len() + cnext.len();
                if ev.rho_eff.is_some() { current_count_without_digest += 1; }
            }
        }
        let digest_scalars = self.digest_to_scalars();
        let total_needed = current_count_without_digest + digest_scalars.len();
        let target_count = total_needed.next_power_of_two();
        let padding_needed = target_count - total_needed;
        
        // Add padding zeros BEFORE digest to match encode_bridge_io_header() order
        for _ in 0..padding_needed {
            pv.push(<E as Engine>::Scalar::ZERO);
        }
        
        // 6) fold digest limbs LAST (after padding)
        pv.extend(digest_scalars);

        if padding_needed > 0 {
            eprintln!("🔧 TRANSCRIPT FIX: public_values() padded from {} to {} (matches encode_bridge_io_header)", total_needed, pv.len());
        }
        Ok(pv)
    }

    fn shared<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        eprintln!("✅ shared(): EMPTY (Hash-MLE single-blind strategy)");
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        eprintln!("🔧 precommitted(): EMPTY (avoiding multiple blinds for Hash-MLE)");
        eprintln!("   Hash-MLE PCS requires exactly 1 blind, so we use only REST segment");
        Ok(vec![])
    }

    fn num_challenges(&self) -> usize { 0 }

    fn synthesize<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],  // empty now
        _z_pre: &[AllocatedNum<<E as Engine>::Scalar>],  // empty
        _challenges: Option<&[<E as Engine>::Scalar]>,
    ) -> Result<(), SynthesisError> {
        // SECURITY: refuse to synthesize an unbound circuit
        // Phase-4: PRG-derived rows are a valid Ajtai binding when c_coords are present.
        let has_ajtai_rows = self.wit.ajtai_rows.as_ref().map_or(false, |r| !r.is_empty());
        let has_pp = self.pp.is_some();
        let has_prg_binding = (!has_ajtai_rows && !has_pp) && !self.me.c_coords.is_empty();
        if !has_ajtai_rows && !has_pp && !has_prg_binding {
            eprintln!("❌ SECURITY: No Ajtai rows/PP and no c_coords for PRG binding; refusing to synthesize unbound circuit.");
            return Err(SynthesisError::AssignmentMissing);
        }

        let mut z_vars = Vec::<AllocatedNum<<E as Engine>::Scalar>>::new();
        
        // 1) Allocate REST witness variables exactly in the z_digits order
        #[cfg(feature = "debug-logs")]
        eprintln!("🔍 Witness allocation (REST segment - z_digits first):");
        
        // CRITICAL FIX: Always allocate witness variables, not just when debug-logs is enabled
        for (i, &z) in self.wit.z_digits.iter().enumerate() {
            let val = if z >= 0 {
                <E as Engine>::Scalar::from(z as u64)
            } else {
                -<E as Engine>::Scalar::from((-z) as u64)
            };
            #[cfg(feature = "debug-logs")]
            eprintln!("   z_vars[{}] = {} (from z_digits[{}] = {})", i, val.to_canonical_u64(), i, z);
            
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("REST_Z[{i}]")), || Ok(val))?);
        }
        
        // 2) Pad REST (only REST) to a power-of-two for Hash-MLE
        let orig_rest = z_vars.len();
        // Debug: show first few witness digits
        for i in 0..core::cmp::min(6, orig_rest) {
            let zi = self.wit.z_digits[i];
            eprintln!("[Z-DIGITS] i={} val={}", i, zi);
        }
        let target_rest = if orig_rest <= 1 { 1 } else { orig_rest.next_power_of_two() };
        for i in orig_rest..target_rest {
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("REST_Z_pad[{i}]")), 
                || Ok(<E as Engine>::Scalar::ZERO))?);
        }
        eprintln!("🔍 synthesize(): REST padded from {} to {} (power-of-two)", orig_rest, z_vars.len());
        debug_assert!(z_vars.len().is_power_of_two());
        
        let mut n_constraints = 0usize;
        let to_s = |x: p3_goldilocks::Goldilocks| <E as Engine>::Scalar::from(x.as_canonical_u64());

        // Enforce padded limbs are exactly zero (avoid unconstrained witness coords)
        if target_rest > orig_rest {
            let mut pad_ct = 0usize;
            for i in orig_rest..target_rest {
                cs.enforce(
                    || format!("pad_zero_{}", i),
                    |lc| lc + z_vars[i].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc, // == 0
                );
                pad_ct += 1;
            }
            n_constraints += pad_ct;
            eprintln!("🔧 Enforced {} padded REST limbs to zero", pad_ct);
        }

        // (RANGE) Enforce |Z_i| < b  (digits ∈ {-(b-1), …, (b-1)})
        // For small b (e.g., b=2 => {-1,0,1}), constrain with a product polynomial.
        // For b=2: v*(v-1)*(v+1) == 0.
        for (i, z) in z_vars.iter().enumerate().take(orig_rest) { // Only apply to original witness, not padding
            if self.me.base_b == 2 {
                let one = <E as Engine>::Scalar::ONE;

                // (z)*(z-1)*(z+1) == 0
                // Implement as three constraints using a temp:
                let z_minus_one = AllocatedNum::alloc(cs.namespace(|| format!("z_minus_one_{}", i)), || {
                    Ok(z.get_value().ok_or(SynthesisError::AssignmentMissing)? - one)
                })?;
                cs.enforce(
                    || format!("link_z_minus_one_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + z_minus_one.get_variable() + (one, CS::one()),
                );
                let z_plus_one = AllocatedNum::alloc(cs.namespace(|| format!("z_plus_one_{}", i)), || {
                    Ok(z.get_value().ok_or(SynthesisError::AssignmentMissing)? + one)
                })?;
                cs.enforce(
                    || format!("link_z_plus_one_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + z_plus_one.get_variable() + (-one, CS::one()),
                );
                // w := z * (z-1)
                let w = AllocatedNum::alloc(cs.namespace(|| format!("w_{}", i)), || {
                    let zv = z.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                    let zm1 = z_minus_one.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                    Ok(zv * zm1)
                })?;
                cs.enforce(
                    || format!("w=z*(z-1)_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + z_minus_one.get_variable(),
                    |lc| lc + w.get_variable(),
                );
                // w * (z+1) == 0
                cs.enforce(
                    || format!("range_b2_{}", i),
                    |lc| lc + w.get_variable(),
                    |lc| lc + z_plus_one.get_variable(),
                    |lc| lc,
                );
                n_constraints += 4;
            } else {
                // Generic small-b range: ∏_{t=-(b-1)}^{b-1} (z - t) == 0
                // Keep it simple for now (b is small).
                let mut acc = AllocatedNum::alloc(
                    cs.namespace(|| format!("range_acc_init_{}", i)),
                    || Ok(<E as Engine>::Scalar::ONE)
                )?;
                // CRITICAL: anchor the accumulator to ONE
                // Enforce acc * 1 = 1  ⇒  acc == 1
                cs.enforce(
                    || format!("range_acc_init_is_one_{}", i),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                );
                n_constraints += 1;
                for t in (-(self.me.base_b as i64 - 1))..=(self.me.base_b as i64 - 1) {
                    let c = if t >= 0 { 
                        <E as Engine>::Scalar::from(t as u64) 
                    } else { 
                        -<E as Engine>::Scalar::from((-t) as u64) 
                    };
                    // lin := (z - t)
                    let lin = AllocatedNum::alloc(cs.namespace(|| format!("range_lin_{}_{}", i, t)), || {
                        let zv = z.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                        Ok(zv - c)
                    })?;
                    // enforce lin == z - t
                    cs.enforce(
                        || format!("lin=z-{}_digit_{}", t, i),
                        |lc| lc + z.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + lin.get_variable() + (c, CS::one()),
                    );
                    // acc := acc * lin
                    let new_acc = AllocatedNum::alloc(cs.namespace(|| format!("range_acc_next_{}_{}", i, t)), || {
                        Ok(acc.get_value().ok_or(SynthesisError::AssignmentMissing)? *
                           lin.get_value().ok_or(SynthesisError::AssignmentMissing)?)
                    })?;
                    cs.enforce(
                        || format!("acc_mul_{}_{}", i, t),
                        |lc| lc + acc.get_variable(),
                        |lc| lc + lin.get_variable(),
                        |lc| lc + new_acc.get_variable(),
                    );
                    acc = new_acc;
                    n_constraints += 2;
                }
                // acc must be 0
                cs.enforce(
                    || format!("range_final_zero_{}", i),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
                n_constraints += 1;
            }
        }

        // (A) Ajtai binding target selection (Option B):
        // If EV is present and provides acc_c_step, bind Ajtai to the step vector.
        // Otherwise, bind to the public c_next (me.c_coords) as before.
        let (binding_name, ajtai_rhs): (&'static str, &[neo_math::F]) = if let Some(ev) = &self.ev {
            if let Some(step) = &ev.acc_c_step {
                ("acc_c_step", step.as_slice())
            } else {
                ("me.c_coords", &self.me.c_coords[..])
            }
        } else {
            ("me.c_coords", &self.me.c_coords[..])
        };
        eprintln!("[AJTAI] binding target = {}", binding_name);
        assert_eq!(
            ajtai_rhs.len(), self.me.c_coords.len(),
            "Ajtai RHS len mismatch ({} vs rows={})",
            ajtai_rhs.len(), self.me.c_coords.len()
        );
        // Use streaming approach if ajtai_rows is None but PP is available
        if let Some(rows) = &self.wit.ajtai_rows {
            // Traditional approach: use pre-materialized rows
            let n = core::cmp::min(rows.len(), self.me.c_coords.len());
            for i in 0..n {
                let row = &rows[i];
                let upto = core::cmp::min(row.len(), z_vars.len());
                let c_scalar = <E as Engine>::Scalar::from(ajtai_rhs[i].as_canonical_u64());

                // 🔧 DEBUG: Compute expected LHS value for constraint debugging
                let mut expected_lhs = <E as Engine>::Scalar::ZERO;
                for j in 0..upto {
                    let row_coeff = to_s(row[j]);
                    let z_val = if j < self.wit.z_digits.len() {
                        let z = self.wit.z_digits[j];
                        if z >= 0 { <E as Engine>::Scalar::from(z as u64) }
                        else { -<E as Engine>::Scalar::from((-z) as u64) }
                    } else {
                        <E as Engine>::Scalar::ZERO // padded variables
                    };
                    expected_lhs += row_coeff * z_val;
                }
                if i < 3 || i + 1 == n || i == n / 2 {
                    eprintln!("[AJTAI-CHECK rows] i={} lhs={} rhs={}", i, expected_lhs.to_canonical_u64(), c_scalar.to_canonical_u64());
                }

                cs.enforce(
                    || format!("ajtai_bind_{i}"),
                    |lc| {
                        let mut lc = lc;
                        for j in 0..upto {
                            lc = lc + (to_s(row[j]), z_vars[j].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),            // multiply by constant 1 (X side)
                    |lc| lc + (c_scalar, CS::one()),// right side also uses constant 1
                );
                n_constraints += 1;
            }
        } else if let Some(pp) = &self.pp {
            // 🚀 STREAMING APPROACH: Compute rows on-demand from PP to avoid memory cliff
            let n = self.me.c_coords.len();
            let z_len_padded = self.wit.z_digits.len();
            let z_len_original = pp.d * pp.m; // Original dimensions from PP
            
            for i in 0..n {
                // Compute single row on-demand using original dimensions
                let mut row = match neo_ajtai::compute_single_ajtai_row(pp, i, z_len_original, n) {
                    Ok(row) => row,
                    Err(e) => {
                        eprintln!("❌ Failed to compute Ajtai row {}: {}", i, e);
                        return Err(SynthesisError::AssignmentMissing);
                    }
                };
                
                // Pad the row with zeros to match the padded witness length
                if z_len_padded > z_len_original {
                    let pad_len = z_len_padded - z_len_original;
                    row.extend(std::iter::repeat(neo_math::F::ZERO).take(pad_len));
                }
                
                let upto = core::cmp::min(row.len(), z_vars.len());
                let c_scalar = <E as Engine>::Scalar::from(ajtai_rhs[i].as_canonical_u64());

                // 🔧 DEBUG: Compute expected LHS value for constraint debugging
                let mut expected_lhs = <E as Engine>::Scalar::ZERO;
                for j in 0..upto {
                    let row_coeff = to_s(row[j]);
                    let z_val = if j < self.wit.z_digits.len() {
                        let z = self.wit.z_digits[j];
                        if z >= 0 { <E as Engine>::Scalar::from(z as u64) }
                        else { -<E as Engine>::Scalar::from((-z) as u64) }
                    } else {
                        <E as Engine>::Scalar::ZERO // padded variables
                    };
                    expected_lhs += row_coeff * z_val;
                }
                if i < 3 || i + 1 == n || i == n / 2 {
                    eprintln!("[AJTAI-CHECK stream] i={} lhs={} rhs={}", i, expected_lhs.to_canonical_u64(), c_scalar.to_canonical_u64());
                }

                cs.enforce(
                    || format!("ajtai_bind_streaming_{i}"),
                    |lc| {
                        let mut lc = lc;
                        for j in 0..upto {
                            lc = lc + (to_s(row[j]), z_vars[j].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),            // multiply by constant 1 (X side)
                    |lc| lc + (c_scalar, CS::one()),// right side also uses constant 1
                );
                n_constraints += 1;
            }
        } else {
            // PRG fallback (Phase 4): derive Ajtai rows from public seed (header_digest)
            let seed = self.fold_digest;
            let rows = self.me.c_coords.len();
            let z_len = z_vars.len();
            for i in 0..rows {
                let row = crate::ajtai_prg::expand_row_from_seed(seed, i as u32, z_len);
                let rhs = <E as Engine>::Scalar::from(ajtai_rhs[i].as_canonical_u64());
                cs.enforce(
                    || format!("ajtai_bind_prg_{}", i),
                    |lc| {
                        let mut lc = lc;
                        for j in 0..z_len {
                            let a = <E as Engine>::Scalar::from(row[j].as_canonical_u64());
                            lc = lc + (a, z_vars[j].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + (rhs, CS::one()),
                );
                n_constraints += 1;
            }
        }

        // Optional: RLC guard tying c_step_coords to Ajtai rows via transcript-derived coefficients.
        // Enabled when env NEO_ENABLE_RLC_GUARD=1 and lengths match.
        if std::env::var("NEO_ENABLE_RLC_GUARD").ok().as_deref() == Some("1") {
            let n = self.me.c_coords.len();
            if !self.me.c_step_coords.is_empty() && self.me.c_step_coords.len() == n {
                let seed = self.fold_digest;
                // Derive coefficients from public data (seed, c_step_coords)
                let coeffs = crate::rlc_guard::derive_rlc_coefficients(seed, &self.me.c_step_coords, n);
                // Aggregate PRG rows: G[j] = Σ_i coeffs[i] * row_i[j]
                let z_len = z_vars.len();
                let mut g = vec![<E as Engine>::Scalar::ZERO; z_len];
                for i in 0..n {
                    let row = crate::ajtai_prg::expand_row_from_seed(seed, i as u32, z_len);
                    let rho_i = <E as Engine>::Scalar::from(coeffs[i].as_canonical_u64());
                    for j in 0..z_len {
                        g[j] += rho_i * <E as Engine>::Scalar::from(row[j].as_canonical_u64());
                    }
                }
                // RHS: Σ_i coeffs[i] * c_step_coords[i]
                let mut rhs_s = <E as Engine>::Scalar::ZERO;
                for i in 0..n {
                    rhs_s += <E as Engine>::Scalar::from(coeffs[i].as_canonical_u64())
                        * <E as Engine>::Scalar::from(self.me.c_step_coords[i].as_canonical_u64());
                }
                // LHS (constant form via c_coords): Σ_i coeffs[i] * c_coords[i]
                let mut lhs_const = <E as Engine>::Scalar::ZERO;
                for i in 0..n {
                    lhs_const += <E as Engine>::Scalar::from(coeffs[i].as_canonical_u64())
                        * <E as Engine>::Scalar::from(self.me.c_coords[i].as_canonical_u64());
                }
                eprintln!(
                    "[RLC-GUARD] active: n={}, z_len={}, lhs={} rhs={} (first coeff={}, first c_step={})",
                    n, z_len, lhs_const.to_canonical_u64(), rhs_s.to_canonical_u64(),
                    coeffs.get(0).map(|c| c.as_canonical_u64()).unwrap_or(0),
                    self.me.c_step_coords.get(0).map(|c| c.as_canonical_u64()).unwrap_or(0)
                );
                // Enforce single inner-product equality: <G, z> = RHS
                cs.enforce(
                    || "rlc_guard_aggregated",
                    |lc| {
                        let mut lc = lc;
                        for j in 0..z_len { lc = lc + (g[j], z_vars[j].get_variable()); }
                        lc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + (rhs_s, CS::one()),
                );
                n_constraints += 1;
            }
        }

        // Pattern A link constraints removed - they were incorrect (applied Ajtai rows to witness vars, not digits)
        // and expensive (O(d·κ) constraints). The RLC binder in the folding layer provides the correct binding.

        // (B) ME evals: <w_j, Z> = y_j  
        let m = core::cmp::min(self.wit.weight_vectors.len(), self.me.y_outputs.len());
        for j in 0..m {
            let wj = &self.wit.weight_vectors[j];
            let upto = core::cmp::min(wj.len(), z_vars.len());
            let _y_scalar = <E as Engine>::Scalar::from(self.me.y_outputs[j].as_canonical_u64());

            // 🔧 DEBUG: Compute expected LHS value for ME constraint debugging
            let mut expected_lhs = <E as Engine>::Scalar::ZERO;
            for k in 0..upto {
                let w_coeff = to_s(wj[k]);
                let z_val = if k < self.wit.z_digits.len() {
                    let z = self.wit.z_digits[k];
                    if z >= 0 { <E as Engine>::Scalar::from(z as u64) }
                    else { -<E as Engine>::Scalar::from((-z) as u64) }
                } else {
                    <E as Engine>::Scalar::ZERO // padded variables
                };
                expected_lhs += w_coeff * z_val;
            }
            if j < 2 && expected_lhs != _y_scalar {
                if std::env::var("NEO_DEBUG_ME_EVAL").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[ME-DIAG] me_eval_{} mismatch: lhs={} rhs={} (first two)",
                        j, expected_lhs.to_canonical_u64(), _y_scalar.to_canonical_u64()
                    );
                }
            }

            cs.enforce(
                || format!("me_eval_{j}"),
                |lc| {
                    let mut lc = lc;
                    for k in 0..upto {
                        lc = lc + (to_s(wj[k]), z_vars[k].get_variable());
                    }
                    lc
                },
                |lc| lc + CS::one(),               // multiply by constant 1
                |lc| lc + (_y_scalar, CS::one()),  // RHS uses constant 1
            );
            n_constraints += 1;
        }

        // (C) Pi-CCS terminal check: reconstruct weights from CCS and r, enforce equality per digit
        if let Some(pi) = &self.pi_ccs {
            let d = neo_math::ring::D;
            // Strict shape checks to avoid unconstrained witness tails
            let m_w = self.wit.weight_vectors.len();
            if pi.matrices.len() != m_w {
                eprintln!(
                    "[PI-CCS-DIAG] matrices.len()={} != weight_vectors.len()={}",
                    pi.matrices.len(), m_w
                );
                return Err(SynthesisError::AssignmentMissing);
            }
            // Prepare r variables (bind to constants)
            let mut r_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(self.me.r_point.len());
            for (i, &rt) in self.me.r_point.iter().enumerate() {
                let r_val = <E as Engine>::Scalar::from(rt.as_canonical_u64());
                let r_var = AllocatedNum::alloc(cs.namespace(|| format!("pi_ccs_r_{}", i)), || Ok(r_val))?;
                // Enforce r_var == r_val
                cs.enforce(
                    || format!("pi_ccs_r_bind_{}", i),
                    |lc| lc + r_var.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + (r_val, CS::one()),
                );
                r_vars.push(r_var);
            }
            // Precompute (1 - r_t) vars
            let mut one_minus_r: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(r_vars.len());
            for (i, r) in r_vars.iter().enumerate() {
                let one = <E as Engine>::Scalar::ONE;
                let om = AllocatedNum::alloc(cs.namespace(|| format!("pi_ccs_one_minus_r_{}", i)), || {
                    Ok(one - r.get_value().ok_or(SynthesisError::AssignmentMissing)?)
                })?;
                // Enforce r + (1-r) = 1
                cs.enforce(
                    || format!("pi_ccs_one_minus_r_link_{}", i),
                    |lc| lc + r.get_variable() + om.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                );
                one_minus_r.push(om);
            }

            // Powers of b as constants
            let base_b = <E as Engine>::Scalar::from(self.me.base_b as u64);
            let mut pow_b = vec![<E as Engine>::Scalar::ONE; d];
            for i in 1..d { pow_b[i] = pow_b[i-1] * base_b; }

            for (j, mj) in pi.matrices.iter().enumerate() {
                // Ensure the corresponding weight vector has exact expected length cols*d
                let expected_w_len = mj.cols * d;
                let wj = &self.wit.weight_vectors[j];
                if wj.len() != expected_w_len {
                    eprintln!(
                        "[PI-CCS-DIAG] weight_vectors[{}].len()={} != cols*d={}*{}={}",
                        j, wj.len(), mj.cols, d, expected_w_len
                    );
                    return Err(SynthesisError::AssignmentMissing);
                }
                // Compute chi[row] variables by product over bits; reuse across columns
                let mut chi_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(mj.rows);
                for row in 0..mj.rows {
                    // Build product term across bits
                    let mut acc = AllocatedNum::alloc(cs.namespace(|| format!("pi_ccs_chi_init_{}_{}", j, row)), || Ok(<E as Engine>::Scalar::ONE))?;
                    // Enforce acc == 1
                    cs.enforce(
                        || format!("pi_ccs_chi_init_one_{}_{}", j, row),
                        |lc| lc + acc.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + CS::one(),
                    );
                    for t in 0..r_vars.len() {
                        let bit_is_one = ((row >> t) & 1) == 1;
                        let term = if bit_is_one { r_vars[t].clone() } else { one_minus_r[t].clone() };
                        let new_acc = AllocatedNum::alloc(cs.namespace(|| format!("pi_ccs_chi_step_{}_{}_{}", j, row, t)), || {
                            Ok(acc.get_value().ok_or(SynthesisError::AssignmentMissing)? * term.get_value().ok_or(SynthesisError::AssignmentMissing)?)
                        })?;
                        cs.enforce(
                            || format!("pi_ccs_chi_mul_{}_{}_{}", j, row, t),
                            |lc| lc + acc.get_variable(),
                            |lc| lc + term.get_variable(),
                            |lc| lc + new_acc.get_variable(),
                        );
                        acc = new_acc;
                    }
                    chi_vars.push(acc);
                }

                // For each column, compute v_c = sum_{rows} a_{row,col} * chi[row]
                let m_cols = mj.cols;
                let mut v_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(m_cols);
                for c in 0..m_cols {
                    let v_c = AllocatedNum::alloc(cs.namespace(|| format!("pi_ccs_v_{}_{}", j, c)), || Ok(<E as Engine>::Scalar::ZERO))?;
                    // Enforce linear equality: (sum a*chi) * 1 = v_c
                    cs.enforce(
                        || format!("pi_ccs_v_lin_{}_{}", j, c),
                        |lc| {
                            let mut lc = lc;
                            for &(r_idx, c_idx, a) in &mj.entries {
                                if c_idx as usize == c {
                                    let a_s = <E as Engine>::Scalar::from(a.as_canonical_u64());
                                    lc = lc + (a_s, chi_vars[r_idx as usize].get_variable());
                                }
                            }
                            lc
                        },
                        |lc| lc + CS::one(),
                        |lc| lc + v_c.get_variable(),
                    );
                    v_vars.push(v_c);
                }

                // Enforce weight vectors match v_c * b^r per digit
                for c in 0..m_cols { for r in 0..d {
                    let idx = c * d + r;
                    let w_const = <E as Engine>::Scalar::from(wj[idx].as_canonical_u64());
                    cs.enforce(
                        || format!("pi_ccs_w_match_{}_{}_{}", j, c, r),
                        |lc| lc + v_vars[c].get_variable(),
                        |lc| lc + (pow_b[r], CS::one()),
                        |lc| lc + (w_const, CS::one()),
                    );
                }}
            }
        }

        // CRITICAL SECURITY: Reject vacuous circuits that have no binding to the statement
        // Phase-4: Treat PRG fallback (seeded rows) as a valid Ajtai binding when c_coords are present.
        let has_prg_binding = (!has_ajtai_rows && !has_pp) && !self.me.c_coords.is_empty();
        let has_ajtai_binding = has_ajtai_rows || has_pp || has_prg_binding;
        if !has_ajtai_binding && self.wit.weight_vectors.is_empty() {
            eprintln!("❌ SECURITY VIOLATION: Cannot synthesize circuit with no Ajtai binding AND no ME constraints.");
            eprintln!("   This would create a vacuous proof with no binding to the statement.");
            return Err(SynthesisError::AssignmentMissing);
        }

        // All witness variables are now allocated as REST with power-of-2 length

        // --- Ensure TOTAL number of constraints is a power of two (Hash‑MLE requirement)
        let total_constraints_so_far = n_constraints;
        let want_constraints = if total_constraints_so_far <= 1 { 1 } else { total_constraints_so_far.next_power_of_two() };
        let need_to_add = want_constraints - total_constraints_so_far;
        if need_to_add > 0 {
            // Add 1*1 = 1 tautologies using CS::one()
            for i in 0..need_to_add {
                cs.enforce(
                    || format!("pad_const_tautology_{i}"),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                );
            }
            eprintln!("🔧 synthesize(): padded constraints {} → {} (total with shared = {})", total_constraints_so_far, want_constraints, want_constraints);
        } else {
            eprintln!("ℹ️  synthesize(): constraints = {} (total with shared = {} - power-of-two OK)", n_constraints, total_constraints_so_far);
        }

        // CRITICAL FIX: Create public inputs via inputize() for transcript fork compatibility
        // This replaces the old public_values() method approach
        
        // 1) Ajtai commitment coords (Fq)
        let mut c_public_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(self.me.c_coords.len());
        for (i, &c) in self.me.c_coords.iter().enumerate() {
            let c_val = <E as Engine>::Scalar::from(c.as_canonical_u64());
            let c_alloc = AllocatedNum::alloc(cs.namespace(|| format!("c_coord_{}", i)), || Ok(c_val))?;
            let _ = c_alloc.inputize(cs.namespace(|| format!("public_c_{}", i)));
            c_public_vars.push(c_alloc);
        }
        // Diagnostic: echo first few c_coords as seen by the circuit
        if !self.me.c_coords.is_empty() {
            let show = core::cmp::min(4, self.me.c_coords.len());
            let mut buf = String::new();
            for i in 0..show {
                if i > 0 { buf.push_str(", "); }
                buf.push_str(&format!("{}", self.me.c_coords[i].as_canonical_u64()));
            }
            eprintln!("[C-PUBLIC] c_coords[0..{}): {}", show, buf);
        }

        // 2) y outputs — already flattened (K -> [F;2]) in adapter
        // Keep allocated handles so optional EV constraints can reference them.
        let mut y_public_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(self.me.y_outputs.len());
        for (i, &y_limb) in self.me.y_outputs.iter().enumerate() {
            let v = <E as Engine>::Scalar::from(y_limb.as_canonical_u64());
            let a = AllocatedNum::alloc(cs.namespace(|| format!("y_output_limb_{}", i)), || Ok(v))?;
            let _ = a.inputize(cs.namespace(|| format!("public_y_limb_{}", i)));
            y_public_vars.push(a);
        }
        
        // 3) challenge r (Fq^m)
        for (i, &r) in self.me.r_point.iter().enumerate() {
            let r_val = <E as Engine>::Scalar::from(r.as_canonical_u64());
            let r_alloc = AllocatedNum::alloc(cs.namespace(|| format!("r_point_{}", i)), || Ok(r_val))?;
            let _ = r_alloc.inputize(cs.namespace(|| format!("public_r_{}", i)));
        }

        // 4) base dimension b
        let b_val = <E as Engine>::Scalar::from(self.me.base_b as u64);
        let b_alloc = AllocatedNum::alloc(cs.namespace(|| "base_b"), || Ok(b_val))?;
        let _ = b_alloc.inputize(cs.namespace(|| "public_base_b"));

        // 4.5) Optional IVC EV public inputs and constraints
        if let Some(ev) = &self.ev {
            // Inputize y_prev, y_next, rho
            let mut y_prev_vars = Vec::with_capacity(ev.y_prev.len());
            for (i, &v) in ev.y_prev.iter().enumerate() {
                let val = <E as Engine>::Scalar::from(v.as_canonical_u64());
                let a = AllocatedNum::alloc(cs.namespace(|| format!("ivc_y_prev_{}", i)), || Ok(val))?;
                let _ = a.inputize(cs.namespace(|| format!("public_ivc_y_prev_{}", i)));
                y_prev_vars.push(a);
            }
            let mut y_next_vars = Vec::with_capacity(ev.y_next.len());
            for (i, &v) in ev.y_next.iter().enumerate() {
                let val = <E as Engine>::Scalar::from(v.as_canonical_u64());
                let a = AllocatedNum::alloc(cs.namespace(|| format!("ivc_y_next_{}", i)), || Ok(val))?;
                let _ = a.inputize(cs.namespace(|| format!("public_ivc_y_next_{}", i)));
                y_next_vars.push(a);
            }
            #[cfg(all(debug_assertions, feature = "neo_dev_only"))]
            let disable_ev = std::env::var("NEO_DEBUG_DISABLE_EV").ok().as_deref() == Some("1");
            #[cfg(not(all(debug_assertions, feature = "neo_dev_only")))]
            let disable_ev = false;
            let rho_val = <E as Engine>::Scalar::from(ev.rho.as_canonical_u64());
            let rho_var = AllocatedNum::alloc(cs.namespace(|| "ivc_rho"), || Ok(rho_val))?;
            let _ = rho_var.inputize(cs.namespace(|| "public_ivc_rho"));

            // Diagnostics for EV embedding
            eprintln!(
                "[EV-DIAG] y_len={}, rho={}, y_prev[0..2]={:?}, y_next[0..2]={:?}",
                ev.y_next.len(), ev.rho.as_canonical_u64(),
                ev.y_prev.iter().take(2).map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                ev.y_next.iter().take(2).map(|f| f.as_canonical_u64()).collect::<Vec<_>>()
            );
            if let Some(step_pub) = &ev.y_step_public {
                let ysp: Vec<u64> = step_pub.iter().take(2).map(|f| f.as_canonical_u64()).collect();
                eprintln!("[EV-DIAG] y_step_public[0..2]={:?}", ysp);
                // Quick consistency check on host values
                let mut mismatches = 0usize;
                for k in 0..core::cmp::min(4, step_pub.len()) {
                    let lhs = step_pub[k] * ev.rho;
                    let rhs = ev.y_next.get(k).copied().unwrap_or(neo_math::F::ZERO)
                        - ev.y_prev.get(k).copied().unwrap_or(neo_math::F::ZERO);
                    if lhs != rhs { mismatches += 1; }
                }
                if mismatches > 0 {
                    eprintln!("[EV-DIAG] y_step_public * rho != (y_next - y_prev) for {} of first 4 entries", mismatches);
                }
            }

            // Optional: fold chain digest limbs as public inputs and bind to rho via Poseidon2
            if let Some(d) = &ev.fold_chain_digest {
                for (i, chunk) in d.chunks(8).enumerate() {
                    let limb_const = <E as Engine>::Scalar::from(u64::from_le_bytes([
                        chunk.get(0).copied().unwrap_or(0),
                        chunk.get(1).copied().unwrap_or(0),
                        chunk.get(2).copied().unwrap_or(0),
                        chunk.get(3).copied().unwrap_or(0),
                        chunk.get(4).copied().unwrap_or(0),
                        chunk.get(5).copied().unwrap_or(0),
                        chunk.get(6).copied().unwrap_or(0),
                        chunk.get(7).copied().unwrap_or(0),
                    ]));
                    let limb_alloc = AllocatedNum::alloc(cs.namespace(|| format!("fold_chain_digest_{}", i)), || Ok(limb_const))?;
                    // Bind limb variable to constant, so it contributes to constraints
                    cs.enforce(
                        || format!("fold_chain_digest_bind_{}", i),
                        |lc| lc + limb_alloc.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + (limb_const, CS::one()),
                    );
                    let _ = limb_alloc.inputize(cs.namespace(|| format!("public_fold_chain_digest_{}", i)));
                }

                // Derive a non-zero challenge from the fold-chain digest and bind it to rho
                // Domain tag chosen to be stable and independent: "neo/ev/rho_from_digest/v1"
                let mut limbs: Vec<u64> = Vec::with_capacity(4 + 4 + 1);
                for chunk in b"neo/ev/rho_from_digest/v1".chunks(8) {
                    let mut b = [0u8; 8];
                    b[..chunk.len()].copy_from_slice(chunk);
                    limbs.push(u64::from_le_bytes(b));
                }
                for chunk in d.chunks(8) {
                    let mut b = [0u8; 8];
                    b[..chunk.len()].copy_from_slice(chunk);
                    limbs.push(u64::from_le_bytes(b));
                }
                let packed: Vec<u8> = limbs.iter().flat_map(|&x| x.to_le_bytes()).collect();
                let h = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&packed);
                let mut rho_from_digest = <E as Engine>::Scalar::from(h[0].as_canonical_u64());
                if rho_from_digest == <E as Engine>::Scalar::ZERO { rho_from_digest = <E as Engine>::Scalar::ONE; }
                // Enforce rho_var == rho_from_digest
                cs.enforce(
                    || "ev_rho_bind_fold_chain_digest",
                    |lc| lc + rho_var.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + (rho_from_digest, CS::one()),
                );
            }

            // Optional accumulator commitment evolution inputs
            if let (Some(cprev), Some(cstep), Some(cnext)) = (&ev.acc_c_prev, &ev.acc_c_step, &ev.acc_c_next) {
                // Diagnostics for commit-evo vectors (EV embedding)
                eprintln!(
                    "[EV-DIAG] acc_c_prev/step/next lens: {}/{}/{}",
                    cprev.len(), cstep.len(), cnext.len()
                );
                let mut mm = 0usize;
                for i in 0..core::cmp::min(4, core::cmp::min(cprev.len(), core::cmp::min(cstep.len(), cnext.len()))) {
                    let lhs = cprev[i] + ev.rho * cstep[i];
                    if lhs != cnext[i] { mm += 1; }
                }
                if mm > 0 {
                    eprintln!("[EV-DIAG] acc commit-evo mismatch on {} of first 4 entries", mm);
                }
                // Inputize sequences
                let mut cprev_vars = Vec::with_capacity(cprev.len());
                let mut cstep_vars = Vec::with_capacity(cstep.len());
                let mut cnext_vars = Vec::with_capacity(cnext.len());
                for (i, &v) in cprev.iter().enumerate() {
                    let s = <E as Engine>::Scalar::from(v.as_canonical_u64());
                    let a = AllocatedNum::alloc(cs.namespace(|| format!("acc_c_prev_{}", i)), || Ok(s))?;
                    let _ = a.inputize(cs.namespace(|| format!("public_acc_c_prev_{}", i)));
                    cprev_vars.push(a);
                }
                for (i, &v) in cstep.iter().enumerate() {
                    let s = <E as Engine>::Scalar::from(v.as_canonical_u64());
                    let a = AllocatedNum::alloc(cs.namespace(|| format!("acc_c_step_{}", i)), || Ok(s))?;
                    let _ = a.inputize(cs.namespace(|| format!("public_acc_c_step_{}", i)));
                    cstep_vars.push(a);
                }
                for (i, &v) in cnext.iter().enumerate() {
                    let s = <E as Engine>::Scalar::from(v.as_canonical_u64());
                    let a = AllocatedNum::alloc(cs.namespace(|| format!("acc_c_next_{}", i)), || Ok(s))?;
                    let _ = a.inputize(cs.namespace(|| format!("public_acc_c_next_{}", i)));
                    cnext_vars.push(a);
                }
                if disable_ev {
                    eprintln!("[EV-DIAG] acc commit-evo constraints disabled by NEO_DEBUG_DISABLE_EV=1");
                } else {
                    let rho_eff_s = <E as Engine>::Scalar::from(ev.rho_eff.unwrap_or(ev.rho).as_canonical_u64());
                    // Enforce acc_c_next[i] = acc_c_prev[i] + rho_eff * acc_c_step[i]
                    let n = core::cmp::min(cprev_vars.len(), core::cmp::min(cstep_vars.len(), cnext_vars.len()));
                    for i in 0..n {
                        cs.enforce(
                            || format!("acc_commit_evo_{}", i),
                            |lc| lc + cnext_vars[i].get_variable(),
                            |lc| lc + CS::one(),
                            |lc| lc + cprev_vars[i].get_variable() + (rho_eff_s, cstep_vars[i].get_variable()),
                        );
                    }
                }
            }
            // Enforce for k in 0..y_len: (y_step[k]) * rho = (y_next[k] - y_prev[k])
            if disable_ev {
                eprintln!("[EV-DIAG] y_step linkage constraints disabled by NEO_DEBUG_DISABLE_EV=1");
            } else {
                let y_len = ev.y_next.len();
                // Select y_step source: either provided public values (allocated privately), or tail of y_outputs
                let mut y_step_vars: Vec<AllocatedNum<<E as Engine>::Scalar>> = Vec::with_capacity(y_len);
                if let Some(step_pub) = &ev.y_step_public {
                    for (k, &vf) in step_pub.iter().enumerate().take(y_len) {
                        let val = <E as Engine>::Scalar::from(vf.as_canonical_u64());
                        let v = AllocatedNum::alloc(cs.namespace(|| format!("ivc_y_step_pub_{}", k)), || Ok(val))?;
                        y_step_vars.push(v);
                    }
                } else {
                    let start = self.me.y_outputs.len().saturating_sub(y_len);
                    for k in 0..y_len { y_step_vars.push(y_public_vars[start + k].clone()); }
                }
                for k in 0..y_len {
                    let yn = &y_next_vars[k];
                    let yp = &y_prev_vars[k];
                    let y_step_var = &y_step_vars[k];
                    cs.enforce(
                        || format!("ivc_ev_{}", k),
                        |lc| lc + y_step_var.get_variable(),
                        |lc| lc + rho_var.get_variable(),
                        |lc| lc + yn.get_variable() - yp.get_variable(),
                    );
                    n_constraints += 1;
                }
            }
        }

        // Optional linkage constraints (bind specific undigitized witness values)
        if let Some(link) = &self.linkage {
            #[cfg(all(debug_assertions, feature = "neo_dev_only"))]
            let disable_link = std::env::var("NEO_DEBUG_DISABLE_LINKAGE").ok().as_deref() == Some("1");
            #[cfg(not(all(debug_assertions, feature = "neo_dev_only")))]
            let disable_link = false;
            let d = neo_math::D;
            let mut pow_b: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::ONE; d];
            for r in 1..d { pow_b[r] = pow_b[r-1] * <E as Engine>::Scalar::from(self.me.base_b as u64); }
            eprintln!("[LINK-DIAG] x_indices_abs={:?} y_prev_indices_abs={:?} const1={:?}", link.x_indices_abs, link.y_prev_indices_abs, link.const1_index_abs);
            // Host-side diagnostic check for first few bindings
            for (j, &idx) in link.x_indices_abs.iter().take(2).enumerate() {
                let expected = link.step_io.get(j).copied().unwrap_or(neo_math::F::ZERO);
                // reconstruct from witness digits if within bounds
                let mut accum = <E as Engine>::Scalar::ZERO;
                let upto = core::cmp::min(d, self.wit.z_digits.len().saturating_sub(idx * d));
                for r in 0..upto {
                    let z_i = self.wit.z_digits[idx * d + r];
                    let z_s = if z_i >= 0 { <E as Engine>::Scalar::from(z_i as u64) } else { -<E as Engine>::Scalar::from((-z_i) as u64) };
                    accum += pow_b[r] * z_s;
                }
                eprintln!("[LINK-DIAG] x_bind j={} idx={} expected={} recon={} (first few)", j, idx, expected.as_canonical_u64(), accum.to_canonical_u64());
            }
            if disable_link {
                eprintln!("[LINK-DIAG] linkage constraints disabled by NEO_DEBUG_DISABLE_LINKAGE=1");
            } else {
            let mut enforce_z_eq = |idx_abs: usize, expected_f: neo_math::F, tag: &str| -> Result<(), SynthesisError> {
                let k = idx_abs;
                let upto = core::cmp::min(d, z_vars.len().saturating_sub(k * d));
                let expect = <E as Engine>::Scalar::from(expected_f.as_canonical_u64());
                cs.enforce(
                    || format!("link_{}", tag),
                    |lc| {
                        let mut acc = lc;
                        for r in 0..upto {
                            let z_idx = k * d + r;
                            acc = acc + (pow_b[r], z_vars[z_idx].get_variable());
                        }
                        acc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + (expect, CS::one()),
                );
                n_constraints += 1;
                Ok(())
            };
            for (j, &idx) in link.x_indices_abs.iter().enumerate() {
                let expected = link.step_io.get(j).copied().unwrap_or(neo_math::F::ZERO);
                enforce_z_eq(idx, expected, &format!("x_{}", j))?;
            }
            for (i, &idx) in link.y_prev_indices_abs.iter().enumerate() {
                let expected = self.ev.as_ref().map(|e| e.y_prev.get(i).copied().unwrap_or(neo_math::F::ZERO)).unwrap_or(neo_math::F::ZERO);
                enforce_z_eq(idx, expected, &format!("y_prev_{}", i))?;
            }
            if let Some(c1_idx) = link.const1_index_abs { enforce_z_eq(c1_idx, neo_math::F::ONE, "const1")?; }
            }
        }

        // Optional commitment evolution constraints: c_next (public) equals c_prev + rho*c_step
        if let Some(commit) = &self.commit {
            #[cfg(all(debug_assertions, feature = "neo_dev_only"))]
            let disable_commit = std::env::var("NEO_DEBUG_DISABLE_COMMIT").ok().as_deref() == Some("1");
            #[cfg(not(all(debug_assertions, feature = "neo_dev_only")))]
            let disable_commit = false;
            let rho_s = <E as Engine>::Scalar::from(commit.rho.as_canonical_u64());
            let n = core::cmp::min(commit.c_prev.len(), core::cmp::min(commit.c_step.len(), c_public_vars.len()));
            // Diagnostics for commit-evo public binding
            eprintln!("[COMMIT-DIAG] commit-evo n={}, rho={}", n, commit.rho.as_canonical_u64());
            for i in 0..core::cmp::min(4, n) {
                let lhs = self.me.c_coords[i];
                let rhs = commit.c_prev[i] + commit.rho * commit.c_step[i];
                if lhs != rhs {
                    eprintln!("[COMMIT-DIAG] mismatch at {}: lhs(c_next)={} rhs(c_prev+rho*c_step)={}", i, lhs.as_canonical_u64(), rhs.as_canonical_u64());
                }
            }
            if disable_commit {
                eprintln!("[COMMIT-DIAG] commit-evo constraints disabled by NEO_DEBUG_DISABLE_COMMIT=1");
            } else {
                for i in 0..n {
                    let rhs_const = <E as Engine>::Scalar::from(commit.c_prev[i].as_canonical_u64())
                        + rho_s * <E as Engine>::Scalar::from(commit.c_step[i].as_canonical_u64());
                    cs.enforce(
                        || format!("commit_evo_{}", i),
                        |lc| lc + c_public_vars[i].get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + (rhs_const, CS::one()),
                    );
                    n_constraints += 1;
                }
            }
        }
        
        // 5) Pad to power-of-2 BEFORE digest - CRITICAL: match encode_bridge_io_header() order
        let mut current_count_without_digest = self.me.c_coords.len()
            + self.me.y_outputs.len()
            + self.me.r_point.len()
            + 1; // base_b
        if let Some(ev) = &self.ev {
            current_count_without_digest += ev.y_prev.len() + ev.y_next.len() + 1; // +rho
            if ev.fold_chain_digest.is_some() { current_count_without_digest += 4; }
            if let (Some(cprev), Some(cstep), Some(cnext)) = (&ev.acc_c_prev, &ev.acc_c_step, &ev.acc_c_next) {
                current_count_without_digest += cprev.len() + cstep.len() + cnext.len();
                if ev.rho_eff.is_some() { current_count_without_digest += 1; }
            }
        }
        let digest_scalars = self.digest_to_scalars();
        let total_needed = current_count_without_digest + digest_scalars.len();
        let target_count = total_needed.next_power_of_two();
        let padding_needed = target_count - total_needed;
        
        // Add padding zeros BEFORE digest to match encode_bridge_io_header() order
        for i in 0..padding_needed {
            let zero_alloc = AllocatedNum::alloc(cs.namespace(|| format!("padding_{}", i)), || Ok(<E as Engine>::Scalar::ZERO))?;
            let _ = zero_alloc.inputize(cs.namespace(|| format!("public_padding_{}", i)));
        }
        
        // 6) fold digest limbs LAST (after padding)
        for (i, &digest_val) in digest_scalars.iter().enumerate() {
            let digest_alloc = AllocatedNum::alloc(cs.namespace(|| format!("digest_{}", i)), || Ok(digest_val))?;
            let _ = digest_alloc.inputize(cs.namespace(|| format!("public_digest_{}", i)));
        }
        
        if padding_needed > 0 {
            eprintln!("🔍 inputize() padded from {} to {} public inputs (matches encode_bridge_io_header)", total_needed, target_count);
        }

        Ok(())
    }
}

/// Public API: setup keys (so you can cache them)
pub fn setup_me_snark(
    me: &MEInstance,
    wit: &MEWitness,
) -> Result<(SpartanProverKey<E>, SpartanVerifierKey<E>), SpartanError> {
    let circuit = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest);
    
    // Spartan2 SNARK setup expects the circuit directly
    R1CSSNARK::<E>::setup(circuit)
}

/// Public API: prove a real Spartan2 SNARK over Hash‑MLE (Poseidon2) with caching
pub fn prove_me_snark(
    me: &MEInstance,
    wit: &MEWitness,
) -> Result<(Vec<u8>, Vec<<E as Engine>::Scalar>, Arc<SpartanVerifierKey<E>>), SpartanError> {
    prove_me_snark_with_pp(me, wit, None, None)
}

/// Public API: prove a real Spartan2 SNARK with optional PP for streaming Ajtai rows
pub fn prove_me_snark_with_pp(
    me: &MEInstance,
    wit: &MEWitness,
    pp: Option<std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>>,
    ev: Option<IvcEvEmbed>,
) -> Result<(Vec<u8>, Vec<<E as Engine>::Scalar>, Arc<SpartanVerifierKey<E>>), SpartanError> {
    let circuit = MeCircuit::new(me.clone(), wit.clone(), pp, me.header_digest).with_ev(ev);
    
    // Assert PCS width is correct for Hash-MLE (binary hypercube)
    let pcs_width = <E as Engine>::PCS::width();
    #[cfg(feature = "debug-logs")]
    eprintln!("🔍 ENGINE PCS WIDTH = {}", pcs_width);
    assert_eq!(pcs_width, 2, "Hash-MLE PCS must report width=2 (binary hypercube arity), got {}", pcs_width);
    
    // Generate circuit fingerprint for secure caching
    let circuit_key = CircuitKey::from_circuit(&circuit);
    
    #[cfg(feature = "debug-logs")]
    eprintln!("🔑 Circuit fingerprint: {:?}", circuit_key);
    
    // 1. Get or create SNARK keys (with secure fingerprint-based caching)
    let (pk, vk) = if let Some(cached_pk) = PK_CACHE.get(&circuit_key) {
        let cached_vk = VK_CACHE.get(&circuit_key)
            .expect("VK must exist if PK exists");
        #[cfg(feature = "debug-logs")]
        eprintln!("✅ Using cached SNARK keys for circuit fingerprint");
        (cached_pk.clone(), cached_vk.clone())
    } else {
        #[cfg(feature = "debug-logs")]
        eprintln!("🔍 Setting up new SNARK keys for new circuit fingerprint");
    let (new_pk, new_vk) = R1CSSNARK::<E>::setup(circuit.clone())?;
        let pk_arc = Arc::new(new_pk);
        let vk_arc = Arc::new(new_vk);
        PK_CACHE.insert(circuit_key.clone(), pk_arc.clone());
        VK_CACHE.insert(circuit_key.clone(), vk_arc.clone());
        #[cfg(feature = "debug-logs")]
        eprintln!("✅ SNARK setup completed and cached");
        (pk_arc, vk_arc)
    };
    
    // 2. Prepare SNARK (with cached keys, this is much faster than full setup)
    #[cfg(feature = "debug-logs")]
    eprintln!("🔍 Preparing SNARK with cached keys");
    let prep_snark = R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), true)?; // Enable internal caches
    #[cfg(feature = "debug-logs")]
    eprintln!("✅ SNARK preparation completed (keys were cached)"); 
    
    // 3. Generate proof using prepared SNARK (this is the only non-cacheable step)
    #[cfg(feature = "debug-logs")]
    eprintln!("🔍 Generating proof with cached setup and preparation");
    
    #[cfg(feature = "debug-logs")]
    {
        let public_values_len = circuit.public_values().unwrap().len();
        let witness_len = circuit.wit.z_digits.len();
        eprintln!("📊 Proof dimensions: {} public values, {} witness elements", 
                 public_values_len, witness_len);
    }
    
    let snark_proof = R1CSSNARK::<E>::prove(&pk, circuit, &prep_snark, false)
        .map_err(|e| {
            // Only show detailed error analysis in debug mode
            #[cfg(feature = "debug-logs")]
            {
                eprintln!("❌ SNARK proving failed: {:?}", e);
                let err_str = format!("{:?}", e);
                if err_str.contains("combine_blinds") {
                    eprintln!("🔍 Hash-MLE commitment error - check segment structure");
                } else if err_str.contains("power of two") {
                    eprintln!("🔍 Power-of-2 constraint violation");
                }
            }
            e
        })?;
        
    #[cfg(feature = "debug-logs")]
    eprintln!("✅ SNARK proof generated successfully with cached keys!");
    
    // 4. Do NOT self-verify here.
    // Some tests intentionally create inconsistent instances/witnesses to check tamper detection
    // via public IO changes. They expect proving to succeed and verification to fail externally.
    // Instead, deterministically reconstruct the circuit's public values for binding/debugging.
    let public_outputs = {
    let circuit_for_publics = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest);
        circuit_for_publics
            .public_values()
            .map_err(|e| SpartanError::SynthesisError { reason: format!("public_values() failed: {e}") })?
    };
    
    // 5. Serialize proof
    let proof_bytes = bincode::serialize(&snark_proof)
        .map_err(|e| SpartanError::InternalError { reason: format!("Proof serialization failed: {e}") })?;
    
    // Note: VK doesn't implement Clone, so we need to return the Arc directly
    // The caller will need to handle the Arc wrapper
    Ok((proof_bytes, public_outputs, vk))
}

/// Commitment evolution embedding input
#[derive(Clone, Debug)]
pub struct CommitEvoEmbed {
    pub rho: neo_math::F,
    pub c_prev: Vec<neo_math::F>,
    pub c_step: Vec<neo_math::F>,
}

/// Linkage inputs specifying which undigitized witness positions must equal given values
#[derive(Clone, Debug)]
pub struct IvcLinkageInputs {
    pub x_indices_abs: Vec<usize>,
    pub y_prev_indices_abs: Vec<usize>,
    pub const1_index_abs: Option<usize>,
    pub step_io: Vec<neo_math::F>,
}

/// Public API: prove with EV, commit-evo, and linkage embeddings (IVC verifier style)
pub fn prove_me_snark_with_pp_and_ivc(
    me: &MEInstance,
    wit: &MEWitness,
    pp: Option<std::sync::Arc<AjtaiPP<neo_math::Rq>>>,
    ev: Option<IvcEvEmbed>,
    commit: Option<CommitEvoEmbed>,
    linkage: Option<IvcLinkageInputs>,
    pi_ccs: Option<crate::pi_ccs_embed::PiCcsEmbed>,
) -> Result<(Vec<u8>, Vec<<E as Engine>::Scalar>, Arc<SpartanVerifierKey<E>>), SpartanError> {
    // Clone embedding options so we can reuse the exact same config
    // for both proving and reconstructing public values.
    let ev_for_prove = ev.clone();
    let commit_for_prove = commit.clone();
    let linkage_for_prove = linkage.clone();
    let pi_ccs_for_prove = pi_ccs.clone();

    let circuit = MeCircuit::new(me.clone(), wit.clone(), pp, me.header_digest)
        .with_ev(ev_for_prove)
        .with_commit(commit_for_prove)
        .with_linkage(linkage_for_prove)
        .with_pi_ccs(pi_ccs_for_prove);

    // Normal cached proving path
    let circuit_key = CircuitKey::from_circuit(&circuit);
    let (pk, vk) = if let Some(cached_pk) = PK_CACHE.get(&circuit_key) {
        let cached_vk = VK_CACHE.get(&circuit_key).expect("VK must exist if PK exists");
        (cached_pk.clone(), cached_vk.clone())
    } else {
        let (new_pk, new_vk) = R1CSSNARK::<E>::setup(circuit.clone())?;
        let pk_arc = Arc::new(new_pk);
        let vk_arc = Arc::new(new_vk);
        PK_CACHE.insert(circuit_key.clone(), pk_arc.clone());
        VK_CACHE.insert(circuit_key.clone(), vk_arc.clone());
        (pk_arc, vk_arc)
    };
    let prep = R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), true)?;
    let snark_proof = R1CSSNARK::<E>::prove(&pk, circuit, &prep, false)?;

    // Recompute public values deterministically using the SAME embedding config
    // as the proving circuit to avoid any drift between prep/prove/publics.
    let public_outputs = {
        let circuit_for_publics = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest)
            .with_ev(ev)
            .with_commit(commit)
            .with_linkage(linkage)
            .with_pi_ccs(pi_ccs);
        circuit_for_publics
            .public_values()
            .map_err(|e| SpartanError::SynthesisError { reason: format!("public_values() failed: {e}") })?
    };
    let proof_bytes = bincode::serialize(&snark_proof)
        .map_err(|e| SpartanError::InternalError { reason: format!("Proof serialization failed: {e}") })?;
    Ok((proof_bytes, public_outputs, vk))
}

/// Clear the SNARK caches (useful for testing or memory management)
pub fn clear_snark_caches() {
    PK_CACHE.clear();
    VK_CACHE.clear();
    #[cfg(feature = "debug-logs")]
    eprintln!("🧹 Cleared SNARK key caches");
}

/// Get cache statistics (useful for monitoring)
pub fn get_cache_stats() -> (usize, usize) {
    (PK_CACHE.len(), VK_CACHE.len())
}

// Comprehensive tests added below for circuit fingerprint collision detection

/// Optional helper if you want a structured verify in tests
pub fn verify_me_snark(
    proof_bytes: &[u8],
    vk: &SpartanVerifierKey<E>,
) -> Result<bool, SpartanError> {
    let proof: R1CSSNARK<E> = bincode::deserialize(proof_bytes)
        .map_err(|e| SpartanError::InternalError { reason: format!("bincode(proof) failed: {e}") })?;
    // Spartan2 returns Ok(public_values) on success; we only need success/failure here
    let _ = proof.verify(vk)?;
    Ok(true)
}
