#![allow(non_snake_case)]

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use ff::PrimeField;
use neo_ajtai::Commitment as Cmt;
use neo_ajtai::{set_global_pp_seeded, s_lincomb, s_mul, AjtaiSModule};
use neo_ccs::relations::{McsInstance, McsWitness, MeInstance};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsMatrix, CcsStructure, CscMat, Mat, SparsePoly, Term};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove_with_witnesses, fold_shard_prove_with_witnesses_with_step_offset, fold_shard_verify,
    fold_shard_verify_with_step_offset, fold_shard_verify_with_step_linking, CommitMixers, StepLinkingConfig,
};
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};
use std::time::Instant;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
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

extern crate ff;

#[derive(PrimeField)]
#[PrimeFieldModulus = "18446744069414584321"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct FpGoldilocks([u64; 2]);

fn fp_to_u64(x: &FpGoldilocks) -> u64 {
    let bytes = x.to_repr();
    u64::from_le_bytes(bytes.0[0..8].try_into().expect("repr is at least 8 bytes"))
}

// Bellpepper's TestConstraintSystem is optimized for debuggability (names, hashing, storage),
// which is expensive for SHA256 circuits. We only need:
// - variable assignments (inputs + aux) to build witnesses, and
// - sparse A/B/C matrices (triplets) for the R1CS → CCS embedding.
const AUX_FLAG: u32 = 1 << 31;

struct TripletConstraintSystem {
    inputs: Vec<F>, // includes input[0] = ONE
    aux: Vec<F>,
    num_constraints: u32,
    a_trips: Vec<(u32, u32, F)>,
    b_trips: Vec<(u32, u32, F)>,
    c_trips: Vec<(u32, u32, F)>,
}

impl TripletConstraintSystem {
    fn new() -> Self {
        Self {
            inputs: vec![F::ONE],
            aux: Vec::new(),
            num_constraints: 0,
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
        }
    }

    fn push_lc_trips(
        row: u32,
        lc: &LinearCombination<FpGoldilocks>,
        trips: &mut Vec<(u32, u32, F)>,
    ) {
        for (var, coeff) in lc.iter() {
            let value = fp_to_u64(coeff);
            if value == 0 {
                continue;
            }
            let col = match var.0 {
                Index::Input(i) => u32::try_from(i).expect("input index should fit in u32"),
                Index::Aux(i) => AUX_FLAG | u32::try_from(i).expect("aux index should fit in u32"),
            };
            trips.push((row, col, F::from_u64(value)));
        }
    }

    fn resolve_triplets(trips: Vec<(u32, u32, F)>, num_inputs: usize) -> Vec<(usize, usize, F)> {
        trips
            .into_iter()
            .map(|(row, col, value)| {
                let row = row as usize;
                if (col & AUX_FLAG) == 0 {
                    (row, col as usize, value)
                } else {
                    let aux_idx = (col & !AUX_FLAG) as usize;
                    (row, num_inputs + aux_idx, value)
                }
            })
            .collect()
    }
}

impl ConstraintSystem<FpGoldilocks> for TripletConstraintSystem {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.aux.len();
        let value = f()?;
        self.aux.push(F::from_u64(fp_to_u64(&value)));
        Ok(Variable::new_unchecked(Index::Aux(idx)))
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.inputs.len();
        let value = f()?;
        self.inputs.push(F::from_u64(fp_to_u64(&value)));
        Ok(Variable::new_unchecked(Index::Input(idx)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LB: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LC: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
    {
        let row = self.num_constraints;
        self.num_constraints += 1;

        let a_lc = a(LinearCombination::zero());
        let b_lc = b(LinearCombination::zero());
        let c_lc = c(LinearCombination::zero());

        Self::push_lc_trips(row, &a_lc, &mut self.a_trips);
        Self::push_lc_trips(row, &b_lc, &mut self.b_trips);
        Self::push_lc_trips(row, &c_lc, &mut self.c_trips);
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

const STATE_BYTES: usize = 32;
const STATE_BITS: usize = STATE_BYTES * 8;

struct Sha256StateChainCircuit {
    prev_state: [u8; STATE_BYTES],
}

impl Circuit<FpGoldilocks> for Sha256StateChainCircuit {
    fn synthesize<CS: ConstraintSystem<FpGoldilocks>>(
        self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        let prev_bit_values: Vec<_> = bellpepper::gadgets::multipack::bytes_to_bits(&self.prev_state)
            .into_iter()
            .map(Some)
            .collect();
        assert_eq!(prev_bit_values.len(), STATE_BITS);

        let prev_bits = prev_bit_values
            .into_iter()
            .enumerate()
            .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("prev bit {i}")), b))
            .map(|b| b.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        // Expose `prev_state` as compact public inputs.
        bellpepper::gadgets::multipack::pack_into_inputs(cs.namespace(|| "prev_state"), &prev_bits)?;

        // Compute `next_state = SHA256(prev_state)`.
        let hash_bits =
            bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &prev_bits)?;
        assert_eq!(hash_bits.len(), STATE_BITS);

        // Expose `next_state` as compact public inputs.
        bellpepper::gadgets::multipack::pack_into_inputs(cs.namespace(|| "next_state"), &hash_bits)?;

        Ok(())
    }
}

fn state_packed_len() -> usize {
    let cap = <FpGoldilocks as PrimeField>::CAPACITY as usize;
    (STATE_BITS + cap - 1) / cap
}

fn pack_state_bytes(state: &[u8; STATE_BYTES]) -> Vec<F> {
    let bits = bellpepper::gadgets::multipack::bytes_to_bits(state);
    let packed =
        bellpepper::gadgets::multipack::compute_multipacking::<FpGoldilocks>(&bits);
    packed.iter().map(|x| F::from_u64(fp_to_u64(x))).collect()
}

fn unpack_state_bytes_from_packed(packed: &[F]) -> [u8; STATE_BYTES] {
    let cap = <FpGoldilocks as PrimeField>::CAPACITY as usize;
    debug_assert_eq!(packed.len(), state_packed_len());
    let mut out = [0u8; STATE_BYTES];
    for byte_idx in 0..STATE_BYTES {
        let mut byte = 0u8;
        for bit_in_byte in 0..8 {
            let bit_idx = byte_idx * 8 + bit_in_byte;
            let elem_idx = bit_idx / cap;
            let shift = bit_idx % cap;
            let bit = (packed[elem_idx].as_canonical_u64() >> shift) & 1;
            if bit == 1 {
                byte |= 1u8 << (7 - bit_in_byte);
            }
        }
        out[byte_idx] = byte;
    }
    out
}

fn sha256_state(state: &[u8; STATE_BYTES]) -> [u8; STATE_BYTES] {
    let digest = Sha256::digest(state);
    let mut out = [0u8; STATE_BYTES];
    out.copy_from_slice(digest.as_ref());
    out
}

fn sha256_bytes(bytes: &[u8]) -> [u8; STATE_BYTES] {
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; STATE_BYTES];
    out.copy_from_slice(digest.as_ref());
    out
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn build_sha256_chain_ccs() -> (CcsStructure<F>, usize) {
    let mut cs = TripletConstraintSystem::new();
    let circuit = Sha256StateChainCircuit {
        prev_state: [0u8; STATE_BYTES],
    };
    circuit
        .synthesize(&mut cs)
        .expect("Circuit synthesis should succeed");

    let TripletConstraintSystem {
        inputs,
        aux,
        num_constraints,
        a_trips,
        b_trips,
        c_trips,
    } = cs;

    let num_constraints = num_constraints as usize;
    let num_inputs = inputs.len();
    let num_aux = aux.len();
    let num_variables = num_inputs + num_aux;

    let a_trips = TripletConstraintSystem::resolve_triplets(a_trips, num_inputs);
    let b_trips = TripletConstraintSystem::resolve_triplets(b_trips, num_inputs);
    let c_trips = TripletConstraintSystem::resolve_triplets(c_trips, num_inputs);

    // R1CS → CCS embedding (rectangular-capable): M_1=A, M_2=B, M_3=C and
    // f(X1,X2,X3) = X1*X2 - X3 (elementwise).
    let f_base = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0],
            }, // X1 * X2
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            }, // -X3
        ],
    );

    let n = num_constraints;
    let m = num_variables;
    let matrices = vec![
        CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, m)),
        CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, m)),
        CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, m)),
    ];
    let ccs = CcsStructure::new_sparse(matrices, f_base).expect("valid R1CS→CCS structure");
    (ccs, num_inputs)
}

fn build_sha256_chain_witness(prev_state: [u8; STATE_BYTES]) -> (Vec<F>, usize) {
    let mut cs = TripletConstraintSystem::new();
    let circuit = Sha256StateChainCircuit { prev_state };
    circuit
        .synthesize(&mut cs)
        .expect("Circuit synthesis should succeed");

    let TripletConstraintSystem { inputs, aux, .. } = cs;
    let m_in = inputs.len();

    let mut witness = inputs;
    witness.extend(aux);
    (witness, m_in)
}

fn build_step<L: SModuleHomomorphism<F, Cmt>>(
    params: &NeoParams,
    l: &L,
    m_in: usize,
    z: Vec<F>,
) -> StepWitnessBundle<Cmt, F, K> {
    let Z = encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    StepWitnessBundle::from((
        McsInstance { c, x, m_in },
        McsWitness { w, Z },
    ))
}

#[test]
fn shard_continuation_extend_and_fold() {
    // Demonstrates “continuation” at the shard folding layer using a real, chainable SHA256 circuit.
    //
    // Concrete statement (human-checkable):
    // - seed = "abc"
    // - y0 = sha256(seed)  = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    // - y1 = sha256(y0)    = 4f8b42c22dd3729b519ba6f68d2da7cc5b2d606d05daed5ad5128cc03e6c6358
    // - y2 = sha256(y1)    = f2a778f1a6ed3d5bc59a5d79104c598f3f07093f240ca4e91333fb09ed4f36da
    //
    // Circuit per step: prove `next_state = sha256(prev_state)` while exposing both states publicly.
    // - State is 32 bytes (256 bits).
    // - Bellpepper multipacks bits into field elements of capacity CAPACITY=63 bits, so 256 bits
    //   become `packed_len = ceil(256/63) = 5` field elements.
    // - Therefore each step's public input vector is:
    //     x = [1, pack(prev_state) (5 elems), pack(next_state) (5 elems)]  => m_in = 11
    //
    // Continuation flow:
    // 1) Stage 0 (prefix): prove one step with public statement (y0 -> y1). Keep the foldable
    //    accumulator + witness (not finalized; still foldable).
    // 2) Stage 1 (extension): prove a second step with statement (y1 -> y2) starting from that
    //    accumulator (this is the "extend the computation" part).
    // 3) Verify both proofs and assert the boundary condition `next(step0) == prev(step1)` i.e.
    //    y1 is both step-0 output and step-1 input, then recompute SHA256 in software to
    //    corroborate the verified public outputs match `sha256(y0)` and `sha256(y1)`.
    //
    // Notes:
    // - We use real Ajtai commitments + real mixers (s_lincomb / s_mul), no dummy committer/mixer.
    // - The CCS is rectangular (currently ~25666 constraints × ~25485 vars) and the optimized engine
    //   supports rectangular CCS (no square padding needed here).
    // - For a single, self-contained “full IVC” proof over both steps, we can concatenate the
    //   step proofs from stage 0 and stage 1 and verify once with `fold_shard_verify_with_step_linking`.
    let total_start = Instant::now();

    let (ccs, m_in) = build_sha256_chain_ccs();
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.b = 3;
    let seed = [42u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, ccs.m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, ccs.m).expect("AjtaiSModule init");
    let mixers = default_mixers();

    println!("=== Shard Continuation Test (SHA256 State Chaining) ===");
    println!("CCS dimensions: n_constraints={}, m_variables={}", ccs.n, ccs.m);
    println!("Params: b={}, m_in={}", params.b, m_in);

    let packed_len = state_packed_len();
    assert_eq!(m_in, 1 + 2 * packed_len);

    let seed = b"abc";
    let y0 = sha256_bytes(seed);
    let y1 = sha256_state(&y0);
    let y2 = sha256_state(&y1);

    // Human-checkable test vectors:
    // y0 = sha256("abc")
    // y1 = sha256(y0)
    // y2 = sha256(y1)
    const Y0_EXPECTED: [u8; STATE_BYTES] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
        0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
        0xf2, 0x00, 0x15, 0xad,
    ];
    const Y1_EXPECTED: [u8; STATE_BYTES] = [
        0x4f, 0x8b, 0x42, 0xc2, 0x2d, 0xd3, 0x72, 0x9b, 0x51, 0x9b, 0xa6, 0xf6, 0x8d, 0x2d,
        0xa7, 0xcc, 0x5b, 0x2d, 0x60, 0x6d, 0x05, 0xda, 0xed, 0x5a, 0xd5, 0x12, 0x8c, 0xc0,
        0x3e, 0x6c, 0x63, 0x58,
    ];
    const Y2_EXPECTED: [u8; STATE_BYTES] = [
        0xf2, 0xa7, 0x78, 0xf1, 0xa6, 0xed, 0x3d, 0x5b, 0xc5, 0x9a, 0x5d, 0x79, 0x10, 0x4c,
        0x59, 0x8f, 0x3f, 0x07, 0x09, 0x3f, 0x24, 0x0c, 0xa4, 0xe9, 0x13, 0x33, 0xfb, 0x09,
        0xed, 0x4f, 0x36, 0xda,
    ];

    assert_eq!(y0, Y0_EXPECTED);
    assert_eq!(y1, Y1_EXPECTED);
    assert_eq!(y2, Y2_EXPECTED);

    let y0_packed = pack_state_bytes(&y0);
    let y1_packed = pack_state_bytes(&y1);
    let y2_packed = pack_state_bytes(&y2);

    println!("Seed: {}", core::str::from_utf8(seed).expect("utf8"));
    println!("y0 = sha256(seed): {}", hex_lower(&y0));
    println!("y1 = sha256(y0):   {}", hex_lower(&y1));
    println!("y2 = sha256(y1):   {}", hex_lower(&y2));
    println!(
        "Stage 0 output (y1) packed public inputs: {:?}",
        y1_packed
            .iter()
            .map(|v| v.as_canonical_u64())
            .collect::<Vec<_>>()
    );

    let (w0, m_in0) = build_sha256_chain_witness(y0);
    let (w1, m_in1) = build_sha256_chain_witness(y1);
    assert_eq!(m_in0, m_in);
    assert_eq!(m_in1, m_in);
    assert_eq!(w0.len(), ccs.m);
    assert_eq!(w1.len(), ccs.m);

    let step0 = build_step(&params, &l, m_in, w0);
    let step1 = build_step(&params, &l, m_in, w1);

    // ---------------------------------------------------------------------
    // Prove a prefix shard, keep the foldable accumulator + its witness, then
    // extend with another step and fold again.
    // ---------------------------------------------------------------------
    println!("\n--- Prover Phase ---");
    let mut tr_p = Poseidon2Transcript::new(b"neo.fold/shard_continuation");

    let prove_prefix_start = Instant::now();
    let (proof0, out0, wits0) = fold_shard_prove_with_witnesses_with_step_offset(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        core::slice::from_ref(&step0),
        &[],
        &[],
        &l,
        mixers,
        0,
    )
    .expect("prove prefix");
    let prove_prefix_duration = prove_prefix_start.elapsed();
    println!("Prove prefix shard (step 0): {:?}", prove_prefix_duration);
    assert!(out0.obligations.val.is_empty(), "CCS-only prefix should not emit val-lane obligations");
    println!("Stage 0 proved output (y1): {}", hex_lower(&y1));

    let acc1: Vec<MeInstance<Cmt, F, K>> = out0.obligations.main.clone();
    let acc1_wit: Vec<Mat<F>> = wits0.final_main_wits.clone();
    assert_eq!(acc1.len(), acc1_wit.len(), "accumulator witness count mismatch");
    println!("  Accumulator size after prefix: {} MeInstance(s)", acc1.len());

    let prove_extend_start = Instant::now();
    let (proof1, out1, _wits1) = fold_shard_prove_with_witnesses_with_step_offset(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        core::slice::from_ref(&step1),
        &acc1,
        &acc1_wit,
        &l,
        mixers,
        1,
    )
    .expect("prove extension");
    let prove_extend_duration = prove_extend_start.elapsed();
    println!("Prove extension shard (step 1): {:?}", prove_extend_duration);
    assert!(out1.obligations.val.is_empty(), "CCS-only extension should not emit val-lane obligations");
    println!("Stage 1 proved output (y2): {}", hex_lower(&y2));
    println!(
        "Stage 1 output (y2) packed public inputs: {:?}",
        y2_packed
            .iter()
            .map(|v| v.as_canonical_u64())
            .collect::<Vec<_>>()
    );

    let total_prove_duration = prove_prefix_duration + prove_extend_duration;
    println!("Total proving time: {:?}", total_prove_duration);

    // Full IVC proof: prove both steps in a *single* shard proof so the verifier can enforce
    // step-to-step chaining using `fold_shard_verify_with_step_linking`.
    //
    // (In contrast, the "continuation" demo above produces two separate shard proofs that must be
    // verified sequentially with a shared transcript state.)
    let mut tr_p_full = Poseidon2Transcript::new(b"neo.fold/shard_continuation_full_ivc");
    let steps_full = [step0.clone(), step1.clone()];
    let prove_full_start = Instant::now();
    let (proof_full, _out_full, _wits_full) = fold_shard_prove_with_witnesses(
        FoldingMode::Optimized,
        &mut tr_p_full,
        &params,
        &ccs,
        &steps_full,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove full IVC shard (steps 0..1)");
    println!(
        "Prove full IVC shard (steps 0..1): {:?}",
        prove_full_start.elapsed()
    );

    // ---------------------------------------------------------------------
    // Verifier side: verify the prefix to get the foldable state, then verify
    // the extension using that foldable state as `acc_init`.
    // ---------------------------------------------------------------------
    println!("\n--- Verifier Phase ---");
    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/shard_continuation");
    let step0_pub: StepInstanceBundle<Cmt, F, K> = StepInstanceBundle::from(&step0);
    let step1_pub: StepInstanceBundle<Cmt, F, K> = StepInstanceBundle::from(&step1);

    // Enforce IVC-style chaining at the statement level:
    // prev_step.next_state == next_step.prev_state (in packed field element form).
    let step_linking = StepLinkingConfig::new(
        (0..packed_len)
            .map(|i| (1 + packed_len + i, 1 + i))
            .collect::<Vec<_>>(),
    );

    // The circuit exposes `prev_state` and `next_state` as packed public inputs:
    // x = [1, pack(prev_state), pack(next_state)].
    assert_eq!(step0_pub.mcs_inst.x[0], F::ONE);
    assert_eq!(&step0_pub.mcs_inst.x[1..1 + packed_len], y0_packed.as_slice());
    assert_eq!(
        &step0_pub.mcs_inst.x[1 + packed_len..1 + 2 * packed_len],
        y1_packed.as_slice()
    );
    assert_eq!(&step1_pub.mcs_inst.x[1..1 + packed_len], y1_packed.as_slice());
    assert_eq!(
        &step1_pub.mcs_inst.x[1 + packed_len..1 + 2 * packed_len],
        y2_packed.as_slice()
    );

    // This is the boundary condition we want to preserve when "continuing" a proof:
    // the next-state of step 0 becomes the prev-state of step 1.
    assert_eq!(
        &step0_pub.mcs_inst.x[1 + packed_len..1 + 2 * packed_len],
        &step1_pub.mcs_inst.x[1..1 + packed_len]
    );

    let verify_prefix_start = Instant::now();
    let out0_v = fold_shard_verify_with_step_offset(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        core::slice::from_ref(&step0_pub),
        &[],
        &proof0,
        mixers,
        0,
    )
    .expect("verify prefix");
    let verify_prefix_duration = verify_prefix_start.elapsed();
    println!("Verify prefix shard (step 0): {:?}", verify_prefix_duration);
    assert_eq!(out0_v.obligations.main, out0.obligations.main, "prefix obligations mismatch");
    assert!(out0_v.obligations.val.is_empty());

    // Sanity check: verifying step 1 with the wrong step offset must fail.
    let mut tr_v_wrong = Poseidon2Transcript::new(b"neo.fold/shard_continuation");
    let out0_v_wrong = fold_shard_verify_with_step_offset(
        FoldingMode::Optimized,
        &mut tr_v_wrong,
        &params,
        &ccs,
        core::slice::from_ref(&step0_pub),
        &[],
        &proof0,
        mixers,
        0,
    )
    .expect("verify prefix (wrong-offset transcript)");
    assert!(
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_v_wrong,
            &params,
            &ccs,
            core::slice::from_ref(&step1_pub),
            &out0_v_wrong.obligations.main,
            &proof1,
            mixers,
        )
        .is_err(),
        "verifying step 1 with step_idx_offset=0 must fail"
    );

    let verify_extend_start = Instant::now();
    let out1_v = fold_shard_verify_with_step_offset(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        core::slice::from_ref(&step1_pub),
        &out0_v.obligations.main,
        &proof1,
        mixers,
        1,
    )
    .expect("verify extension");
    let verify_extend_duration = verify_extend_start.elapsed();
    println!("Verify extension shard (step 1): {:?}", verify_extend_duration);
    assert_eq!(out1_v.obligations.main, out1.obligations.main, "extended obligations mismatch");
    assert!(out1_v.obligations.val.is_empty());

    // Full IVC verification: verify a single 2-step shard proof with step linking enabled.
    let mut tr_v_full = Poseidon2Transcript::new(b"neo.fold/shard_continuation_full_ivc");
    let verify_full_start = Instant::now();
    let _out_full_v = fold_shard_verify_with_step_linking(
        FoldingMode::Optimized,
        &mut tr_v_full,
        &params,
        &ccs,
        &[step0_pub.clone(), step1_pub.clone()],
        &[],
        &proof_full,
        mixers,
        &step_linking,
    )
    .expect("verify full IVC proof with step linking");
    println!(
        "Verify full IVC shard (steps 0..1, with step linking): {:?}",
        verify_full_start.elapsed()
    );

    let step0_next = unpack_state_bytes_from_packed(
        &step0_pub.mcs_inst.x[1 + packed_len..1 + 2 * packed_len],
    );
    let step1_prev = unpack_state_bytes_from_packed(&step1_pub.mcs_inst.x[1..1 + packed_len]);
    let step1_next = unpack_state_bytes_from_packed(
        &step1_pub.mcs_inst.x[1 + packed_len..1 + 2 * packed_len],
    );
    assert_eq!(step0_next, step1_prev, "chaining boundary mismatch");
    println!("Stage 1 verified input (y1):  {}", hex_lower(&step1_prev));
    println!("Stage 1 verified output (y2): {}", hex_lower(&step1_next));

    // Extra corroboration: recompute SHA256 outside the proving system (pure software hash) and
    // confirm it matches the public statement that was verified.
    let step0_prev = unpack_state_bytes_from_packed(&step0_pub.mcs_inst.x[1..1 + packed_len]);
    let step0_next_recomputed = sha256_state(&step0_prev);
    let step1_next_recomputed = sha256_state(&step1_prev);
    assert_eq!(step0_next_recomputed, step0_next, "external sha256(y0) != y1");
    assert_eq!(step1_next_recomputed, step1_next, "external sha256(y1) != y2");
    println!(
        "External check: sha256(y0) == y1: {}",
        hex_lower(&step0_next_recomputed)
    );
    println!(
        "External check: sha256(y1) == y2: {}",
        hex_lower(&step1_next_recomputed)
    );

    let total_verify_duration = verify_prefix_duration + verify_extend_duration;
    println!("Total verification time: {:?}", total_verify_duration);

    println!("\n--- Summary ---");
    println!("Total proving time:      {:?}", total_prove_duration);
    println!("Total verification time: {:?}", total_verify_duration);
    println!("Total elapsed time:      {:?}", total_start.elapsed());
    println!("Final accumulator size:  {} MeInstance(s)", out1.obligations.main.len());
}
