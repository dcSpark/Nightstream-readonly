use neo_arithmetize::fibonacci_ccs;
use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_fold::FoldState;
use neo_ring::RingElement;
use neo_modint::ModInt;
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

fn main() {
    println!("Neo Lattice Demo: Proving Fibonacci Series");

    // Secure parameters for production use
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);

    // High-level input: Length of Fibonacci series to prove (toy size)
    let fib_length = 3;

    // Convert high-level input to CCS (using new module)
    let ccs = fibonacci_ccs(fib_length);

    // Generate witness: Fibonacci sequence
    let mut z: Vec<ExtF> = vec![ExtF::ZERO; fib_length];
    z[0] = embed_base_to_ext(F::ZERO);
    z[1] = embed_base_to_ext(F::ONE);
    for i in 2..fib_length {
        z[i] = z[i - 1] + z[i - 2];
    }
    let witness = CcsWitness { z: z.clone() };

    // Pack witness into ring elements for commitment
    let z_base: Vec<F> = z.iter().map(|e| e.to_array()[0]).collect();  // Project to base (real part)
    let decomp_mat = decomp_b(&z_base, committer.params().b, committer.params().d);
    let z_packed: Vec<RingElement<ModInt>> = AjtaiCommitter::pack_decomp(&decomp_mat, &committer.params());

    // Commit to packed witness
    let (commitment, _, _, _) = committer.commit(&z_packed, &mut vec![]).unwrap();

    // Create CCS instance
    let instance = CcsInstance {
        commitment,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    // Check satisfiability
    if !check_satisfiability(&ccs, &instance, &witness) {
        println!("Error: Fibonacci witness does not satisfy CCS constraints");
        return;
    }

    // Setup folding state
    let mut fold_state = FoldState::new(ccs);

    // Generate proof for the Fibonacci instance (fold with itself for demo)
    println!("\n=== PROOF GENERATION ===");
    let proof_start = Instant::now();
    let proof = fold_state.generate_proof((instance.clone(), witness.clone()), (instance, witness), &committer);
    let proof_generation_time = proof_start.elapsed();
    let proof_size_bytes = proof.transcript.len();
    println!("Proof generation completed in {:.2} ms", proof_generation_time.as_secs_f64() * 1000.0);
    println!("Proof size: {} bytes", proof_size_bytes);

    // Verify the proof
    println!("\n=== PROOF VERIFICATION ===");
    let verify_start = Instant::now();
    let verified = fold_state.verify(&proof.transcript, &committer);
    let verification_time = verify_start.elapsed();
    println!("Proof verification: {}", if verified { "SUCCESS" } else { "FAILURE" });
    println!("Verification completed in {:.2} ms", verification_time.as_secs_f64() * 1000.0);

    // Final Performance Summary
    println!("\n==========================================");
    println!("🏁 FINAL PERFORMANCE SUMMARY");
    println!("==========================================");
    println!("Proof Generation Time:    {:>8.2} ms", proof_generation_time.as_secs_f64() * 1000.0);
    println!("Proof Size:               {:>8} bytes", proof_size_bytes);
    println!("Proof Verification Time:  {:>8.2} ms", verification_time.as_secs_f64() * 1000.0);
    println!("Total Time:               {:>8.2} ms", (proof_generation_time + verification_time).as_secs_f64() * 1000.0);
    println!("Verification Result:      {}", if verified { "✅ PASSED" } else { "❌ FAILED" });
    println!("==========================================");
}
