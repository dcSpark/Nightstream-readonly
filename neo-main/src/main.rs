use neo_fold::FoldState;
use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use neo_decomp::decomp_b;
use p3_matrix::dense::RowMajorMatrix;
use p3_field::PrimeCharacteristicRing;
use std::time::{Duration, Instant};

fn setup_test_structure() -> CcsStructure {
    let a = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], 3);
    let b = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO], 3);
    let c = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE], 3);
    let mats = vec![a, b, c];
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() != 3 {
                ExtF::ZERO
            } else {
                inputs[0] * inputs[1] - inputs[2]
            }
        },
        2,
    );
    CcsStructure::new(mats, f)
}

struct BenchmarkResults {
    setup_time: Duration,
    proof_generation_time: Duration,
    verification_time: Duration,
    total_time: Duration,
    success: bool,
}

impl BenchmarkResults {
    fn print_summary(&self, impl_name: &str) {
        println!("=== {} Performance Summary ===", impl_name);
        println!("Setup time:           {:>8.2} ms", self.setup_time.as_secs_f64() * 1000.0);
        println!("Proof generation:     {:>8.2} ms", self.proof_generation_time.as_secs_f64() * 1000.0);
        println!("Verification:         {:>8.2} ms", self.verification_time.as_secs_f64() * 1000.0);
        println!("Total time:           {:>8.2} ms", self.total_time.as_secs_f64() * 1000.0);
        println!("Result:               {:>8}", if self.success { "SUCCESS" } else { "FAILED" });
        println!();
    }
}

fn run_proof_generation_and_verification(impl_name: &str) -> BenchmarkResults {
    println!("=== Running proof generation and verification with {} ===", impl_name);
    let total_start = Instant::now();

    // Setup phase
    let setup_start = Instant::now();
    let structure = setup_test_structure();
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);

    // Setup first instance
    let z1_base = vec![F::ONE, F::from_u64(3), F::from_u64(3)];
    let z1 = z1_base.iter().copied().map(from_base).collect();
    let witness1 = CcsWitness { z: z1 };
    let z1_mat = decomp_b(&z1_base, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&z1_mat, &params);
    let mut t1 = Vec::new();
    let (commit1, _, _, _) = committer.commit(&w1, &mut t1).expect("commit");
    let instance1 = CcsInstance {
        commitment: commit1,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    // Setup second instance
    let z2_base = vec![F::from_u64(2), F::from_u64(2), F::from_u64(4)];
    let z2 = z2_base.iter().copied().map(from_base).collect();
    let witness2 = CcsWitness { z: z2 };
    let z2_mat = decomp_b(&z2_base, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&z2_mat, &params);
    let mut t2 = Vec::new();
    let (commit2, _, _, _) = committer.commit(&w2, &mut t2).expect("commit");
    let instance2 = CcsInstance {
        commitment: commit2,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let setup_time = setup_start.elapsed();
    println!("Setup completed in {:.2} ms", setup_time.as_secs_f64() * 1000.0);

    // Proof generation phase
    let proof_start = Instant::now();
    let mut state = FoldState::new(structure.clone());
    let proof = state.generate_proof((instance1.clone(), witness1.clone()), (instance2.clone(), witness2.clone()), &committer);
    let proof_generation_time = proof_start.elapsed();
    println!("Proof generation completed in {:.2} ms", proof_generation_time.as_secs_f64() * 1000.0);

    // Verification phase
    let verify_start = Instant::now();
    let verifier_state = FoldState::new(structure);
    let verify_result = verifier_state.verify(&proof.transcript, &committer);
    let verification_time = verify_start.elapsed();
    println!("Verification completed in {:.2} ms", verification_time.as_secs_f64() * 1000.0);

    let total_time = total_start.elapsed();

    if verify_result {
        println!("{} proof verification succeeded!", impl_name);
    } else {
        println!("{} proof verification failed!", impl_name);
    }
    
    BenchmarkResults {
        setup_time,
        proof_generation_time,
        verification_time,
        total_time,
        success: verify_result,
    }
}

fn main() {
    println!("Generating and verifying proofs with benchmarking...");
    println!("Note: The FRI implementation used depends on the neo-sumcheck features enabled at compile time.");
    println!();

    // Run first test
    println!("=== FIRST RUN ===");
    let first_result = run_proof_generation_and_verification("First Run");
    
    println!();
    
    // Run second test (demonstrates running twice with same implementation)
    println!("=== SECOND RUN ===");
    let second_result = run_proof_generation_and_verification("Second Run");

    println!();
    
    // Print detailed benchmark summaries
    first_result.print_summary("First Run");
    second_result.print_summary("Second Run");

    // Performance comparison
    println!("=== PERFORMANCE COMPARISON ===");
    println!("Setup time difference:        {:>8.2} ms", 
             (second_result.setup_time.as_secs_f64() - first_result.setup_time.as_secs_f64()) * 1000.0);
    println!("Proof generation difference:  {:>8.2} ms", 
             (second_result.proof_generation_time.as_secs_f64() - first_result.proof_generation_time.as_secs_f64()) * 1000.0);
    println!("Verification difference:      {:>8.2} ms", 
             (second_result.verification_time.as_secs_f64() - first_result.verification_time.as_secs_f64()) * 1000.0);
    println!("Total time difference:        {:>8.2} ms", 
             (second_result.total_time.as_secs_f64() - first_result.total_time.as_secs_f64()) * 1000.0);
    
    // Average performance
    let avg_setup = (first_result.setup_time + second_result.setup_time) / 2;
    let avg_proof = (first_result.proof_generation_time + second_result.proof_generation_time) / 2;
    let avg_verify = (first_result.verification_time + second_result.verification_time) / 2;
    let avg_total = (first_result.total_time + second_result.total_time) / 2;
    
    println!();
    println!("=== AVERAGE PERFORMANCE ===");
    println!("Average setup time:           {:>8.2} ms", avg_setup.as_secs_f64() * 1000.0);
    println!("Average proof generation:     {:>8.2} ms", avg_proof.as_secs_f64() * 1000.0);
    println!("Average verification:         {:>8.2} ms", avg_verify.as_secs_f64() * 1000.0);
    println!("Average total time:           {:>8.2} ms", avg_total.as_secs_f64() * 1000.0);

    println!();
    println!("=== FINAL SUMMARY ===");
    println!("First run result: {}", if first_result.success { "PASSED" } else { "FAILED" });
    println!("Second run result: {}", if second_result.success { "PASSED" } else { "FAILED" });
    
    if first_result.success && second_result.success {
        println!("Both runs succeeded!");
    } else if first_result.success || second_result.success {
        println!("At least one run succeeded.");
    } else {
        println!("Both runs failed.");
    }
}
