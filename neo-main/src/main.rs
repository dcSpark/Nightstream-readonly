use neo_fold::{FoldState, spartan_compress, spartan_verify};
use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use neo_decomp::decomp_b;
use p3_matrix::dense::RowMajorMatrix;
use p3_field::PrimeCharacteristicRing;
use std::time::{Duration, Instant};
use std::fmt::Write;

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



fn run_proof_generation_and_verification(impl_name: &str, log: &mut String) -> BenchmarkResults {
    writeln!(log, "=== Running proof generation and verification with {} ===", impl_name).unwrap();
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
    writeln!(log, "Setup completed in {:.2} ms", setup_time.as_secs_f64() * 1000.0).unwrap();

    // Proof generation phase
    let proof_start = Instant::now();
    let mut state = FoldState::new(structure.clone());
    let proof = state.generate_proof((instance1.clone(), witness1.clone()), (instance2.clone(), witness2.clone()), &committer);
    let proof_generation_time = proof_start.elapsed();
    writeln!(log, "Proof generation completed in {:.2} ms", proof_generation_time.as_secs_f64() * 1000.0).unwrap();

    // Verification phase
    let verify_start = Instant::now();
    let verifier_state = FoldState::new(structure);
    let verify_result = verifier_state.verify(&proof.transcript, &committer);
    let verification_time = verify_start.elapsed();
    writeln!(log, "Verification completed in {:.2} ms", verification_time.as_secs_f64() * 1000.0).unwrap();

    let total_time = total_start.elapsed();

    if verify_result {
        writeln!(log, "{} proof verification succeeded!", impl_name).unwrap();
    } else {
        writeln!(log, "{} proof verification failed!", impl_name).unwrap();
    }
    
    BenchmarkResults {
        setup_time,
        proof_generation_time,
        verification_time,
        total_time,
        success: verify_result,
    }
}

fn run_recursive_ivc_demo(log: &mut String) -> BenchmarkResults {
    writeln!(log, "=== Running Recursive IVC with Spartan Demo ===").unwrap();
    let total_start = Instant::now();

    // Setup phase
    let setup_start = Instant::now();
    let structure = setup_test_structure();
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);

    // Setup initial CCS instance for recursive IVC
    let z_base = vec![F::from_u64(2), F::from_u64(3), F::from_u64(6)]; // 2 * 3 = 6
    let z = z_base.iter().copied().map(from_base).collect();
    let witness = CcsWitness { z };
    let z_mat = decomp_b(&z_base, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&z_mat, &params);
    let mut t = Vec::new();
    let (commit, _, _, _) = committer.commit(&w, &mut t).expect("commit");
    let instance = CcsInstance {
        commitment: commit,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let setup_time = setup_start.elapsed();
    writeln!(log, "Setup completed in {:.2} ms", setup_time.as_secs_f64() * 1000.0).unwrap();

    // Recursive IVC phase
    let ivc_start = Instant::now();
    let mut state = FoldState::new(structure.clone());
    state.ccs_instance = Some((instance.clone(), witness.clone()));
    
    writeln!(log, "Starting recursive IVC with depth 3...").unwrap();
    let ivc_result = state.recursive_ivc(3, &committer);
    let ivc_time = ivc_start.elapsed();
    writeln!(log, "Recursive IVC completed in {:.2} ms", ivc_time.as_secs_f64() * 1000.0).unwrap();

    // Test Spartan compression directly
    let spartan_start = Instant::now();
    writeln!(log, "Testing Spartan compression directly...").unwrap();
    match spartan_compress(&structure, &instance, &witness, b"test_transcript") {
        Ok((proof, vk)) => {
            writeln!(log, "Spartan compression successful: proof.len()={}, vk.len()={}", 
                    proof.len(), vk.len()).unwrap();
            
            // Test verification (NARK mode - dummy verification)
            let transcript = b"demo_transcript";
            
            match spartan_verify(&proof, &vk, &structure, &instance, transcript) {
                Ok(true) => writeln!(log, "Spartan verification: PASSED").unwrap(),
                Ok(false) => writeln!(log, "Spartan verification: FAILED").unwrap(),
                Err(e) => writeln!(log, "Spartan verification error: {}", e).unwrap(),
            }
        },
        Err(e) => {
            writeln!(log, "Spartan compression failed: {}", e).unwrap();
        }
    }
    let spartan_time = spartan_start.elapsed();
    writeln!(log, "Spartan test completed in {:.2} ms", spartan_time.as_secs_f64() * 1000.0).unwrap();

    let total_time = total_start.elapsed();

    if ivc_result {
        writeln!(log, "Recursive IVC succeeded!").unwrap();
    } else {
        writeln!(log, "Recursive IVC failed!").unwrap();
    }
    
    BenchmarkResults {
        setup_time,
        proof_generation_time: ivc_time,
        verification_time: spartan_time,
        total_time,
        success: ivc_result,
    }
}

fn main() {
    let mut log = String::new();
    
    writeln!(log, "\n=== NEO + SPARTAN RECURSIVE IVC DEMO ===").unwrap();
    writeln!(log, "Running Neo folding with Spartan compression for recursive IVC...").unwrap();
    writeln!(log).unwrap();

    // Run the recursive IVC demo first
    let ivc_result = run_recursive_ivc_demo(&mut log);
    writeln!(log).unwrap();
    
    writeln!(log, "\n=== FRI IMPLEMENTATION COMPARISON ===").unwrap();
    writeln!(log, "Running proof generation and verification with BOTH implementations...").unwrap();
    writeln!(log).unwrap();

    // Test CUSTOM FRI (if available)
    let custom_result = if cfg!(feature = "custom-fri") {
        writeln!(log, "🔧 Testing CUSTOM FRI implementation...").unwrap();
        Some(run_proof_generation_and_verification("CUSTOM FRI", &mut log))
    } else {
        writeln!(log, "⚠️  CUSTOM FRI not available (feature not enabled)").unwrap();
        None
    };
    writeln!(log).unwrap();

    // Test p3-fri (if available)  
    let p3_result = if cfg!(feature = "p3-fri") {
        writeln!(log, "🔧 Testing p3-fri (Plonky3) implementation...").unwrap();
        Some(run_proof_generation_and_verification("p3-fri (Plonky3)", &mut log))
    } else {
        writeln!(log, "⚠️  p3-fri not available (feature not enabled)").unwrap();
        None
    };
    writeln!(log).unwrap();

    // Generate FINAL PERFORMANCE SUMMARY
    writeln!(log, "==========================================").unwrap();
    writeln!(log, "🏁 FINAL PERFORMANCE SUMMARY").unwrap();
    writeln!(log, "==========================================").unwrap();
    
    // Show IVC results first
    writeln!(log, "🔄 Recursive IVC Results:").unwrap();
    writeln!(log, "Setup time:       {:>8.2} ms", ivc_result.setup_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "IVC execution:    {:>8.2} ms", ivc_result.proof_generation_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Spartan test:     {:>8.2} ms", ivc_result.verification_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Total time:       {:>8.2} ms", ivc_result.total_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Success:          {}", if ivc_result.success { "✅ PASSED" } else { "❌ FAILED" }).unwrap();
    writeln!(log, "").unwrap();
    
    match (custom_result, p3_result) {
        (Some(custom), Some(p3)) => {
            // Both implementations available - show comparison
            writeln!(log, "📊 CUSTOM FRI vs p3-fri (Plonky3) Comparison:").unwrap();
            writeln!(log, "").unwrap();
            writeln!(log, "{:<20} {:>15} {:>15}", "Metric", "CUSTOM FRI", "p3-fri").unwrap();
            writeln!(log, "----------------------------------------------------").unwrap();
            writeln!(log, "{:<20} {:>12.2} ms {:>12.2} ms", "Setup time:", custom.setup_time.as_secs_f64() * 1000.0, p3.setup_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "{:<20} {:>12.2} ms {:>12.2} ms", "Proof generation:", custom.proof_generation_time.as_secs_f64() * 1000.0, p3.proof_generation_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "{:<20} {:>12.2} ms {:>12.2} ms", "Verification:", custom.verification_time.as_secs_f64() * 1000.0, p3.verification_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "{:<20} {:>12.2} ms {:>12.2} ms", "Total time:", custom.total_time.as_secs_f64() * 1000.0, p3.total_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "----------------------------------------------------").unwrap();
            
            // Determine winner
            let custom_total = custom.total_time.as_secs_f64() * 1000.0;
            let p3_total = p3.total_time.as_secs_f64() * 1000.0;
            let (winner, diff) = if custom_total < p3_total {
                ("CUSTOM FRI", p3_total - custom_total)
            } else {
                ("p3-fri", custom_total - p3_total)
            };
            
            writeln!(log, "").unwrap();
            writeln!(log, "🏆 Winner: {} (faster by {:.2} ms)", winner, diff).unwrap();
            writeln!(log, "").unwrap();
            writeln!(log, "✅ Both implementations: {}", 
                if custom.success && p3.success { "PASSED" } else { "SOME FAILED" }).unwrap();
        },
        (Some(custom), None) => {
            // Only CUSTOM FRI available
            writeln!(log, "🔧 CUSTOM FRI Results:").unwrap();
            writeln!(log, "Setup time:       {:>8.2} ms", custom.setup_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Proof generation: {:>8.2} ms", custom.proof_generation_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Verification:     {:>8.2} ms", custom.verification_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Total time:       {:>8.2} ms", custom.total_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Success:          {}", if custom.success { "✅ PASSED" } else { "❌ FAILED" }).unwrap();
        },
        (None, Some(p3)) => {
            // Only p3-fri available
            writeln!(log, "🔧 p3-fri (Plonky3) Results:").unwrap();
            writeln!(log, "Setup time:       {:>8.2} ms", p3.setup_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Proof generation: {:>8.2} ms", p3.proof_generation_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Verification:     {:>8.2} ms", p3.verification_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Total time:       {:>8.2} ms", p3.total_time.as_secs_f64() * 1000.0).unwrap();
            writeln!(log, "Success:          {}", if p3.success { "✅ PASSED" } else { "❌ FAILED" }).unwrap();
        },
        (None, None) => {
            writeln!(log, "❌ No FRI implementations available!").unwrap();
            writeln!(log, "Please enable either 'custom-fri' or 'p3-fri' features.").unwrap();
        }
    }
    
    writeln!(log, "==========================================").unwrap();

    print!("{}", log);
}
