use neo_fold::FoldState;
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

fn main() {
    let mut log = String::new();
    
    // Detect which FRI implementation is being used
    let fri_impl = if cfg!(feature = "custom-fri") {
        "CUSTOM FRI"
    } else if cfg!(feature = "p3-fri") {
        "p3-fri (Plonky3)"
    } else {
        "Unknown FRI Implementation"
    };
    
    writeln!(log, "\n=== {} BENCHMARK ===", fri_impl).unwrap();
    writeln!(log, "Running proof generation and verification...").unwrap();
    writeln!(log).unwrap();

    // Run single test with proper FRI implementation name
    let result = run_proof_generation_and_verification(fri_impl, &mut log);
    writeln!(log).unwrap();

    writeln!(log, "\n").unwrap();
    writeln!(log, "==========================================").unwrap();
    writeln!(log, "🏁 FINAL PERFORMANCE SUMMARY").unwrap();
    writeln!(log, "==========================================").unwrap();
    writeln!(log, "🔧 FRI Implementation: {}", fri_impl).unwrap();
    writeln!(log, "Setup time:       {:>8.2} ms", result.setup_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Proof generation: {:>8.2} ms", result.proof_generation_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Verification:     {:>8.2} ms", result.verification_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Total time:       {:>8.2} ms", result.total_time.as_secs_f64() * 1000.0).unwrap();
    writeln!(log, "Success:          {}", if result.success { "✅ PASSED" } else { "❌ FAILED" }).unwrap();
    writeln!(log, "==========================================").unwrap();

    print!("{}", log);
}
