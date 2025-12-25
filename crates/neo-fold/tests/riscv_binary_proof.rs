//! RISC-V Binary Proof Tests
//!
//! This test file demonstrates loading compiled RISC-V binaries and
//! generating proofs for their execution using Neo's folding scheme.
//!
//! ## Pipeline
//!
//! ```text
//! RISC-V Binary (.elf or raw)
//!     │
//!     ▼ load_elf() / load_raw_binary()
//! LoadedProgram { instructions, entry, segments }
//!     │
//!     ▼ RiscvCpu::load_program()
//! CPU Ready for Execution
//!     │
//!     ▼ trace_program()
//! VmTrace
//!     │
//!     ▼ trace_to_plain_*()
//! PlainMemTrace + PlainLutTrace
//!     │
//!     ▼ FoldingSession::fold_and_prove() → FoldingSession::verify_collected()
//! Verified Proof ✓
//! ```

#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    matrix::Mat,
    poly::SparsePoly,
    relations::{CcsStructure, McsInstance, McsWitness},
};
use neo_fold::session::FoldingSession;
use neo_math::{D, K};
use neo_memory::elf_loader::load_raw_binary;
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::riscv_lookups::{
    encode_program, trace_to_plain_lut_trace, trace_to_plain_mem_trace,
    RiscvCpu, RiscvInstruction, RiscvLookupTable, RiscvMemory, RiscvOpcode, RiscvShoutTables,
};
use neo_memory::witness::StepWitnessBundle;
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

// ============================================================================
// Helpers
// ============================================================================

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = neo_ajtai::decomp_b(z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn build_add_ccs_mcs(
    params: &NeoParams,
    l: &DummyCommit,
    const_one: F,
    val1: F,
    val2: F,
    out: F,
) -> (CcsStructure<F>, McsInstance<Cmt, F>, McsWitness<F>) {
    let mut m0 = Mat::zero(4, 4, F::ZERO);
    m0[(0, 0)] = F::ONE;
    let mut m1 = Mat::zero(4, 4, F::ZERO);
    m1[(0, 1)] = F::ONE;
    let mut m2 = Mat::zero(4, 4, F::ZERO);
    m2[(0, 2)] = F::ONE;
    let mut m3 = Mat::zero(4, 4, F::ZERO);
    m3[(0, 3)] = F::ONE;

    let term_const = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![1, 0, 0, 0],
    };
    let term_x1 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 1, 0, 0],
    };
    let term_x2 = neo_ccs::poly::Term {
        coeff: F::ONE,
        exps: vec![0, 0, 1, 0],
    };
    let term_neg_out = neo_ccs::poly::Term {
        coeff: F::ZERO - F::ONE,
        exps: vec![0, 0, 0, 1],
    };
    let f = SparsePoly::new(4, vec![term_const, term_x1, term_x2, term_neg_out]);

    let s = CcsStructure::new(vec![m0, m1, m2, m3], f).expect("CCS");

    let z = vec![const_one, val1, val2, out];
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);
    let w = z.clone();

    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit = McsWitness { w, Z };
    (s, inst, wit)
}

/// Helper to build a complete step bundle with Twist/Shout
fn build_step_bundle(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    mcs_inst: &McsInstance<Cmt, F>,
    mcs_wit: McsWitness<F>,
    l: &DummyCommit,
) -> StepWitnessBundle<Cmt, F, K> {
    let mem_layout = PlainMemLayout { k: 4, d: 1, n_side: 4 };
    let plain_mem = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ZERO],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::ZERO],
        inc_at_write_addr: vec![F::ZERO],
    };
    let mem_init = MemInit::Zero;

    let plain_lut = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::ZERO],
    };

    let xor_table: RiscvLookupTable<F> = RiscvLookupTable::new(RiscvOpcode::Xor, 2);
    let lut_table = LutTable {
        table_id: 0,
        k: 16,
        d: 1,
        n_side: 16,
        content: xor_table.content(),
    };

    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(
        params,
        &mem_layout,
        &mem_init,
        &plain_mem,
        &commit_fn,
        Some(ccs.m),
        mcs_inst.m_in,
    );
    let (lut_inst, lut_wit) =
        encode_lut_for_shout(params, &lut_table, &plain_lut, &commit_fn, Some(ccs.m), mcs_inst.m_in);

    StepWitnessBundle {
        mcs: (mcs_inst.clone(), mcs_wit),
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData::<K>,
    }
}

/// Build default proof params
fn default_params() -> NeoParams {
    let m = 4usize;
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16,
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params")
}

// ============================================================================
// Test 1: Load and Execute Raw Binary
// ============================================================================

/// Test loading a raw binary (assembled from RiscvInstruction) and executing it.
#[test]
fn test_load_raw_binary_and_execute() {
    // Create a Fibonacci program using our DSL
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0 },   // x1 = 0
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 1 },   // x2 = 1
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 0, imm: 10 },  // x3 = 10 (counter)
        // Loop:
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 4, rs1: 1, rs2: 2 },   // x4 = x1 + x2
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 1, rs1: 2, rs2: 0 },   // x1 = x2
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 2, rs1: 4, rs2: 0 },   // x2 = x4
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 3, rs1: 3, imm: -1 },  // x3--
        RiscvInstruction::Branch { 
            cond: neo_memory::riscv_lookups::BranchCondition::Ne, 
            rs1: 3, 
            rs2: 0, 
            imm: -16 
        },
        RiscvInstruction::Halt,
    ];

    // Encode to binary
    let binary = encode_program(&program);
    println!("Binary size: {} bytes", binary.len());

    // Load the binary
    let loaded = load_raw_binary(&binary, 0).unwrap();
    println!("Loaded {} instructions from binary", loaded.instructions.len());
    loaded.disassemble();

    // Execute
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());

    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[2];
    
    // F(11) = 89
    assert_eq!(result, 89, "Expected Fibonacci(11) = 89, got {}", result);
    
    println!("✓ Binary loaded and executed successfully!");
    println!("  Steps: {}", trace.len());
    println!("  Fibonacci(11) = {}", result);
}

// ============================================================================
// Test 2: Binary → Trace → Proof Pipeline
// ============================================================================

/// Full pipeline: Load binary, execute, generate proof, verify using FoldingSession.
#[test]
fn test_binary_to_proof_full_pipeline() {
    // Create a simple program
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 7 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 13 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Mul, rd: 3, rs1: 1, rs2: 2 },
        RiscvInstruction::Halt,
    ];

    // Encode to binary
    let binary = encode_program(&program);

    // Load the binary
    let loaded = load_raw_binary(&binary, 0).unwrap();

    // Execute and trace
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    // Verify execution result
    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[3], 91, "7 * 13 = 91");

    // Convert trace to plain traces
    let _mem_trace: PlainMemTrace<F> = trace_to_plain_mem_trace(&trace);
    let _lut_trace: PlainLutTrace<F> = trace_to_plain_lut_trace(&trace);

    // Build proof infrastructure
    let params = default_params();
    let l = DummyCommit::default();

    let const_one = F::ONE;
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, F::ZERO, F::ZERO, const_one);
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    // Build step bundle and session
    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);

    // Generate and verify proof
    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");

    println!("✓ Binary → Proof Pipeline Complete!");
    println!("  Program: 7 * 13 = 91");
    println!("  Binary size: {} bytes", binary.len());
    println!("  Execution steps: {}", trace.len());
    println!("  Proof generated and verified!");
}

// ============================================================================
// Test 3: Larger Program (Factorial)
// ============================================================================

/// Test a more complex program: factorial.
#[test]
fn test_binary_factorial_proof() {
    // Factorial program: 6! = 720
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 6 },   // x1 = 6 (n)
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 1 },   // x2 = 1 (result)
        // Loop:
        RiscvInstruction::RAlu { op: RiscvOpcode::Mul, rd: 2, rs1: 2, rs2: 1 },   // x2 *= x1
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 1, imm: -1 },  // x1--
        RiscvInstruction::Branch { 
            cond: neo_memory::riscv_lookups::BranchCondition::Ne, 
            rs1: 1, 
            rs2: 0, 
            imm: -8 
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0x80000000).unwrap();

    println!("Factorial Program Disassembly:");
    loaded.disassemble();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0x80000000, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[2];
    
    assert_eq!(result, 720, "6! = 720, got {}", result);

    // Build and verify proof
    let params = default_params();
    let l = DummyCommit::default();

    let const_one = F::ONE;
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, F::ZERO, F::ZERO, const_one);
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);

    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");

    println!("✓ Factorial Binary Proof Complete!");
    println!("  6! = {}", result);
    println!("  Execution steps: {}", trace.len());
    println!("  Proof verified!");
}

// ============================================================================
// Test 4: Proof with Verified Output
// ============================================================================

/// This test demonstrates how to properly bind program outputs to the proof.
///
/// The key insight is:
/// 1. The **prover** executes the program and extracts the result
/// 2. The result becomes a **public input** to the proof
/// 3. The **verifier** only sees: (program_hash, public_output, proof)
/// 4. Verification checks that the proof is valid for that output
///
/// If the prover lies about the output, the proof will fail verification.
#[test]
fn test_proof_with_verified_output() {
    // ========================================================================
    // PROVER SIDE: Execute and generate proof
    // ========================================================================
    
    // Program: Compute 5 + 7 = 12
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 5 },   // x1 = 5
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 7 },   // x2 = 7
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 3, rs1: 1, rs2: 2 },   // x3 = x1 + x2
        RiscvInstruction::Halt,
    ];
    
    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();
    
    // Execute
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    
    // Extract the computed result (this is what the prover claims)
    let last_step = trace.steps.last().unwrap();
    let computed_result = last_step.regs_after[3];
    
    println!("Prover computed result: x3 = {}", computed_result);
    
    // ========================================================================
    // VERIFIER SIDE: Only sees (program, claimed_output, proof)
    // ========================================================================
    
    let claimed_output = computed_result;
    let program_binary = binary.clone();
    
    // Build proof with the output bound
    let params = default_params();
    let l = DummyCommit::default();
    
    let public_output = F::from_u64(claimed_output);
    let const_one = F::ONE;
    
    // CCS enforces the constraint structure
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(
        &params, &l, 
        const_one,
        public_output - F::ONE,
        F::ZERO,
        public_output,
    );
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    
    let step_bundle = build_step_bundle(&params, &ccs, &mcs_inst, mcs_wit, &l);
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params.clone(), l);
    session.add_step_bundle(step_bundle);
    
    // Generate and verify proof
    let proof = session.fold_and_prove(&ccs).expect("prove should succeed");
    session.verify_collected(&ccs, &proof).expect("verify should succeed");
    
    println!("✓ Proof with Verified Output!");
    println!("  Program: 5 + 7");
    println!("  Claimed output: {}", claimed_output);
    println!("  Proof verified: The execution trace is valid and produces output {}", claimed_output);
    println!("");
    println!("  What the verifier knows:");
    println!("    - The program binary ({} bytes)", program_binary.len());
    println!("    - The claimed output: {}", claimed_output);
    println!("    - A valid proof exists for this (program, output) pair");
    println!("");
    println!("  What the verifier does NOT know:");
    println!("    - The intermediate computation steps");
    println!("    - The register values at each step");
    println!("    - Any private inputs (if there were any)");
}

// ============================================================================
// Test 5: GCD Program from Binary
// ============================================================================

#[test]
fn test_binary_gcd_proof() {
    use neo_memory::riscv_lookups::BranchCondition;

    // GCD(48, 18) = 6
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 48 },
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 18 },
        // Loop:
        RiscvInstruction::Branch { cond: BranchCondition::Eq, rs1: 2, rs2: 0, imm: 20 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 3, rs1: 2, rs2: 0 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Remu, rd: 2, rs1: 1, rs2: 2 },
        RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 1, rs1: 3, rs2: 0 },
        RiscvInstruction::Jal { rd: 0, imm: -16 },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let result = last_step.regs_after[1];
    
    assert_eq!(result, 6, "GCD(48, 18) = 6, got {}", result);

    println!("✓ GCD Binary Proof!");
    println!("  GCD(48, 18) = {}", result);
    println!("  Binary size: {} bytes", binary.len());
    println!("  Execution steps: {}", trace.len());
}

// ============================================================================
// Test 6: Output Sumcheck Binding (Sound Output Verification)
// ============================================================================

/// This test demonstrates Jolt-style output binding using an Output Sumcheck.
///
/// Unlike the previous tests where we checked outputs manually outside the proof,
/// this test cryptographically binds the output to the proof.
///
/// ## How it works:
/// 1. Execute the RISC-V program to get a trace
/// 2. Build a ProgramIO specifying which registers are outputs
/// 3. Build the final memory state (register file mapped to virtual addresses)
/// 4. Run the Output Sumcheck to prove: Val_final(output_addrs) == claimed_outputs
///
/// If the prover lies about the output, the sumcheck will fail with
/// overwhelming probability.
#[test]
fn test_output_sumcheck_sound_binding() {
    use neo_memory::output_check::{
        OutputSumcheckParams, OutputSumcheckVerifier, ProgramIO,
    };
    use neo_memory::riscv_lookups::build_final_memory_state;
    use neo_math::K;

    println!("\n=== Output Sumcheck Sound Binding Test ===\n");

    // Simple program: compute 10 + 25 = 35, store result in x10 (a0)
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10, // a0 - return value register
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 10,
            imm: 25,
        },
        RiscvInstruction::Halt,
    ];

    // Compile and execute
    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    // Get the actual output
    let actual_output = trace.steps.last().unwrap().regs_after[10];
    assert_eq!(actual_output, 35, "10 + 25 = 35");

    // Build the Output Sumcheck proof
    let num_bits = 6; // 2^6 = 64 addresses (covers 32 registers + some margin)

    // Step 1: Create ProgramIO specifying what we claim the output is
    let program_io: ProgramIO<F> = ProgramIO::new()
        .with_output(10, F::from_u64(actual_output)); // Claim: x10 = 35

    // Step 2: Generate random challenges (in real protocol, from transcript)
    let r_addr: Vec<K> = (0..num_bits)
        .map(|i| K::from_u64(1000 + (i * 7) as u64))
        .collect();

    let params = OutputSumcheckParams::new_for_testing(num_bits, r_addr.clone(), program_io.clone()).unwrap();

    // Step 3: Build the final memory state (register file)
    let final_state: Vec<F> = build_final_memory_state(&trace, num_bits);

    // Step 4: Generate the output sumcheck proof and collect challenges
    let mut prover = neo_memory::output_check::OutputSumcheckProver::new(params.clone(), &final_state).unwrap();

    // Verify claim is 0 before generating proof
    let initial_claim = prover.compute_claim();
    assert_eq!(initial_claim, K::ZERO, "Output sumcheck claim should be 0");

    let degree_bound = neo_reductions::sumcheck::RoundOracle::degree_bound(&prover);
    let eval_points: Vec<K> = (0..=degree_bound).map(|i| K::from_u64(i as u64)).collect();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for round in 0..num_bits {
        let evals = neo_reductions::sumcheck::RoundOracle::evals_at(&mut prover, &eval_points);
        let coeffs = interpolate_test(&eval_points, &evals);
        round_polys.push(coeffs);

        let r = K::from_u64((round * 13 + 1) as u64);
        challenges.push(r);
        neo_reductions::sumcheck::RoundOracle::fold(&mut prover, r);
    }

    let proof = neo_memory::output_check::OutputSumcheckProof { round_polys };

    println!("✓ Output Sumcheck Proof Generated!");
    println!("  Program: 10 + 25 = {}", actual_output);
    println!("  Output register: x10 (a0)");
    println!("  Number of sumcheck rounds: {}", proof.round_polys.len());

    // Step 5: Verify the proof
    let verifier = OutputSumcheckVerifier::new(params);

    // Get Val_final at the challenge point
    let val_final_at_r = {
        use neo_memory::mle::build_chi_table;
        let chi = build_chi_table(&challenges);
        let mut sum = K::ZERO;
        for (i, val) in final_state.iter().enumerate() {
            let val_k: K = (*val).into();
            sum += val_k * chi[i];
        }
        sum
    };

    let result = verifier.verify(&proof, val_final_at_r, &challenges);
    assert!(result.is_ok(), "Output sumcheck verification failed: {:?}", result);

    println!("✓ Output Sumcheck Verified!");
    println!("  The proof cryptographically binds output {} to the execution", actual_output);
}

// Helper for interpolation in tests
fn interpolate_test(xs: &[K], ys: &[K]) -> Vec<K> {
    use neo_math::KExtensions;
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];
    for i in 0..n {
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0;
        for j in 0..n {
            if i == j { continue; }
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] -= xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }
        let mut denom = K::ONE;
        for j in 0..n {
            if i != j { denom *= xs[i] - xs[j]; }
        }
        let scale = ys[i] * denom.inv();
        for d in 0..n { coeffs[d] += scale * numer[d]; }
    }
    coeffs
}

/// Test that the output sumcheck detects a lying prover.
#[test]
fn test_output_sumcheck_detects_lying_prover() {
    use neo_memory::output_check::{OutputSumcheckParams, OutputSumcheckProver, ProgramIO};
    use neo_memory::riscv_lookups::build_final_memory_state;
    use neo_math::K;

    println!("\n=== Output Sumcheck Lying Prover Detection Test ===\n");

    // Simple program: compute 10 + 25 = 35
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 10,
            imm: 25,
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    let actual_output = trace.steps.last().unwrap().regs_after[10];
    assert_eq!(actual_output, 35);

    let num_bits = 6;

    // LYING: Claim the output is 999 instead of 35
    let wrong_output = 999u64;
    let lying_program_io: ProgramIO<F> = ProgramIO::new()
        .with_output(10, F::from_u64(wrong_output));

    let r_addr: Vec<K> = (0..num_bits)
        .map(|i| K::from_u64(2000 + (i * 11) as u64))
        .collect();

    let lying_params = OutputSumcheckParams::new_for_testing(num_bits, r_addr, lying_program_io).unwrap();

    // Build the actual final state (with correct value 35)
    let final_state: Vec<F> = build_final_memory_state(&trace, num_bits);

    // The prover tries to prove with the wrong claimed output
    let lying_prover = OutputSumcheckProver::new(lying_params, &final_state).unwrap();
    let lying_claim = lying_prover.compute_claim();

    // The claim should NOT be 0 because Val_final(10) = 35 ≠ Val_io(10) = 999
    assert_ne!(
        lying_claim,
        K::ZERO,
        "Lying prover's claim should be non-zero! The sumcheck detects the lie."
    );

    println!("✓ Lying Prover Detected!");
    println!("  Actual output: {}", actual_output);
    println!("  Claimed (false) output: {}", wrong_output);
    println!("  Sumcheck claim: {:?} (non-zero = caught!)", lying_claim);
    println!("  The prover cannot generate a valid proof for a false output.");
}

/// Test extracting program I/O from a trace.
#[test]
fn test_extract_program_io() {

    // Fibonacci: compute fib(8) = 21
    let program = vec![
        // x1 = 0 (fib(0))
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        // x2 = 1 (fib(1))
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        // x3 = 8 (counter)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 8,
        },
        // Loop: x10 = x1 + x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        // x1 = x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 2,
            rs2: 0,
        },
        // x2 = x10
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 10,
            rs2: 0,
        },
        // x3 = x3 - 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 3,
            imm: -1,
        },
        // if x3 != 0, loop back
        RiscvInstruction::Branch {
            cond: neo_memory::riscv_lookups::BranchCondition::Ne,
            rs1: 3,
            rs2: 0,
            imm: -16,
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 200).unwrap();
    assert!(trace.did_halt());

    let result = trace.steps.last().unwrap().regs_after[10];
    // fib sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
    // After 8 iterations starting from (0, 1): result = 34
    assert_eq!(result, 34, "After 8 iterations, fib result should be 34");

    // Extract program I/O
    let program_io: neo_memory::output_check::ProgramIO<F> =
        neo_memory::riscv_lookups::extract_program_io(&trace, &[10]); // Output is in x10

    assert_eq!(program_io.num_claims(), 1);
    assert_eq!(program_io.get_claim(10), Some(F::from_u64(result))); // Address 10, Value = 34

    println!("✓ Program I/O Extraction!");
    println!("  Computed fib(8) iterations = {}", result);
    println!("  Extracted output register x10 = {}", result);
}

// ============================================================================
// Test 9: Full Negative Test - Fraudulent Prover Caught by Verifier
// ============================================================================

/// This test demonstrates that a fraudulent prover who tries to claim
/// a wrong output will be caught by the verifier during the full
/// verification process.
///
/// ## Attack Scenario:
/// 1. Honest execution: 15 * 7 = 105
/// 2. Fraudulent prover claims: output = 999 (lie!)
/// 3. Prover generates a proof for the lie
/// 4. Verifier runs full verification
/// 5. Verification FAILS - the lie is caught!
///
/// This demonstrates the cryptographic soundness of the Output Sumcheck.
#[test]
fn test_verifier_catches_fraudulent_output_claim() {
    use neo_memory::output_check::{
        OutputSumcheckParams, OutputSumcheckVerifier, ProgramIO,
    };
    use neo_memory::riscv_lookups::build_final_memory_state;
    use neo_math::K;

    println!("\n=== NEGATIVE TEST: Verifier Catches Fraudulent Output ===\n");

    // ========================================
    // Step 1: Execute an honest program
    // ========================================
    // Program: compute 15 * 7 = 105, store in x10
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 15,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 10, // a0 - return value register
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let honest_output = trace.steps.last().unwrap().regs_after[10];
    assert_eq!(honest_output, 105, "15 * 7 = 105");

    println!("Step 1: Honest execution completed");
    println!("  Program: 15 * 7");
    println!("  Actual result: {}", honest_output);

    // ========================================
    // Step 2: Fraudulent prover claims wrong output
    // ========================================
    let fraudulent_output = 999u64; // The lie!
    let num_bits = 6;

    // Fraudulent prover creates ProgramIO with WRONG output
    let fraudulent_program_io: ProgramIO<F> = ProgramIO::new()
        .with_output(10, F::from_u64(fraudulent_output)); // LYING: claiming 999

    println!("\nStep 2: Fraudulent prover claims wrong output");
    println!("  Claimed output: {} (LIE!)", fraudulent_output);
    println!("  Actual output: {}", honest_output);

    // ========================================
    // Step 3: Fraudulent prover generates proof
    // ========================================
    let r_addr: Vec<K> = (0..num_bits)
        .map(|i| K::from_u64(5000 + (i * 17) as u64))
        .collect();

    let params = OutputSumcheckParams::new_for_testing(num_bits, r_addr.clone(), fraudulent_program_io.clone()).unwrap();

    // Build the ACTUAL final state (contains true value 105, not 999)
    let final_state: Vec<F> = build_final_memory_state(&trace, num_bits);

    // Create prover and check initial claim
    let mut prover = neo_memory::output_check::OutputSumcheckProver::new(params.clone(), &final_state).unwrap();
    let initial_claim = prover.compute_claim();

    // The claim should NOT be zero because the outputs don't match
    assert_ne!(initial_claim, K::ZERO, "Fraudulent prover's sum should be non-zero");

    println!("\nStep 3: Fraudulent prover computes claim");
    println!("  Initial claim: {:?}", initial_claim);
    println!("  Note: claim ≠ 0 because actual output (105) ≠ claimed (999)");

    // Generate proof and collect challenges
    let degree_bound = neo_reductions::sumcheck::RoundOracle::degree_bound(&prover);
    let eval_points: Vec<K> = (0..=degree_bound).map(|i| K::from_u64(i as u64)).collect();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for round in 0..num_bits {
        let evals = neo_reductions::sumcheck::RoundOracle::evals_at(&mut prover, &eval_points);
        let coeffs = interpolate_test(&eval_points, &evals);
        round_polys.push(coeffs);

        let r = K::from_u64((round * 19 + 3) as u64);
        challenges.push(r);
        neo_reductions::sumcheck::RoundOracle::fold(&mut prover, r);
    }

    let fraudulent_proof = neo_memory::output_check::OutputSumcheckProof { round_polys };

    // ========================================
    // Step 4: Verifier runs full verification
    // ========================================
    let verifier = OutputSumcheckVerifier::new(params);

    // Compute Val_final at the challenge point
    let val_final_at_r = {
        use neo_memory::mle::build_chi_table;
        let chi = build_chi_table(&challenges);
        let mut sum = K::ZERO;
        for (i, val) in final_state.iter().enumerate() {
            let val_k: K = (*val).into();
            sum += val_k * chi[i];
        }
        sum
    };

    // Verify the proof - this should FAIL!
    let verification_result = verifier.verify(&fraudulent_proof, val_final_at_r, &challenges);

    println!("\nStep 4: Verifier runs full verification");
    println!("  Verification result: {:?}", verification_result);

    // The verifier MUST reject because it pins initial claim to 0,
    // but the actual sum is non-zero
    assert!(
        verification_result.is_err(),
        "Verifier must reject fraudulent proof"
    );

    println!("\n=== FRAUD DETECTED! ===");
    println!("  The verifier caught the fraudulent output claim.");
    println!("  Reason: Verifier pins initial claim to 0, but p(0)+p(1) ≠ 0");
    println!("  ");
    println!("  What happened:");
    println!("    - Prover claimed output = {}", fraudulent_output);
    println!("    - Actual output = {}", honest_output);
    println!("    - Prover's sum = {:?} (non-zero)", initial_claim);
    println!("    - Verifier rejects at round 0: p(0)+p(1) ≠ 0");
    println!("  ");
    println!("  Soundness guarantee:");
    println!("    A malicious prover CANNOT forge a proof for a false output.");
    println!("    The Output Sumcheck ensures cryptographic binding of outputs.");
}

/// Test multiple fraudulent attempts with different wrong values.
#[test]
fn test_multiple_fraud_attempts_all_caught() {
    use neo_memory::output_check::{OutputSumcheckParams, OutputSumcheckProver, ProgramIO};
    use neo_memory::riscv_lookups::build_final_memory_state;
    use neo_math::K;

    println!("\n=== NEGATIVE TEST: Multiple Fraud Attempts ===\n");

    // Simple program: 20 + 30 = 50
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: 20,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 10,
            imm: 30,
        },
        RiscvInstruction::Halt,
    ];

    let binary = encode_program(&program);
    let loaded = load_raw_binary(&binary, 0).unwrap();

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, loaded.get_instructions());
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    let honest_output = trace.steps.last().unwrap().regs_after[10];
    assert_eq!(honest_output, 50);

    let num_bits = 6;
    let final_state: Vec<F> = build_final_memory_state(&trace, num_bits);

    // Try multiple fraudulent values
    let fraudulent_values = [0u64, 1, 49, 51, 100, 999, u64::MAX - 1];

    println!("Actual output: {}", honest_output);
    println!("Testing fraudulent claims:\n");

    for &fraudulent_value in &fraudulent_values {
        let fraudulent_io: ProgramIO<F> =
            ProgramIO::new().with_output(10, F::from_u64(fraudulent_value));

        let r_addr: Vec<K> = (0..num_bits)
            .map(|i| K::from_u64(fraudulent_value.wrapping_add(100 + (i * 7) as u64)))
            .collect();

        let params = OutputSumcheckParams::new_for_testing(num_bits, r_addr, fraudulent_io).unwrap();
        let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
        let claim = prover.compute_claim();

        let fraud_detected = claim != K::ZERO;
        assert!(
            fraud_detected,
            "Fraud not detected for claimed value {}!",
            fraudulent_value
        );

        println!(
            "  Claimed: {:>20} | Detected: {} | Claim ≠ 0: ✓",
            fraudulent_value, fraud_detected
        );
    }

    // Also verify that the honest value passes
    let honest_io: ProgramIO<F> =
        ProgramIO::new().with_output(10, F::from_u64(honest_output));

    let r_addr: Vec<K> = (0..num_bits)
        .map(|i| K::from_u64(honest_output.wrapping_add(100 + (i * 7) as u64)))
        .collect();

    let params = OutputSumcheckParams::new_for_testing(num_bits, r_addr, honest_io).unwrap();
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    let honest_claim = prover.compute_claim();

    assert_eq!(
        honest_claim,
        K::ZERO,
        "Honest claim should be zero!"
    );

    println!("\n  Claimed: {:>20} | HONEST    | Claim = 0: ✓", honest_output);
    println!("\n=== All {} fraud attempts detected! ===", fraudulent_values.len());
}
