use std::time::{Duration, Instant};

use neo_fold::riscv_shard::Rv32B1;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_fold::shard::ShardProof;
use neo_ccs::MeInstance;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, Rv32TraceCcsLayout};

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test perf -- --ignored --nocapture compare_single_mixed_metrics_nightstream_only`"]
fn compare_single_mixed_metrics_nightstream_only() {
    let instruction_label = "Mixed sequence (ADD/AND/OR/XOR/SLT/SLTU/SLL/SRL/SRA/BNE)";

    let ns_program = mixed_instruction_sequence();
    let ns_program_bytes = encode_program(&ns_program);
    let ns_chunk_size = ns_program.len();
    let ns_max_steps = ns_program.len();
    let ns_ram_bytes = 4usize;

    let ns_total_start = Instant::now();
    let mut ns_run = Rv32B1::from_rom(/*program_base=*/ 0, &ns_program_bytes)
        .chunk_size(ns_chunk_size)
        .ram_bytes(ns_ram_bytes)
        .max_steps(ns_max_steps)
        .prove()
        .expect("Nightstream prove");

    let ns_constraints = ns_run.ccs_num_constraints();
    let ns_witness_cols = ns_run.ccs_num_variables();
    let ns_constraints_padded_pow2 = ns_constraints.next_power_of_two();
    let ns_witness_cols_padded_pow2 = ns_witness_cols.next_power_of_two();
    let ns_fold_count = ns_run.fold_count();
    let ns_trace_len = ns_run.riscv_trace_len().expect("Nightstream trace length");
    let ns_shout_lookups = ns_run
        .shout_lookup_count()
        .expect("Nightstream shout lookup count");
    let ns_step0 = ns_run
        .steps_public()
        .first()
        .cloned()
        .expect("Nightstream collected steps");
    let ns_m_in = ns_step0.mcs_inst.m_in;
    let ns_witness_private = ns_witness_cols.saturating_sub(ns_m_in);
    let ns_lut_instances = ns_step0.lut_insts.len();
    let ns_mem_instances = ns_step0.mem_insts.len();

    ns_run.verify().expect("Nightstream verify");
    let ns_prove_time = ns_run.prove_duration();
    let ns_verify_time = ns_run
        .verify_duration()
        .expect("Nightstream verify duration");
    let ns_total_duration = ns_total_start.elapsed();

    println!();
    println!("Instruction under test: {instruction_label}");
    println!();
    println!("**Nightstream (Neo RV32 B1)**");
    println!(
        "- CCS: n={} constraints (padded_pow2_n={}), m={} cols (padded_pow2_m={}) (m_in={} public, w={} private)",
        ns_constraints,
        ns_constraints_padded_pow2,
        ns_witness_cols,
        ns_witness_cols_padded_pow2,
        ns_m_in,
        ns_witness_private
    );
    println!(
        "- Trace: executed_steps={} (max_steps={}), fold_chunks={} (chunk_size={})",
        ns_trace_len, ns_max_steps, ns_fold_count, ns_chunk_size
    );
    println!(
        "- Sidecars: lut_instances={} mem_instances={} shout_lookups_used={}",
        ns_lut_instances, ns_mem_instances, ns_shout_lookups
    );
    println!(
        "- Time: prove={} verify={} total_end_to_end={}",
        fmt_duration(ns_prove_time),
        fmt_duration(ns_verify_time),
        fmt_duration(ns_total_duration)
    );
    println!();

    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Metric", "Nightstream");
    println!("{:<40} {:>18}", "", "(RV32 B1)");
    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Rows per step (raw)", ns_constraints);
    println!(
        "{:<40} {:>18}",
        "Rows per step (padded pow2)", ns_constraints_padded_pow2
    );
    println!(
        "{:<40} {:>18}",
        "Total rows in proof (padded)",
        ns_constraints_padded_pow2.saturating_mul(ns_fold_count)
    );
    println!(
        "{:<40} {:>18}",
        "Total rows (estimate, unpadded)",
        ns_constraints.saturating_mul(ns_trace_len)
    );
    println!("{:<40} {:>18}", "Cols / vars (raw)", ns_witness_cols);
    println!(
        "{:<40} {:>18}",
        "Cols / vars (padded pow2)", ns_witness_cols_padded_pow2
    );
    println!("{:<40} {:>18}", "Public inputs (m_in)", ns_m_in);
    println!(
        "{:<40} {:>18}",
        "Trace len (unpadded)",
        format!("{} steps", ns_trace_len)
    );
    println!("{:<40} {:>18}", "Lookup tables", format!("{} Shout", ns_lut_instances));
    println!("{:<40} {:>18}", "Lookups used", ns_shout_lookups);
    println!("{:<40} {:>18}", "Prove time", fmt_duration(ns_prove_time));
    println!("{:<40} {:>18}", "Verify time", fmt_duration(ns_verify_time));
    println!("{:-<80}", "");
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs_f64() < 1.0 {
        format!("{:.3}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OpeningSurfaceBuckets {
    core_ccs: usize,
    sidecars: usize,
    claim_reduction_linkage: usize,
    pcs_open: usize,
}

impl OpeningSurfaceBuckets {
    fn total(self) -> usize {
        self.core_ccs + self.sidecars + self.claim_reduction_linkage + self.pcs_open
    }
}

fn sum_y_scalars<C, FF, KK>(claims: &[MeInstance<C, FF, KK>]) -> usize {
    claims.iter().map(|me| me.y_scalars.len()).sum()
}

fn opening_surface_from_shard_proof(proof: &ShardProof) -> OpeningSurfaceBuckets {
    let mut buckets = OpeningSurfaceBuckets::default();
    for step in &proof.steps {
        buckets.core_ccs += sum_y_scalars(&step.fold.ccs_out);

        buckets.sidecars += sum_y_scalars(&step.mem.shout_me_claims_time);
        buckets.sidecars += sum_y_scalars(&step.mem.twist_me_claims_time);
        buckets.sidecars += sum_y_scalars(&step.mem.val_me_claims);

        buckets.claim_reduction_linkage += sum_y_scalars(&step.mem.wb_me_claims);
        buckets.claim_reduction_linkage += sum_y_scalars(&step.mem.wp_me_claims);
        buckets.claim_reduction_linkage += step.batched_time.claimed_sums.len();

        buckets.pcs_open += step.fold.dec_children.len();
        buckets.pcs_open += step.val_fold.iter().map(|p| p.dec_children.len()).sum::<usize>();
        buckets.pcs_open += step
            .twist_time_fold
            .iter()
            .map(|p| p.dec_children.len())
            .sum::<usize>();
        buckets.pcs_open += step
            .shout_time_fold
            .iter()
            .map(|p| p.dec_children.len())
            .sum::<usize>();
        buckets.pcs_open += step.wb_fold.iter().map(|p| p.dec_children.len()).sum::<usize>();
        buckets.pcs_open += step.wp_fold.iter().map(|p| p.dec_children.len()).sum::<usize>();
    }
    buckets
}

fn opening_surface_from_rv32_b1_run(run: &neo_fold::riscv_shard::Rv32B1Run) -> OpeningSurfaceBuckets {
    let mut buckets = opening_surface_from_shard_proof(&run.proof().main);
    buckets.sidecars += sum_y_scalars(&run.proof().decode_plumbing.me_out);
    buckets.sidecars += sum_y_scalars(&run.proof().semantics.me_out);
    if let Some(rv32m) = &run.proof().rv32m {
        for chunk in rv32m {
            buckets.sidecars += sum_y_scalars(&chunk.me_out);
            buckets.pcs_open += chunk.me_out.len();
        }
    }
    buckets.pcs_open += run.proof().decode_plumbing.me_out.len();
    buckets.pcs_open += run.proof().semantics.me_out.len();
    buckets
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn mixed_instruction_sequence() -> Vec<RiscvInstruction> {
    vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ]
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_single_n_mixed_ops"]
fn debug_trace_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);
    assert!(n > 0);
    assert!(chunk_rows > 0);

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove()
        .expect("trace prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace verify");
    let verify_time = run.verify_duration().expect("trace verify duration");
    let total_time = total_start.elapsed();
    let phases = run.prove_phase_durations();

    println!(
        "TRACE n={} chunk_rows={} ccs_n={} ccs_m={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={} phases(setup={}, chunk_commit={}, fold={})",
        n,
        chunk_rows,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        run.trace_len(),
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        fmt_duration(phases.setup),
        fmt_duration(phases.chunk_build_commit),
        fmt_duration(phases.fold_and_prove),
    );
    let openings = opening_surface_from_shard_proof(run.proof());
    println!(
        "TRACE_OPENINGS core_ccs={} sidecars={} claim_reduction_linkage={} pcs_open={} total={}",
        openings.core_ccs,
        openings.sidecars,
        openings.claim_reduction_linkage,
        openings.pcs_open,
        openings.total()
    );
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_chunked_single_n_mixed_ops"]
fn debug_chunked_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    assert!(n > 0);

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32B1::from_rom(0, &program_bytes)
        .chunk_size(steps)
        .ram_bytes(4)
        .max_steps(steps)
        .prove()
        .expect("chunked prove");
    let prove_time = run.prove_duration();
    run.verify().expect("chunked verify");
    let verify_time = run.verify_duration().expect("chunked verify duration");
    let total_time = total_start.elapsed();
    let trace_len = run.riscv_trace_len().expect("trace len");
    let phases = run.prove_phase_durations();

    println!(
        "CHUNKED n={} ccs_n={} ccs_m={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={} phases(setup={}, build_commit={}, fold={})",
        n,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        trace_len,
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        fmt_duration(phases.setup),
        fmt_duration(phases.build_commit),
        fmt_duration(phases.fold_and_prove),
    );
    let openings = opening_surface_from_rv32_b1_run(&run);
    println!(
        "CHUNKED_OPENINGS core_ccs={} sidecars={} claim_reduction_linkage={} pcs_open={} total={}",
        openings.core_ccs,
        openings.sidecars,
        openings.claim_reduction_linkage,
        openings.pcs_open,
        openings.total()
    );
}

#[test]
#[ignore = "perf-style report hook: cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_core_rows_per_cycle_equiv"]
fn debug_trace_core_rows_per_cycle_equiv() {
    let t = env_usize("NS_DEBUG_T", 257);
    let layout = Rv32TraceCcsLayout::new(t).expect("trace layout");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace core ccs");
    println!(
        "TRACE_CORE t={} trace_width={} core_ccs_n={} rows_per_cycle={:.3}",
        t,
        layout.trace.cols,
        ccs.n,
        ccs.n as f64 / t as f64
    );
}

#[test]
#[ignore = "W0 snapshot: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture report_track_a_w0_w1_snapshot"]
fn report_track_a_w0_w1_snapshot() {
    let n = env_usize("NS_DEBUG_N", 256);
    assert!(n > 0);
    let chunk_rows = n + 1;

    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove()
        .expect("trace prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace verify");
    let verify_time = run.verify_duration().expect("trace verify duration");
    let total_time = total_start.elapsed();
    let openings = opening_surface_from_shard_proof(run.proof());

    let layout = Rv32TraceCcsLayout::new(steps).expect("trace layout");
    let core_ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace core ccs");
    let rows_per_cycle = core_ccs.n as f64 / steps as f64;

    // W0 lock values from spec section 7.
    let baseline_trace_width = 160usize;
    let baseline_rows = 425usize;
    let post_w1_trace_width = 148usize;
    let post_w1_rows = 399usize;

    println!(
        "W0_W1_LOCK baseline(trace_width={},rows_per_cycle={}) post_w1(trace_width={},rows_per_cycle={})",
        baseline_trace_width, baseline_rows, post_w1_trace_width, post_w1_rows
    );
    println!(
        "TRACK_A_MEASURED n={} trace_width={} core_ccs_n={} rows_per_cycle={:.3} ccs_n={} ccs_m={} prove={} verify={} total={} openings(core_ccs={},sidecars={},claim_reduction_linkage={},pcs_open={},total={})",
        n,
        layout.trace.cols,
        core_ccs.n,
        rows_per_cycle,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        openings.core_ccs,
        openings.sidecars,
        openings.claim_reduction_linkage,
        openings.pcs_open,
        openings.total()
    );
    println!(
        "TRACK_A_USED_SETS memory_ids={:?} shout_table_ids={:?}",
        run.used_memory_ids(),
        run.used_shout_table_ids()
    );
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_vs_chunked_single_n_mixed_ops"]
fn debug_trace_vs_chunked_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);
    assert!(n > 0);
    assert!(chunk_rows > 0);
    let base = mixed_instruction_sequence();
    assert_eq!(base.len(), 10);

    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let chunk_total_start = Instant::now();
    let mut chunk_run = Rv32B1::from_rom(0, &program_bytes)
        .chunk_size(steps)
        .ram_bytes(4)
        .max_steps(steps)
        .prove()
        .expect("chunked prove (mixed)");
    let chunk_prove = chunk_run.prove_duration();
    let chunk_phases = chunk_run.prove_phase_durations();
    chunk_run.verify().expect("chunked verify (mixed)");
    let chunk_verify = chunk_run
        .verify_duration()
        .expect("chunked verify duration");
    let chunk_total = chunk_total_start.elapsed();

    let trace_total_start = Instant::now();
    let trace_res = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove();
    match trace_res {
        Ok(mut trace_run) => {
            let trace_prove = trace_run.prove_duration();
            trace_run.verify().expect("trace verify (mixed)");
            let trace_verify = trace_run.verify_duration().expect("trace verify duration");
            let trace_total = trace_total_start.elapsed();
            let trace_phases = trace_run.prove_phase_durations();
            println!(
                "MIXED n={} TRACE(prove={}, verify={}, total={}, n_p2={}, m_p2={}, phases: setup={}, chunk_commit={}, fold={}) CHUNKED(prove={}, verify={}, total={}, n_p2={}, m_p2={}, phases: setup={}, build_commit={}, fold={}) ratio_prove={:.2}x",
                n,
                fmt_duration(trace_prove),
                fmt_duration(trace_verify),
                fmt_duration(trace_total),
                trace_run.ccs_num_constraints().next_power_of_two(),
                trace_run.ccs_num_variables().next_power_of_two(),
                fmt_duration(trace_phases.setup),
                fmt_duration(trace_phases.chunk_build_commit),
                fmt_duration(trace_phases.fold_and_prove),
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
                fmt_duration(chunk_phases.setup),
                fmt_duration(chunk_phases.build_commit),
                fmt_duration(chunk_phases.fold_and_prove),
                trace_prove.as_secs_f64() / chunk_prove.as_secs_f64(),
            );
        }
        Err(e) => {
            println!(
                "MIXED n={} TRACE(prove=ERROR:{}) CHUNKED(prove={}, verify={}, total={}, n_p2={}, m_p2={})",
                n,
                e,
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PerfSample {
    end_to_end: Duration,
    prove: Duration,
    verify: Duration,
    setup: Duration,
    build_commit: Duration,
    fold: Duration,
}

fn median_duration(values: &[Duration]) -> Duration {
    let mut nanos: Vec<u128> = values.iter().map(|d| d.as_nanos()).collect();
    nanos.sort_unstable();
    Duration::from_nanos(nanos[nanos.len() / 2] as u64)
}

fn spread_pct(values: &[Duration], median: Duration) -> f64 {
    if values.is_empty() || median.is_zero() {
        return 0.0;
    }
    let med = median.as_secs_f64();
    let max_abs = values
        .iter()
        .map(|v| (v.as_secs_f64() - med).abs())
        .fold(0.0f64, f64::max);
    (max_abs / med) * 100.0
}

fn build_mixed_program(n: usize) -> Vec<RiscvInstruction> {
    let base = mixed_instruction_sequence();
    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    program
}

fn run_trace_sample(program: &[RiscvInstruction]) -> PerfSample {
    let steps = program.len();
    let program_bytes = encode_program(program);
    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(steps)
        .prove()
        .expect("trace prove");
    let prove = run.prove_duration();
    let phases = run.prove_phase_durations();
    run.verify().expect("trace verify");
    let verify = run.verify_duration().expect("trace verify duration");
    PerfSample {
        end_to_end: total_start.elapsed(),
        prove,
        verify,
        setup: phases.setup,
        build_commit: phases.chunk_build_commit,
        fold: phases.fold_and_prove,
    }
}

fn run_chunked_sample(program: &[RiscvInstruction]) -> PerfSample {
    let steps = program.len();
    let program_bytes = encode_program(program);
    let total_start = Instant::now();
    let mut run = Rv32B1::from_rom(0, &program_bytes)
        .chunk_size(steps)
        .ram_bytes(4)
        .max_steps(steps)
        .prove()
        .expect("chunked prove");
    let prove = run.prove_duration();
    let phases = run.prove_phase_durations();
    run.verify().expect("chunked verify");
    let verify = run.verify_duration().expect("chunked verify duration");
    PerfSample {
        end_to_end: total_start.elapsed(),
        prove,
        verify,
        setup: phases.setup,
        build_commit: phases.build_commit,
        fold: phases.fold_and_prove,
    }
}

fn report_samples(label: &str, samples: &[PerfSample]) {
    let end_vals: Vec<Duration> = samples.iter().map(|s| s.end_to_end).collect();
    let prove_vals: Vec<Duration> = samples.iter().map(|s| s.prove).collect();
    let verify_vals: Vec<Duration> = samples.iter().map(|s| s.verify).collect();
    let setup_vals: Vec<Duration> = samples.iter().map(|s| s.setup).collect();
    let build_vals: Vec<Duration> = samples.iter().map(|s| s.build_commit).collect();
    let fold_vals: Vec<Duration> = samples.iter().map(|s| s.fold).collect();
    let prove_window_vals: Vec<Duration> = samples
        .iter()
        .map(|s| s.setup + s.build_commit + s.fold)
        .collect();

    let end_med = median_duration(&end_vals);
    let prove_med = median_duration(&prove_vals);
    let verify_med = median_duration(&verify_vals);
    let setup_med = median_duration(&setup_vals);
    let build_med = median_duration(&build_vals);
    let fold_med = median_duration(&fold_vals);
    let prove_window_med = median_duration(&prove_window_vals);

    println!(
        "{}: median(end={}, prove_api={}, prove_window={}, verify={}, setup={}, build_commit={}, fold={}) spread(end={:.2}%, prove_window={:.2}%, fold={:.2}%)",
        label,
        fmt_duration(end_med),
        fmt_duration(prove_med),
        fmt_duration(prove_window_med),
        fmt_duration(verify_med),
        fmt_duration(setup_med),
        fmt_duration(build_med),
        fmt_duration(fold_med),
        spread_pct(&end_vals, end_med),
        spread_pct(&prove_window_vals, prove_window_med),
        spread_pct(&fold_vals, fold_med),
    );
}

#[test]
#[ignore = "perf baseline report: cargo test -p neo-fold --release --test perf -- --ignored --nocapture report_trace_vs_chunked_medians"]
fn report_trace_vs_chunked_medians() {
    const RUNS: usize = 5;
    let cases = [
        ("mixed", 10usize, build_mixed_program(10)),
        ("mixed", 256usize, build_mixed_program(256)),
    ];

    for (kind, n, program) in cases {
        let mut trace_samples = Vec::with_capacity(RUNS);
        let mut chunked_samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            trace_samples.push(run_trace_sample(&program));
            chunked_samples.push(run_chunked_sample(&program));
        }
        println!("CASE kind={} n={} runs={}", kind, n, RUNS);
        report_samples("TRACE", &trace_samples);
        report_samples("CHUNKED", &chunked_samples);
    }
}
