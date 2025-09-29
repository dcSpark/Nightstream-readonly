//! Transcript parity harness (prover vs verifier) without touching library code.
//!
//! How it works:
//! - Runs a dedicated subtest that executes prover then verifier back-to-back.
//! - Enables `NEO_TRANSCRIPT_DUMP=1` and sets `NEO_TRANSCRIPT_TAG` to PROVER/VERIFIER
//!   so neo-transcript prints event dumps at each challenge/digest.
//! - The parent test spawns the subtest as a child process with `--nocapture`,
//!   captures its stderr, parses the event lines, and diffs the two streams.
//!
//! NOTE: This requires neo-transcript to be compiled with the `debug-log` feature.
//! If not enabled, this test will detect the absence of dumps and skip.

#![allow(deprecated)]

use std::process::{Command, Stdio};

use neo_spartan_bridge::{compress_me_to_lean_proof_with_pp, verify_lean_proof};
use neo_ccs::{MEInstance, MEWitness};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

/// Build the tiniest honest Ajtai-bound instance that Spartan can handle.
/// (m=1, balanced digits in {-1,0,1}, PP-backed binding)
fn tiny_honest_instance() -> (MEInstance, MEWitness, std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>) {
    let d = neo_math::ring::D;
    let m = 1usize;
    let kappa = 2usize;

    // Balanced small digits in {-1,0,1}
    let mut z_digits: Vec<i64> = Vec::with_capacity(d * m);
    for i in 0..(d * m) {
        z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 });
    }
    // z as field elements (for committing with PP)
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();

    // Public parameters + commitment c = L·z
    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([42u8; 32]);
    let pp = std::sync::Arc::new(neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup"));
    let c = neo_ajtai::commit(&pp, &z_f);

    // Legacy ME instance + witness
    let me = MEInstance {
        c_coords: c.data.clone(),
        y_outputs: vec![],           // Ajtai alone binds z
        r_point: vec![],
        base_b: 2,
        header_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0, u_len: 0,
    };
    let wit = MEWitness {
        z_digits,
        weight_vectors: vec![],
        ajtai_rows: None,            // PP-backed, no materialized rows
    };
    (me, wit, pp)
}

/// Event parsed from neo-transcript dump line
#[derive(Debug, Clone, PartialEq, Eq)]
struct EventLine {
    op: String,
    label: String,
    len: usize,
    st_prefix: [u64; 4],
}

fn parse_event_line(line: &str) -> Option<EventLine> {
    // Expected format:
    // [0000] op=append_message label="foo" len=3 st0..3=[1, 2, 3, 4]
    if !line.starts_with('[') || !line.contains(" op=") { return None; }
    let op_start = line.find("op=")? + 3;
    let label_key = " label=\"";
    let label_pos = line[op_start..].find(label_key)? + op_start;
    let op = line[op_start..label_pos].trim().to_string();

    let label_start = label_pos + label_key.len();
    let label_end = line[label_start..].find('"')? + label_start;
    let label = line[label_start..label_end].to_string();

    let len_key = " len=";
    let len_pos = line[label_end..].find(len_key)? + label_end + len_key.len();
    let st_key = " st0..3=[";
    let st_pos = line[len_pos..].find(st_key)? + len_pos;
    let len_str = line[len_pos..st_pos].trim();
    let len: usize = len_str.parse().ok()?;

    let st_start = st_pos + st_key.len();
    let st_end = line[st_start..].find(']')? + st_start;
    let mut it = line[st_start..st_end].split(',').map(|s| s.trim());
    let mut arr = [0u64; 4];
    for i in 0..4 { arr[i] = it.next()?.parse().ok()?; }
    Some(EventLine { op, label, len, st_prefix: arr })
}

fn parse_events_grouped_by_tag(output: &str) -> (Vec<EventLine>, Vec<EventLine>) {
    let mut cur_tag: Option<String> = None;
    let mut prover: Vec<EventLine> = Vec::new();
    let mut verifier: Vec<EventLine> = Vec::new();
    for line in output.lines() {
        if let Some(hdr_pos) = line.find("--- ") {
            // header looks like: --- [TAG] Transcript dump [ctx] ---
            if let Some(open) = line[hdr_pos + 4..].find('[') {
                let open_idx = hdr_pos + 4 + open + 1;
                if let Some(close_rel) = line[open_idx..].find(']') {
                    let tag = &line[open_idx..open_idx + close_rel];
                    cur_tag = Some(tag.to_string());
                    continue;
                }
            }
        }
        if let Some(ev) = parse_event_line(line) {
            match cur_tag.as_deref() {
                Some("PROVER") => prover.push(ev),
                Some("VERIFIER") => verifier.push(ev),
                _ => { /* ignore untagged */ }
            }
        }
    }
    (prover, verifier)
}

/// Run a specific test function (by name) as a child and capture combined stdout+stderr.
fn run_child_test_capture(test_name: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.arg("--nocapture")
        .arg("--test-threads=1")
        .arg("--exact")
        .arg(test_name)
        .env("NEO_TRANSCRIPT_DUMP", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = cmd.output().expect("child run");
    let mut s = String::new();
    s.push_str(&String::from_utf8_lossy(&out.stdout));
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn transcript_parity_prover_vs_verifier() {
    // Spawn the subtest that runs prover then verifier with tagged transcript dumps.
    let out = run_child_test_capture("__transcript_parity_subprocess");

    let (prover_events, verifier_events) = parse_events_grouped_by_tag(&out);

    // If debug logging is off in neo-transcript, there will be no dumps — skip.
    if prover_events.is_empty() && verifier_events.is_empty() {
        eprintln!("[SKIP] neo-transcript debug-log feature not enabled; no dumps captured");
        return;
    }

    let n = prover_events.len().min(verifier_events.len());
    for i in 0..n {
        let a = &prover_events[i];
        let b = &verifier_events[i];
        let same_core = a.op == b.op && a.label == b.label && a.len == b.len;
        let same_state = a.st_prefix == b.st_prefix;
        if !same_core || !same_state {
            panic!(
                "Transcript diverged at event #{i}:\n\
                 PROVER:   op={} label=\"{}\" len={} st0..3={:?}\n\
                 VERIFIER: op={} label=\"{}\" len={} st0..3={:?}",
                a.op, a.label, a.len, a.st_prefix,
                b.op, b.label, b.len, b.st_prefix
            );
        }
    }
    assert_eq!(prover_events.len(), verifier_events.len(),
        "Transcript length mismatch: prover {} vs verifier {}",
        prover_events.len(), verifier_events.len());
}

// Subprocess-only test: executes prover then verifier with different NEO_TRANSCRIPT_TAGs
#[test]
fn __transcript_parity_subprocess() {
    let (me, wit, pp) = tiny_honest_instance();

    // Prover run (tagged)
    std::env::set_var("NEO_TRANSCRIPT_TAG", "PROVER");
    let proof = compress_me_to_lean_proof_with_pp(&me, &wit, Some(pp)).expect("prove");

    // Verifier run (tagged)
    std::env::set_var("NEO_TRANSCRIPT_TAG", "VERIFIER");
    let ok = verify_lean_proof(&proof).expect("verify call");
    assert!(ok, "honest proof should verify");
}

