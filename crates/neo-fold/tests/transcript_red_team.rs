// crates/neo-fold/tests/transcript_red_team.rs
use neo_fold::transcript::{FoldTranscript, Domain};

#[test]
fn domain_separation_is_binding() {
    let mut t1 = FoldTranscript::new();
    let mut t2 = FoldTranscript::new();

    t1.domain(Domain::CCS);
    t1.absorb_u64(&[1,2,3]);

    t2.domain(Domain::Rlc); // different domain, same payload
    t2.absorb_u64(&[1,2,3]);

    let c1 = t1.challenge_f();
    let c2 = t2.challenge_f();
    assert_ne!(c1, c2, "different domains must yield different challenges");
}

#[test]
fn order_matters_in_transcript() {
    let mut t1 = FoldTranscript::new();
    let mut t2 = FoldTranscript::new();

    t1.domain(Domain::CCS);
    t1.absorb_u64(&[42]);
    t1.domain(Domain::Dec);
    t1.absorb_u64(&[7]);

    t2.domain(Domain::Dec);
    t2.absorb_u64(&[7]);
    t2.domain(Domain::CCS);
    t2.absorb_u64(&[42]);

    let c1 = t1.challenge_f();
    let c2 = t2.challenge_f();
    assert_ne!(c1, c2, "challenge must bind to message ordering");
}
