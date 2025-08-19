use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neo_sumcheck::oracle::{create_fri_backend, FriImpl};
use neo_sumcheck::{ExtF, from_base, F};
use neo_poly::Polynomial;
use p3_field::PrimeCharacteristicRing;

fn bench_fri_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_backends");
    
    // Test different polynomial degrees
    let degrees = vec![3, 7, 15, 31];
    
    for degree in degrees {
        let coeffs: Vec<ExtF> = (0..=degree)
            .map(|i| from_base(F::from_u64(i as u64 + 1)))
            .collect();
        let poly = Polynomial::new(coeffs);
        let polys = vec![poly];
        let point = vec![from_base(F::from_u64(42))];
        
        group.throughput(Throughput::Elements(degree as u64));
        
        // Benchmark Custom FRI backend
        group.bench_with_input(
            BenchmarkId::new("custom_fri_commit_open", degree),
            &degree,
            |b, _| {
                b.iter(|| {
                    let mut transcript = vec![];
                    let mut backend = create_fri_backend(
                        FriImpl::Custom,
                        polys.clone(),
                        &mut transcript,
                    );
                    
                    let commits = backend.commit(polys.clone()).unwrap();
                    let (evals, proofs) = backend.open_at_point(&point).unwrap();
                    let verified = backend.verify_openings(&commits, &point, &evals, &proofs);
                    
                    black_box((commits, evals, proofs, verified))
                })
            },
        );
        
        // Benchmark P3 FRI backend (if available)
        #[cfg(feature = "p3-fri")]
        group.bench_with_input(
            BenchmarkId::new("p3_fri_commit_open", degree),
            &degree,
            |b, _| {
                b.iter(|| {
                    let mut transcript = vec![];
                    let mut backend = create_fri_backend(
                        FriImpl::Plonky3,
                        polys.clone(),
                        &mut transcript,
                    );
                    
                    let commits = backend.commit(polys.clone()).unwrap();
                    let (evals, proofs) = backend.open_at_point(&point).unwrap();
                    let verified = backend.verify_openings(&commits, &point, &evals, &proofs);
                    
                    black_box((commits, evals, proofs, verified))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_fri_commitment_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_commitment");
    
    // Test larger polynomials for commitment-only benchmark
    let degrees = vec![7, 15, 31, 63];
    
    for degree in degrees {
        let coeffs: Vec<ExtF> = (0..=degree)
            .map(|i| from_base(F::from_u64((i * 17 + 3) as u64))) // Add some variation
            .collect();
        let poly = Polynomial::new(coeffs);
        let polys = vec![poly];
        
        group.throughput(Throughput::Elements(degree as u64));
        
        // Custom FRI commitment
        group.bench_with_input(
            BenchmarkId::new("custom_commit", degree),
            &degree,
            |b, _| {
                b.iter(|| {
                    let mut transcript = vec![];
                    let mut backend = create_fri_backend(
                        FriImpl::Custom,
                        polys.clone(),
                        &mut transcript,
                    );
                    let commits = backend.commit(polys.clone()).unwrap();
                    black_box(commits)
                })
            },
        );
        
        // P3 FRI commitment
        #[cfg(feature = "p3-fri")]
        group.bench_with_input(
            BenchmarkId::new("p3_commit", degree),
            &degree,
            |b, _| {
                b.iter(|| {
                    let mut transcript = vec![];
                    let mut backend = create_fri_backend(
                        FriImpl::Plonky3,
                        polys.clone(),
                        &mut transcript,
                    );
                    let commits = backend.commit(polys.clone()).unwrap();
                    black_box(commits)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_fri_opening_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_opening");
    
    let degrees = vec![7, 15, 31];
    
    for degree in degrees {
        let coeffs: Vec<ExtF> = (0..=degree)
            .map(|i| from_base(F::from_u64((i * 23 + 7) as u64)))
            .collect();
        let poly = Polynomial::new(coeffs);
        let polys = vec![poly];
        let point = vec![from_base(F::from_u64(99))];
        
        group.throughput(Throughput::Elements(degree as u64));
        
        // Custom FRI opening
        group.bench_with_input(
            BenchmarkId::new("custom_open", degree),
            &degree,
            |b, _| {
                // Pre-commit for fair comparison
                let mut transcript = vec![];
                let mut backend = create_fri_backend(
                    FriImpl::Custom,
                    polys.clone(),
                    &mut transcript,
                );
                let _commits = backend.commit(polys.clone()).unwrap();
                
                b.iter(|| {
                    let (evals, proofs) = backend.open_at_point(&point).unwrap();
                    black_box((evals, proofs))
                })
            },
        );
        
        // P3 FRI opening
        #[cfg(feature = "p3-fri")]
        group.bench_with_input(
            BenchmarkId::new("p3_open", degree),
            &degree,
            |b, _| {
                // Pre-commit for fair comparison
                let mut transcript = vec![];
                let mut backend = create_fri_backend(
                    FriImpl::Plonky3,
                    polys.clone(),
                    &mut transcript,
                );
                let _commits = backend.commit(polys.clone()).unwrap();
                
                b.iter(|| {
                    let (evals, proofs) = backend.open_at_point(&point).unwrap();
                    black_box((evals, proofs))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_fri_backends,
    bench_fri_commitment_only,
    bench_fri_opening_only
);
criterion_main!(benches);
