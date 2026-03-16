//! Sample random group elements and verify statistical properties.
//!
//! Run: `cargo run --example random_sampling`

use lie_groups::prelude::*;
use rand::SeedableRng;

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Sample from each compact group
    println!("=== Random Haar sampling ===\n");

    // SU(2)
    let g2 = SU2::random_haar(&mut rng);
    println!("SU(2): {}", g2);
    println!("  unitary: {}", g2.verify_unitarity(1e-10));

    // SO(3)
    let r = SO3::random_haar(&mut rng);
    println!("SO(3): {}", r);
    println!("  orthogonal: {}", r.verify_orthogonality(1e-10));

    // SU(3) — the QCD gauge group
    let g3 = SU3::random_haar(&mut rng);
    println!("SU(3): {}", g3);
    println!("  unitary: {}", g3.verify_unitarity(1e-10));

    // SU(4) — Pati-Salam model
    let g4 = SU4::random_haar(&mut rng);
    println!("SU(4): {}", g4);
    println!("  unitary: {}", g4.verify_unitarity(1e-10));

    // Verify Haar measure: average trace should be 0 for SU(N), N≥2
    println!("\n=== Haar measure check: <tr(U)> → 0 ===\n");
    let n_samples = 10_000;

    let avg_trace_su2: f64 = (0..n_samples)
        .map(|_| SU2::random_haar(&mut rng).trace().re)
        .sum::<f64>()
        / n_samples as f64;
    println!("SU(2): <Re tr(U)> = {:.6} (expect ~0)", avg_trace_su2);

    let avg_trace_su3: f64 = (0..n_samples)
        .map(|_| SU3::random_haar(&mut rng).trace().re)
        .sum::<f64>()
        / n_samples as f64;
    println!("SU(3): <Re tr(U)> = {:.6} (expect ~0)", avg_trace_su3);

    let avg_trace_sun4: f64 = (0..n_samples)
        .map(|_| SUN::<4>::random_haar(&mut rng).trace().re)
        .sum::<f64>()
        / n_samples as f64;
    println!("SU(4): <Re tr(U)> = {:.6} (expect ~0)", avg_trace_sun4);
}
