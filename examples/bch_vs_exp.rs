//! Verify the Baker-Campbell-Hausdorff formula against direct exp·exp.
//!
//! The BCH formula gives: log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + ...
//! This example compares the truncated series against the exact answer.
//!
//! Run: `cargo run --example bch_vs_exp`

use lie_groups::prelude::*;

fn main() {
    println!("=== BCH formula vs direct computation ===\n");

    // Small algebra elements (BCH converges well)
    let x = Su2Algebra::new([0.1, 0.0, 0.0]);
    let y = Su2Algebra::new([0.0, 0.1, 0.0]);

    // Direct: log(exp(X) · exp(Y))
    let direct = SU2::exp(&x) * SU2::exp(&y);
    let z_exact = direct.log().unwrap();

    // BCH approximation at various orders
    for order in [2, 3, 4, 5] {
        let z_bch = lie_groups::bch_safe::<SU2>(&x, &y, order).unwrap();
        let error = z_exact.add(&z_bch.scale(-1.0)).norm();
        println!("Order {}: error = {:.2e}", order, error);
    }

    // bch_checked with highest order
    let z_best = lie_groups::bch_safe::<SU2>(&x, &y, 5).unwrap();
    let error_best = z_exact.add(&z_best.scale(-1.0)).norm();
    println!("Best(5): error = {:.2e}", error_best);

    // Show the bracket structure
    println!("\n=== Lie bracket structure constants ===\n");
    for i in 0..3 {
        for j in (i + 1)..3 {
            let ei = Su2Algebra::basis_element(i);
            let ej = Su2Algebra::basis_element(j);
            let bracket = ei.bracket(&ej);
            println!("[e{}, e{}] = {:?}", i, j, bracket.to_components());
        }
    }

    // SU(3) bracket
    println!("\n=== SU(3) brackets ===\n");
    let t1 = Su3Algebra::basis_element(0);
    let t2 = Su3Algebra::basis_element(1);
    let bracket_12 = t1.bracket(&t2);
    println!("[T1, T2] = {:?}", bracket_12.to_components());
}
