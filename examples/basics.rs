//! Basic Lie group operations in 30 lines.
//!
//! Run: `cargo run --example basics`

use lie_groups::prelude::*;

fn main() {
    // Group elements
    let g = SU2::rotation_x(0.5);
    let h = SU2::rotation_y(0.3);
    println!("g = {}", g);
    println!("h = {}", h);

    // Composition via * operator
    let product = &g * &h;
    println!("g·h = {}", product);

    // Inverse
    let g_inv = g.inverse();
    let should_be_identity = g_inv.compose(&g);
    println!(
        "g⁻¹·g = {} (distance from I: {:.2e})",
        should_be_identity,
        should_be_identity.distance_to_identity()
    );

    // Exponential map: algebra → group
    let x = Su2Algebra::new([0.1, 0.2, 0.3]);
    let exp_x = SU2::exp(&x);
    println!("exp({:?}) = {}", x, exp_x);

    // Logarithm: group → algebra
    let log_g = g.log().unwrap();
    println!("log(g) = {:?}", log_g);

    // Compose a path of rotations
    let path: Vec<SU2> = (0..10).map(|i| SU2::rotation_z(0.1 * i as f64)).collect();
    let holonomy: SU2 = path.iter().product();
    println!("holonomy = {}", holonomy);

    // BCH formula: log(exp(X)·exp(Y)) ≈ X + Y + ½[X,Y] + ...
    let y = Su2Algebra::new([0.05, -0.1, 0.0]);
    let bch = bch_checked::<Su2Algebra>(&x, &y, 5).unwrap();
    println!("BCH(X,Y) = {:?}", bch);

    // Works generically over any Lie group
    fn parallel_transport<G: LieGroup + for<'a> std::iter::Product<&'a G>>(path: &[G]) -> G {
        path.iter().product()
    }
    let u1_path = vec![
        U1::from_angle(0.1),
        U1::from_angle(0.2),
        U1::from_angle(0.3),
    ];
    println!("U(1) holonomy = {}", parallel_transport(&u1_path));
}
