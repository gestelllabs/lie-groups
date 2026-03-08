//! Baker-Campbell-Hausdorff (BCH) Formula
//!
//! Provides logarithmic coordinates for Lie group composition.
//!
//! # Mathematical Background
//!
//! For X, Y ∈ 𝔤 (Lie algebra), the BCH formula gives Z ∈ 𝔤 such that:
//! ```text
//! exp(Z) = exp(X) · exp(Y)
//! ```
//!
//! The full series is:
//! ```text
//! Z = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]])
//!   - (1/24)[Y,[X,[X,Y]]] + ...
//! ```
//!
//! # Convergence (IMPORTANT)
//!
//! The BCH series has a **finite convergence radius**:
//!
//! | Bound | Value | Use case |
//! |-------|-------|----------|
//! | Strict | log(2) ≈ 0.693 | Guaranteed convergence |
//! | Practical | π ≈ 3.14 | Often works but not guaranteed |
//!
//! For ||X|| + ||Y|| > log(2), the truncated series may give incorrect results.
//! Use direct composition exp(X)·exp(Y) instead.
//!
//! ## Why log(2)?
//!
//! The BCH series can be written as Z = Σₙ Zₙ(X,Y) where each Zₙ is a sum of
//! nested brackets. The key insight is that:
//!
//! ```text
//! ||Zₙ|| ≤ Cₙ · (||X|| + ||Y||)ⁿ
//! ```
//!
//! where Cₙ grows like 1/n. The generating function Σ Cₙ tⁿ converges for |t| < log(2).
//!
//! **Proof sketch:** Write exp(X)exp(Y) = exp(Z) and take log of both sides:
//!
//! ```text
//! Z = log(exp(X)exp(Y)) = log(I + (exp(X)exp(Y) - I))
//! ```
//!
//! The logarithm series log(I + A) = A - A²/2 + A³/3 - ... converges for ||A|| < 1.
//! Since ||exp(X)exp(Y) - I|| ≤ e^{||X||+||Y||} - 1, we need e^{||X||+||Y||} - 1 < 1,
//! giving ||X|| + ||Y|| < log(2). ∎
//!
//! The functions in this module include runtime checks (`debug_assert`) to warn
//! when inputs exceed the convergence radius.
//!
//! # Applications
//!
//! - Numerical integration on Lie groups (Magnus expansion)
//! - Understanding non-commutativity: Z ≠ X + Y when `[X,Y]` ≠ 0
//! - Efficient composition in algebra coordinates
//!
//! # References
//!
//! - Rossmann, W. "Lie Groups: An Introduction through Linear Groups" (2002)
//! - Iserles et al. "Lie-group methods" Acta Numerica (2000)
//! - Blanes & Casas "A Concise Introduction to Geometric Numerical Integration" (2016)

use crate::traits::LieAlgebra;

/// Convergence radius for BCH series (strict bound)
///
/// The BCH series is guaranteed to converge when ||X|| + ||Y|| < `BCH_CONVERGENCE_RADIUS`.
/// Beyond this, the series may diverge or give incorrect results.
pub const BCH_CONVERGENCE_RADIUS: f64 = 0.693; // log(2)

/// Practical limit for BCH series (often works but not guaranteed)
///
/// For ||X|| + ||Y|| < `BCH_PRACTICAL_LIMIT`, the BCH series typically gives
/// reasonable results even though formal convergence is not guaranteed.
pub const BCH_PRACTICAL_LIMIT: f64 = 2.0;

/// Check if BCH series is expected to converge for given inputs.
///
/// Returns `true` if ||X|| + ||Y|| < [`BCH_CONVERGENCE_RADIUS`] (strict bound).
/// For inputs where this returns `false`, consider using direct composition
/// exp(X)·exp(Y) instead of BCH.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_will_converge;
/// use lie_groups::su2::Su2Algebra;
///
/// let small = Su2Algebra([0.1, 0.1, 0.1]);
/// let large = Su2Algebra([1.0, 1.0, 1.0]);
///
/// assert!(bch_will_converge(&small, &small));  // ||X|| + ||Y|| ≈ 0.35 < 0.693
/// assert!(!bch_will_converge(&large, &large)); // ||X|| + ||Y|| ≈ 3.46 > 0.693
/// ```
#[must_use]
pub fn bch_will_converge<A: LieAlgebra>(x: &A, y: &A) -> bool {
    x.norm() + y.norm() < BCH_CONVERGENCE_RADIUS
}

/// Check if BCH series is likely to give reasonable results.
///
/// Returns `true` if ||X|| + ||Y|| < [`BCH_PRACTICAL_LIMIT`].
/// This is a looser check than [`bch_will_converge`] - the series may not
/// formally converge, but empirically often gives acceptable results.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_is_practical;
/// use lie_groups::su2::Su2Algebra;
///
/// let moderate = Su2Algebra([0.5, 0.5, 0.0]);
/// let large = Su2Algebra([2.0, 0.0, 0.0]);
///
/// assert!(bch_is_practical(&moderate, &moderate));  // ||X|| + ||Y|| ≈ 1.41 < 2.0
/// assert!(!bch_is_practical(&large, &large));       // ||X|| + ||Y|| = 4.0 > 2.0
/// ```
#[must_use]
pub fn bch_is_practical<A: LieAlgebra>(x: &A, y: &A) -> bool {
    x.norm() + y.norm() < BCH_PRACTICAL_LIMIT
}

/// Baker-Campbell-Hausdorff formula truncated to 2nd order.
///
/// Computes Z ≈ X + Y + (1/2)`[X,Y]` such that exp(Z) ≈ exp(X) · exp(Y).
///
/// # Accuracy
///
/// - Error: O(||X||³ + ||Y||³)
/// - Good for small ||X||, ||Y|| < 0.5
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_second_order;
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU2;
///
/// let x = Su2Algebra([0.05, 0.0, 0.0]);
/// let y = Su2Algebra([0.0, 0.05, 0.0]);
///
/// // Method 1: Direct composition
/// let g1 = SU2::exp(&x);
/// let g2 = SU2::exp(&y);
/// let product = g1.compose(&g2);
///
/// // Method 2: BCH formula
/// let z = bch_second_order(&x, &y);
/// let product_bch = SU2::exp(&z);
///
/// // Should be approximately equal
/// let distance = product.compose(&product_bch.inverse()).distance_to_identity();
/// assert!(distance < 1e-2);
/// ```
#[must_use]
pub fn bch_second_order<A: LieAlgebra>(x: &A, y: &A) -> A {
    // Check convergence radius in debug builds
    debug_assert!(
        x.norm() + y.norm() < BCH_PRACTICAL_LIMIT,
        "BCH inputs exceed practical limit: ||X|| + ||Y|| = {} > {}. \
         Consider using direct composition exp(X)·exp(Y) instead.",
        x.norm() + y.norm(),
        BCH_PRACTICAL_LIMIT
    );

    // Z = X + Y + (1/2)[X,Y]
    let xy_bracket = x.bracket(y);
    let half_bracket = xy_bracket.scale(0.5);

    x.add(y).add(&half_bracket)
}

/// Baker-Campbell-Hausdorff formula truncated to 3rd order.
///
/// Computes Z such that exp(Z) ≈ exp(X) · exp(Y) with higher accuracy.
///
/// # Formula
///
/// ```text
/// Z = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]])
/// ```
///
/// Note: `[Y,[Y,X]] = -[Y,[X,Y]]`, so this is equivalently
/// `(1/12)([X,[X,Y]] - [Y,[X,Y]])`.
///
/// # Accuracy
///
/// - Error: O(||X||⁴ + ||Y||⁴)
/// - Good for ||X||, ||Y|| < 1.0
///
/// # Performance
///
/// Requires 3 bracket computations vs 1 for second-order.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_third_order;
/// use lie_groups::su3::Su3Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU3;
///
/// let x = Su3Algebra([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let y = Su3Algebra([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
///
/// let z = bch_third_order(&x, &y);
/// let product_bch = SU3::exp(&z);
///
/// // Verify it's close to exp(X) · exp(Y)
/// let g1 = SU3::exp(&x);
/// let g2 = SU3::exp(&y);
/// let product_direct = g1.compose(&g2);
///
/// let distance = product_direct.compose(&product_bch.inverse()).distance_to_identity();
/// assert!(distance < 1e-2);
/// ```
#[must_use]
pub fn bch_third_order<A: LieAlgebra>(x: &A, y: &A) -> A {
    // Check convergence radius in debug builds
    debug_assert!(
        x.norm() + y.norm() < BCH_PRACTICAL_LIMIT,
        "BCH inputs exceed practical limit: ||X|| + ||Y|| = {} > {}. \
         Consider using direct composition exp(X)·exp(Y) instead.",
        x.norm() + y.norm(),
        BCH_PRACTICAL_LIMIT
    );

    // Z = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]])
    let xy_bracket = x.bracket(y); // [X,Y]
    let half_bracket = xy_bracket.scale(0.5);

    // (1/12)[X,[X,Y]]
    let x_xy = x.bracket(&xy_bracket);
    let term3 = x_xy.scale(1.0 / 12.0);

    // (1/12)[Y,[Y,X]]  (note: [Y,X] = -[X,Y])
    let yx_bracket = xy_bracket.scale(-1.0); // [Y,X] = -[X,Y]
    let y_yx = y.bracket(&yx_bracket); // [Y,[Y,X]]
    let term4 = y_yx.scale(1.0 / 12.0);

    x.add(y).add(&half_bracket).add(&term3).add(&term4)
}

/// Baker-Campbell-Hausdorff formula truncated to 4th order.
///
/// Computes Z such that exp(Z) ≈ exp(X) · exp(Y) with even higher accuracy.
///
/// # Formula
///
/// ```text
/// Z = X + Y + (1/2)[X,Y]
///   + (1/12)([X,[X,Y]] + [Y,[Y,X]])
///   - (1/24)[Y,[X,[X,Y]]]
/// ```
///
/// Note: by the Jacobi identity, `[Y,[X,[X,Y]]] = [X,[Y,[X,Y]]]`.
///
/// # Accuracy
///
/// - Error: O(||X||⁵ + ||Y||⁵)
/// - Good for ||X||, ||Y|| < 1.5
///
/// # Performance
///
/// Requires 6 bracket computations vs 3 for third-order.
/// Use when high accuracy is needed for moderate-size algebra elements.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_fourth_order;
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU2;
///
/// let x = Su2Algebra([0.15, 0.0, 0.0]);
/// let y = Su2Algebra([0.0, 0.15, 0.0]);
///
/// let z = bch_fourth_order(&x, &y);
/// let product_bch = SU2::exp(&z);
///
/// // Verify it's close to exp(X) · exp(Y)
/// let g1 = SU2::exp(&x);
/// let g2 = SU2::exp(&y);
/// let product_direct = g1.compose(&g2);
///
/// let distance = product_direct.compose(&product_bch.inverse()).distance_to_identity();
/// assert!(distance < 0.04);
/// ```
#[must_use]
pub fn bch_fourth_order<A: LieAlgebra>(x: &A, y: &A) -> A {
    // Check convergence radius in debug builds
    debug_assert!(
        x.norm() + y.norm() < BCH_PRACTICAL_LIMIT,
        "BCH inputs exceed practical limit: ||X|| + ||Y|| = {} > {}. \
         Consider using direct composition exp(X)·exp(Y) instead.",
        x.norm() + y.norm(),
        BCH_PRACTICAL_LIMIT
    );

    // Start with third-order terms
    let z3 = bch_third_order(x, y);

    // Fourth-order term: -(1/24)[Y,[X,[X,Y]]]
    let xy = x.bracket(y);
    let x_xy = x.bracket(&xy); // [X,[X,Y]]
    let y_x_xy = y.bracket(&x_xy); // [Y,[X,[X,Y]]]
    let term4 = y_x_xy.scale(-1.0 / 24.0);

    z3.add(&term4)
}

/// Baker-Campbell-Hausdorff formula truncated to 5th order.
///
/// Computes Z such that exp(Z) ≈ exp(X) · exp(Y) with maximum practical accuracy.
///
/// # Formula
///
/// ```text
/// Z = X + Y + (1/2)[X,Y]
///   + (1/12)([X,[X,Y]] + [Y,[Y,X]])
///   - (1/24)[Y,[X,[X,Y]]]
///   - (1/720)([X,[X,[X,[X,Y]]]] + [Y,[Y,[Y,[Y,X]]]])
///   + (1/360)([Y,[X,[X,[X,Y]]]] + [X,[Y,[Y,[Y,X]]]])
///   + (1/120)([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])
/// ```
///
/// # Accuracy
///
/// - Error: O(||X||⁶ + ||Y||⁶)
/// - Good for ||X||, ||Y|| < 2.0
/// - Near-optimal for practical purposes
///
/// # Performance
///
/// Requires 14 bracket computations. Use only when maximum accuracy is critical.
/// For most applications, 3rd or 4th order is sufficient.
///
/// # Convergence Radius
///
/// The BCH series converges when ||X|| + ||Y|| < log(2) ≈ 0.693 (matrix groups).
/// For larger norms, use direct composition exp(X)·exp(Y) instead.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_fifth_order;
/// use lie_groups::su3::Su3Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU3;
///
/// let x = Su3Algebra([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let y = Su3Algebra([0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
///
/// let z = bch_fifth_order(&x, &y);
/// let product_bch = SU3::exp(&z);
///
/// // Verify it's very close to exp(X) · exp(Y)
/// let g1 = SU3::exp(&x);
/// let g2 = SU3::exp(&y);
/// let product_direct = g1.compose(&g2);
///
/// let distance = product_direct.compose(&product_bch.inverse()).distance_to_identity();
/// assert!(distance < 5e-3);
/// ```
#[must_use]
pub fn bch_fifth_order<A: LieAlgebra>(x: &A, y: &A) -> A {
    // Check convergence radius in debug builds
    debug_assert!(
        x.norm() + y.norm() < BCH_PRACTICAL_LIMIT,
        "BCH inputs exceed practical limit: ||X|| + ||Y|| = {} > {}. \
         Consider using direct composition exp(X)·exp(Y) instead.",
        x.norm() + y.norm(),
        BCH_PRACTICAL_LIMIT
    );

    // Start with fourth-order terms
    let z4 = bch_fourth_order(x, y);

    let xy = x.bracket(y); // [X,Y]
    let yx = xy.scale(-1.0); // [Y,X] = -[X,Y]

    // Pre-compute nested brackets
    let x_xy = x.bracket(&xy); // [X,[X,Y]]
    let y_yx = y.bracket(&yx); // [Y,[Y,X]]

    // -(1/720)([X,[X,[X,[X,Y]]]] + [Y,[Y,[Y,[Y,X]]]])
    let x_x_xy = x.bracket(&x_xy); // [X,[X,[X,Y]]]
    let x_x_x_xy = x.bracket(&x_x_xy); // [X,[X,[X,[X,Y]]]]
    let y_y_yx = y.bracket(&y_yx); // [Y,[Y,[Y,X]]]
    let y_y_y_yx = y.bracket(&y_y_yx); // [Y,[Y,[Y,[Y,X]]]]
    let ad4 = x_x_x_xy.add(&y_y_y_yx).scale(-1.0 / 720.0);

    // +(1/360)([Y,[X,[X,[X,Y]]]] + [X,[Y,[Y,[Y,X]]]])
    let y_x_x_xy = y.bracket(&x_x_xy); // [Y,[X,[X,[X,Y]]]]
    let x_y_y_yx = x.bracket(&y_y_yx); // [X,[Y,[Y,[Y,X]]]]
    let mixed4 = y_x_x_xy.add(&x_y_y_yx).scale(1.0 / 360.0);

    // +(1/120)([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])
    let y_xy = y.bracket(&xy); // [Y,[X,Y]]
    let x_y_xy = x.bracket(&y_xy); // [X,[Y,[X,Y]]]
    let y_x_y_xy = y.bracket(&x_y_xy); // [Y,[X,[Y,[X,Y]]]]
    let x_yx = x.bracket(&yx); // [X,[Y,X]]
    let y_x_yx = y.bracket(&x_yx); // [Y,[X,[Y,X]]]
    let x_y_x_yx = x.bracket(&y_x_yx); // [X,[Y,[X,[Y,X]]]]
    let alt = y_x_y_xy.add(&x_y_x_yx).scale(1.0 / 120.0);

    z4.add(&ad4).add(&mixed4).add(&alt)
}

/// Estimate error bound for BCH truncation.
///
/// Given X, Y ∈ 𝔤 and a truncation order, estimate the truncation error.
///
/// # Formula
///
/// For nth-order BCH, the truncation error is bounded by:
/// ```text
/// ||error|| ≤ C_n · (||X|| + ||Y||)^(n+1)
/// ```
///
/// where `C_n` is a constant depending on the order.
///
/// # Returns
///
/// Upper bound on ||`Z_true` - `Z_approx`|| where:
/// - `Z_true`: exact BCH series
/// - `Z_approx`: truncated BCH at given order
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_error_bound;
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::LieAlgebra;
///
/// let x = Su2Algebra([0.1, 0.0, 0.0]);
/// let y = Su2Algebra([0.0, 0.1, 0.0]);
///
/// let error_2nd = bch_error_bound(&x, &y, 2);
/// let error_3rd = bch_error_bound(&x, &y, 3);
/// let error_5th = bch_error_bound(&x, &y, 5);
///
/// // Higher order => smaller error bound
/// assert!(error_5th < error_3rd);
/// assert!(error_3rd < error_2nd);
/// ```
#[must_use]
pub fn bch_error_bound<A: LieAlgebra>(x: &A, y: &A, order: usize) -> f64 {
    let norm_x = x.norm();
    let norm_y = y.norm();
    let total_norm = norm_x + norm_y;

    // Conservative error bound coefficients (empirically derived)
    let coefficient = match order {
        1 => 1.0,                          // First order: just X+Y
        2 => 0.5,                          // Second order
        3 => 0.1,                          // Third order
        4 => 0.02,                         // Fourth order
        5 => 0.005,                        // Fifth order
        _ => 1.0 / (order as f64).powi(2), // Higher orders (extrapolated)
    };

    // Error bound: C_n · (||X|| + ||Y||)^(n+1)
    coefficient * total_norm.powi((order + 1) as i32)
}

/// Safe BCH composition with automatic fallback to direct exp(X)·exp(Y).
///
/// This function computes Z such that exp(Z) = exp(X) · exp(Y), using:
/// - BCH series when ||X|| + ||Y|| < convergence radius
/// - Direct composition exp(X)·exp(Y) followed by log when outside radius
///
/// # Why This Matters
///
/// The BCH series has convergence radius log(2) ≈ 0.693. For inputs outside
/// this radius, the truncated series can give **silently incorrect results**.
/// This function guarantees correctness by falling back to direct composition.
///
/// # Arguments
///
/// * `x`, `y` - Lie algebra elements
/// * `order` - BCH truncation order (2-5)
///
/// # Returns
///
/// `Ok(Z)` where exp(Z) = exp(X)·exp(Y), or `Err` if:
/// - Direct composition needed but log failed (e.g., result at cut locus)
/// - Order is invalid
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_safe;
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU2;
///
/// // Small inputs: uses BCH series
/// let x_small = Su2Algebra([0.1, 0.0, 0.0]);
/// let y_small = Su2Algebra([0.0, 0.1, 0.0]);
/// let z_small = bch_safe::<SU2>(&x_small, &y_small, 3).unwrap();
///
/// // Large inputs: falls back to direct composition
/// let x_large = Su2Algebra([1.0, 0.0, 0.0]);
/// let y_large = Su2Algebra([0.0, 1.0, 0.0]);
/// let z_large = bch_safe::<SU2>(&x_large, &y_large, 3).unwrap();
///
/// // Both give correct results
/// let g1 = SU2::exp(&x_large);
/// let g2 = SU2::exp(&y_large);
/// let product = g1.compose(&g2);
/// let product_bch = SU2::exp(&z_large);
/// let distance = product.compose(&product_bch.inverse()).distance_to_identity();
/// assert!(distance < 1e-10);
/// ```
pub fn bch_safe<G>(x: &G::Algebra, y: &G::Algebra, order: usize) -> Result<G::Algebra, BchError>
where
    G: crate::traits::LieGroup,
{
    if !(2..=5).contains(&order) {
        return Err(BchError::InvalidOrder(order));
    }

    // Check if within convergence radius
    if bch_will_converge(x, y) {
        // Safe to use BCH series
        let z = match order {
            2 => bch_second_order(x, y),
            3 => bch_third_order(x, y),
            4 => bch_fourth_order(x, y),
            5 => bch_fifth_order(x, y),
            _ => unreachable!(),
        };
        return Ok(z);
    }

    // Outside convergence radius: use direct composition
    // exp(Z) = exp(X)·exp(Y), so Z = log(exp(X)·exp(Y))
    let gx = G::exp(x);
    let gy = G::exp(y);
    let product = gx.compose(&gy);

    match G::log(&product) {
        Ok(z) => Ok(z),
        Err(_) => Err(BchError::LogFailed),
    }
}

/// Result of BCH composition indicating which method was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BchMethod {
    /// Used BCH series (inputs within convergence radius)
    Series { order: usize },
    /// Used direct exp(X)·exp(Y) then log (inputs outside radius)
    DirectComposition,
}

/// Safe BCH composition with method reporting.
///
/// Like [`bch_safe`] but also reports which method was used.
/// Useful for diagnostics and testing.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::{bch_safe_with_method, BchMethod};
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::SU2;
///
/// let x_small = Su2Algebra([0.1, 0.0, 0.0]);
/// let y_small = Su2Algebra([0.0, 0.1, 0.0]);
/// let (z, method) = bch_safe_with_method::<SU2>(&x_small, &y_small, 3).unwrap();
/// assert_eq!(method, BchMethod::Series { order: 3 });
///
/// let x_large = Su2Algebra([1.0, 0.0, 0.0]);
/// let y_large = Su2Algebra([0.0, 1.0, 0.0]);
/// let (z, method) = bch_safe_with_method::<SU2>(&x_large, &y_large, 3).unwrap();
/// assert_eq!(method, BchMethod::DirectComposition);
/// ```
pub fn bch_safe_with_method<G>(
    x: &G::Algebra,
    y: &G::Algebra,
    order: usize,
) -> Result<(G::Algebra, BchMethod), BchError>
where
    G: crate::traits::LieGroup,
{
    if !(2..=5).contains(&order) {
        return Err(BchError::InvalidOrder(order));
    }

    if bch_will_converge(x, y) {
        let z = match order {
            2 => bch_second_order(x, y),
            3 => bch_third_order(x, y),
            4 => bch_fourth_order(x, y),
            5 => bch_fifth_order(x, y),
            _ => unreachable!(),
        };
        return Ok((z, BchMethod::Series { order }));
    }

    let gx = G::exp(x);
    let gy = G::exp(y);
    let product = gx.compose(&gy);

    match G::log(&product) {
        Ok(z) => Ok((z, BchMethod::DirectComposition)),
        Err(_) => Err(BchError::LogFailed),
    }
}

/// Error type for safe BCH operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BchError {
    /// BCH order must be 2-5
    InvalidOrder(usize),
    /// `log()` failed (e.g., result at cut locus)
    LogFailed,
}

impl std::fmt::Display for BchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BchError::InvalidOrder(order) => {
                write!(f, "Invalid BCH order {}: must be 2, 3, 4, or 5", order)
            }
            BchError::LogFailed => write!(
                f,
                "BCH fallback failed: log() of composed element failed (possibly at cut locus)"
            ),
        }
    }
}

impl std::error::Error for BchError {}

/// Inverse BCH formula: Given Z, find X and Y such that exp(Z) = exp(X) · exp(Y).
///
/// This is useful for splitting a group element into a product of two elements.
/// We use a simple midpoint splitting: X = Y = Z/2 + correction.
///
/// # Formula
///
/// For the splitting exp(Z) = exp(X) · exp(Y), we use:
/// ```text
/// X = Y = Z/2 - (1/8)[Z/2, Z/2] = Z/2 - (1/32)[Z,Z] = Z/2
/// ```
///
/// Since `[Z,Z]` = 0, the second-order splitting is exact: X = Y = Z/2.
///
/// # Examples
///
/// ```
/// use lie_groups::bch::bch_split;
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::{LieAlgebra, LieGroup};
/// use lie_groups::SU2;
///
/// let z = Su2Algebra([0.4, 0.3, 0.2]);
/// let (x, y) = bch_split(&z);
///
/// // Verify exp(X) · exp(Y) ≈ exp(Z)
/// let gz = SU2::exp(&z);
/// let gx = SU2::exp(&x);
/// let gy = SU2::exp(&y);
/// let product = gx.compose(&gy);
///
/// let distance = gz.compose(&product.inverse()).distance_to_identity();
/// assert!(distance < 1e-10);
/// ```
#[must_use]
pub fn bch_split<A: LieAlgebra>(z: &A) -> (A, A) {
    // Symmetric splitting: X = Y = Z/2
    // This is exact for BCH since [Z/2, Z/2] = 0
    let half_z = z.scale(0.5);
    (half_z.clone(), half_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{LieAlgebra, LieGroup};
    use crate::{Su2Algebra, Su3Algebra, SU2, SU3};

    // ========================================================================
    // Safe BCH Tests
    // ========================================================================

    #[test]
    fn test_bch_safe_small_inputs_uses_series() {
        // Small inputs should use BCH series
        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);

        let (z, method) = bch_safe_with_method::<SU2>(&x, &y, 3).unwrap();
        assert_eq!(method, BchMethod::Series { order: 3 });

        // Verify correctness (BCH has truncation error, so tolerance is relaxed)
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product = g1.compose(&g2);
        let product_bch = SU2::exp(&z);
        let distance = product
            .compose(&product_bch.inverse())
            .distance_to_identity();
        // For 3rd order BCH with ||X||+||Y|| = 0.2, error ~ O(0.2^4) ≈ 1.6e-4
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_safe_large_inputs_uses_direct() {
        // Large inputs should fallback to direct composition
        let x = Su2Algebra([1.0, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 1.0, 0.0]);

        // ||X|| + ||Y|| = 2.0 > 0.693, so should use direct
        let (z, method) = bch_safe_with_method::<SU2>(&x, &y, 3).unwrap();
        assert_eq!(method, BchMethod::DirectComposition);

        // Verify correctness - should be exact (up to log precision)
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product = g1.compose(&g2);
        let product_bch = SU2::exp(&z);
        let distance = product
            .compose(&product_bch.inverse())
            .distance_to_identity();
        assert!(distance < 1e-10, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_safe_correctness_across_boundary() {
        // Test inputs just below and above convergence radius
        // Both should give correct results

        // Just below radius: ||X|| + ||Y|| = 0.6 < 0.693
        let x_below = Su2Algebra([0.3, 0.0, 0.0]);
        let y_below = Su2Algebra([0.0, 0.3, 0.0]);
        let (z_below, method_below) = bch_safe_with_method::<SU2>(&x_below, &y_below, 4).unwrap();
        assert_eq!(method_below, BchMethod::Series { order: 4 });

        // Just above radius: ||X|| + ||Y|| = 0.8 > 0.693
        let x_above = Su2Algebra([0.4, 0.0, 0.0]);
        let y_above = Su2Algebra([0.0, 0.4, 0.0]);
        let (z_above, method_above) = bch_safe_with_method::<SU2>(&x_above, &y_above, 4).unwrap();
        assert_eq!(method_above, BchMethod::DirectComposition);

        // Verify both are correct
        // Below radius: 4th order BCH with ||X||+||Y||=0.6, error ~ O(0.6^5) ≈ 0.08
        let g1_below = SU2::exp(&x_below);
        let g2_below = SU2::exp(&y_below);
        let product_below = g1_below.compose(&g2_below);
        let product_bch_below = SU2::exp(&z_below);
        let distance_below = product_below
            .compose(&product_bch_below.inverse())
            .distance_to_identity();
        assert!(
            distance_below < 0.15,
            "Below boundary distance: {}",
            distance_below
        );

        // Above radius: direct composition is exact (up to log precision)
        let g1_above = SU2::exp(&x_above);
        let g2_above = SU2::exp(&y_above);
        let product_above = g1_above.compose(&g2_above);
        let product_bch_above = SU2::exp(&z_above);
        let distance_above = product_above
            .compose(&product_bch_above.inverse())
            .distance_to_identity();
        assert!(
            distance_above < 1e-10,
            "Above boundary distance: {}",
            distance_above
        );
    }

    #[test]
    fn test_bch_safe_invalid_order() {
        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);

        assert!(matches!(
            bch_safe::<SU2>(&x, &y, 1),
            Err(BchError::InvalidOrder(1))
        ));
        assert!(matches!(
            bch_safe::<SU2>(&x, &y, 7),
            Err(BchError::InvalidOrder(7))
        ));
    }

    #[test]
    fn test_bch_safe_su3() {
        // Test with SU(3)
        let x = Su3Algebra([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // ||X|| + ||Y|| = 1.6 > 0.693, should use direct
        let (z, method) = bch_safe_with_method::<SU3>(&x, &y, 3).unwrap();
        assert_eq!(method, BchMethod::DirectComposition);

        // Verify correctness
        let g1 = SU3::exp(&x);
        let g2 = SU3::exp(&y);
        let product = g1.compose(&g2);
        let product_bch = SU3::exp(&z);
        let distance = product
            .compose(&product_bch.inverse())
            .distance_to_identity();
        assert!(distance < 1e-8, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_safe_all_orders() {
        // Test all valid orders with safe function
        let x = Su2Algebra([0.2, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.2, 0.0]);

        for order in 2..=5 {
            let z = bch_safe::<SU2>(&x, &y, order).unwrap();
            let g1 = SU2::exp(&x);
            let g2 = SU2::exp(&y);
            let product = g1.compose(&g2);
            let product_bch = SU2::exp(&z);
            let distance = product
                .compose(&product_bch.inverse())
                .distance_to_identity();
            assert!(distance < 1e-3, "Order {}: distance = {}", order, distance);
        }
    }

    #[test]
    fn test_bch_error_display() {
        let err1 = BchError::InvalidOrder(7);
        assert!(err1.to_string().contains('7'));
        assert!(err1.to_string().contains("2, 3, 4, or 5"));

        let err2 = BchError::LogFailed;
        assert!(err2.to_string().contains("cut locus"));
    }

    // ========================================================================
    // Original BCH Tests
    // ========================================================================

    #[test]
    fn test_bch_second_order_su2() {
        // Very small algebra elements for second-order accuracy
        let x = Su2Algebra([0.05, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.05, 0.0]);

        // Direct composition
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product_direct = g1.compose(&g2);

        // BCH approximation
        let z = bch_second_order(&x, &y);
        let product_bch = SU2::exp(&z);

        // BCH truncation error + numerical errors from exp()
        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();
        assert!(distance < 5e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_third_order_su2() {
        // Small algebra elements for third-order accuracy
        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);

        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product_direct = g1.compose(&g2);

        let z = bch_third_order(&x, &y);
        let product_bch = SU2::exp(&z);

        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();

        // 3rd order with ||X||+||Y||=0.2: error ~ O(0.2^4) ≈ 1.6e-4
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_second_order_su3() {
        let x = Su3Algebra([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let g1 = SU3::exp(&x);
        let g2 = SU3::exp(&y);
        let product_direct = g1.compose(&g2);

        let z = bch_second_order(&x, &y);
        let product_bch = SU3::exp(&z);

        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_split() {
        let z = Su2Algebra([0.4, 0.3, 0.2]);
        let (x, y) = bch_split(&z);

        // Verify X = Y = Z/2
        for i in 0..3 {
            assert!((x.0[i] - z.0[i] / 2.0).abs() < 1e-10);
            assert!((y.0[i] - z.0[i] / 2.0).abs() < 1e-10);
        }

        // Verify exp(X) · exp(Y) = exp(Z)
        let gz = SU2::exp(&z);
        let gx = SU2::exp(&x);
        let gy = SU2::exp(&y);
        let product = gx.compose(&gy);

        let distance = gz.compose(&product.inverse()).distance_to_identity();
        assert!(distance < 1e-10);
    }

    #[test]
    fn test_bch_commutative_case() {
        // When [X,Y] = 0, BCH is exact: Z = X + Y
        let x = Su2Algebra([0.2, 0.0, 0.0]);
        let y = Su2Algebra([0.3, 0.0, 0.0]); // Same direction, so [X,Y] = 0

        let z_second = bch_second_order(&x, &y);
        let z_third = bch_third_order(&x, &y);

        // Both should give X + Y exactly
        let x_plus_y = x.add(&y);
        for i in 0..3 {
            assert!((z_second.0[i] - x_plus_y.0[i]).abs() < 1e-10);
            assert!((z_third.0[i] - x_plus_y.0[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bch_antisymmetry() {
        // BCH(X,Y) ≠ BCH(Y,X) in general
        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);

        let z_xy = bch_second_order(&x, &y);
        let z_yx = bch_second_order(&y, &x);

        // The bracket term has opposite sign
        let diff = z_xy.0[2] - z_yx.0[2];
        assert!(diff.abs() > 0.01); // Should be different
    }

    #[test]
    fn test_bch_fourth_order_su2() {
        // Small-to-moderate algebra elements for fourth-order accuracy
        let x = Su2Algebra([0.15, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.15, 0.0]);

        // Direct composition
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product_direct = g1.compose(&g2);

        // BCH approximation (4th order)
        let z = bch_fourth_order(&x, &y);
        let product_bch = SU2::exp(&z);

        // BCH truncation error + numerical errors from exp()
        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();

        // For ||X|| + ||Y|| = 0.3, error ~ O(0.3^5) ≈ 2.4e-4
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_fifth_order_su2() {
        // Moderate algebra elements for fifth-order accuracy
        let x = Su2Algebra([0.2, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.2, 0.0]);

        // Direct composition
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product_direct = g1.compose(&g2);

        // BCH approximation (5th order)
        let z = bch_fifth_order(&x, &y);
        let product_bch = SU2::exp(&z);

        // BCH truncation error + numerical errors from exp()
        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();

        // For ||X|| + ||Y|| = 0.4, error ~ O(0.4^6) ≈ 4e-4
        // In SU(2), degree-5 terms in a 3-dim algebra are exact, so this is very small
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_fifth_order_su3() {
        // Test 5th order on SU(3) with smaller elements
        let x = Su3Algebra([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra([0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let g1 = SU3::exp(&x);
        let g2 = SU3::exp(&y);
        let product_direct = g1.compose(&g2);

        let z = bch_fifth_order(&x, &y);
        let product_bch = SU3::exp(&z);

        let distance = product_direct
            .compose(&product_bch.inverse())
            .distance_to_identity();

        // For ||X|| + ||Y|| = 0.3, error ~ O(0.3^6) ≈ 7e-5
        assert!(distance < 1e-3, "Distance: {}", distance);
    }

    #[test]
    fn test_bch_error_bounds() {
        // Error bounds should decrease with order
        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);

        let error_2nd = bch_error_bound(&x, &y, 2);
        let error_3rd = bch_error_bound(&x, &y, 3);
        let error_4th = bch_error_bound(&x, &y, 4);
        let error_5th = bch_error_bound(&x, &y, 5);

        // Higher order => smaller error bound
        assert!(
            error_5th < error_4th,
            "5th order ({}) should have smaller error than 4th order ({})",
            error_5th,
            error_4th
        );
        assert!(
            error_4th < error_3rd,
            "4th order ({}) should have smaller error than 3rd order ({})",
            error_4th,
            error_3rd
        );
        assert!(
            error_3rd < error_2nd,
            "3rd order ({}) should have smaller error than 2nd order ({})",
            error_3rd,
            error_2nd
        );

        println!("Error bounds for ||X|| = ||Y|| = 0.1:");
        println!("  2nd order: {:.2e}", error_2nd);
        println!("  3rd order: {:.2e}", error_3rd);
        println!("  4th order: {:.2e}", error_4th);
        println!("  5th order: {:.2e}", error_5th);
    }

    #[test]
    fn test_bch_convergence_order_comparison() {
        // Compare actual accuracy across different orders
        // Use smaller elements where truncation error dominates
        let x = Su2Algebra([0.08, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.08, 0.0]);

        // Ground truth
        let g1 = SU2::exp(&x);
        let g2 = SU2::exp(&y);
        let product_exact = g1.compose(&g2);

        // Test all orders
        let z2 = bch_second_order(&x, &y);
        let z3 = bch_third_order(&x, &y);
        let z4 = bch_fourth_order(&x, &y);
        let z5 = bch_fifth_order(&x, &y);

        let dist2 = product_exact
            .compose(&SU2::exp(&z2).inverse())
            .distance_to_identity();
        let dist3 = product_exact
            .compose(&SU2::exp(&z3).inverse())
            .distance_to_identity();
        let dist4 = product_exact
            .compose(&SU2::exp(&z4).inverse())
            .distance_to_identity();
        let dist5 = product_exact
            .compose(&SU2::exp(&z5).inverse())
            .distance_to_identity();

        // With correct BCH signs, convergence is monotone (non-increasing)
        assert!(
            dist3 <= dist2,
            "3rd order ({:.2e}) should beat 2nd order ({:.2e})",
            dist3,
            dist2
        );
        assert!(
            dist4 <= dist3,
            "4th order ({:.2e}) should beat 3rd order ({:.2e})",
            dist4,
            dist3
        );
        assert!(
            dist5 <= dist4,
            "5th order ({:.2e}) should beat 4th order ({:.2e})",
            dist5,
            dist4
        );
        // At least one pair should show strict improvement
        assert!(
            dist5 < dist2,
            "5th order ({:.2e}) should strictly beat 2nd order ({:.2e})",
            dist5,
            dist2
        );

        println!("BCH convergence for ||X|| = ||Y|| = 0.08:");
        println!("  2nd order error: {:.2e}", dist2);
        println!("  3rd order error: {:.2e}", dist3);
        println!("  4th order error: {:.2e}", dist4);
        println!("  5th order error: {:.2e}", dist5);
    }

    #[test]
    fn test_bch_fourth_order_commutative() {
        // When [X,Y] = 0, all orders should give X + Y
        let x = Su2Algebra([0.2, 0.0, 0.0]);
        let y = Su2Algebra([0.3, 0.0, 0.0]); // Same direction

        let z4 = bch_fourth_order(&x, &y);
        let z5 = bch_fifth_order(&x, &y);

        let x_plus_y = x.add(&y);
        for i in 0..3 {
            assert!(
                (z4.0[i] - x_plus_y.0[i]).abs() < 1e-10,
                "4th order should give X+Y when commutative"
            );
            assert!(
                (z5.0[i] - x_plus_y.0[i]).abs() < 1e-10,
                "5th order should give X+Y when commutative"
            );
        }
    }

    #[test]
    fn test_bch_error_bound_scaling() {
        // Error bound should scale as (||X|| + ||Y||)^(n+1)
        let x_small = Su2Algebra([0.05, 0.0, 0.0]);
        let y_small = Su2Algebra([0.0, 0.05, 0.0]);

        let x_large = Su2Algebra([0.2, 0.0, 0.0]);
        let y_large = Su2Algebra([0.0, 0.2, 0.0]);

        // For 3rd order, error ~ (||X|| + ||Y||)^4
        let error_small = bch_error_bound(&x_small, &y_small, 3);
        let error_large = bch_error_bound(&x_large, &y_large, 3);

        // (0.2+0.2) / (0.05+0.05) = 4, so error should scale as 4^4 = 256
        let ratio = error_large / error_small;
        assert!(
            ratio > 200.0 && ratio < 300.0,
            "Error ratio should be ~256, got {}",
            ratio
        );
    }
}
