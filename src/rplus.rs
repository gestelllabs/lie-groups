//! ℝ⁺: The Positive Reals (Multiplicative Scaling Group)
//!
//! This module implements ℝ⁺, the group of positive real numbers under multiplication.
//! ℝ⁺ is the symmetry group for **scaling transformations** in applications like:
//! - Volatility surface analysis (IV scaling)
//! - Image processing (contrast/brightness)
//! - Economic growth models (multiplicative factors)
//!
//! # Mathematical Background
//!
//! ## Definition
//!
//! ```text
//! ℝ⁺ = (0, ∞) with multiplication
//! ```
//!
//! The positive reals form a Lie group under multiplication.
//!
//! ## Group Structure
//!
//! - **Multiplication**: a · b (standard multiplication)
//! - **Identity**: 1
//! - **Inverse**: a⁻¹ = 1/a
//! - **Abelian**: a · b = b · a
//!
//! ## Lie Algebra
//!
//! ```text
//! Lie(ℝ⁺) ≅ ℝ (the real line with addition)
//! Exponential map: exp(x) = eˣ
//! Logarithm: log(a) = ln(a)
//! ```
//!
//! ## Topological Properties
//!
//! - **Non-compact**: Unlike U(1), ℝ⁺ extends to infinity
//! - **Simply connected**: π₁(ℝ⁺) = 0 (no winding numbers)
//! - **Contractible**: Homotopy equivalent to a point
//!
//! ## Isomorphism with (ℝ, +)
//!
//! The exponential map provides a Lie group isomorphism:
//! ```text
//! exp: (ℝ, +) → (ℝ⁺, ×)
//! log: (ℝ⁺, ×) → (ℝ, +)
//! ```
//!
//! ## Applications
//!
//! 1. **Volatility surfaces**: `IV(K,T) = λ · IV_eq(K,T)` models level shifts
//! 2. **Scale invariance**: Physical systems with no preferred scale
//! 3. **Log-returns**: `r = log(S_t/S_0)` lives in the Lie algebra

use crate::{LieAlgebra, LieGroup};
use std::fmt;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

// ============================================================================
// Lie Algebra: ℝ (with addition)
// ============================================================================

/// Lie algebra of ℝ⁺, isomorphic to (ℝ, +)
///
/// Elements represent infinitesimal scaling factors. The exponential map
/// converts these to finite scalings: exp(x) = eˣ.
///
/// # Examples
///
/// ```
/// use lie_groups::rplus::RPlusAlgebra;
/// use lie_groups::traits::LieAlgebra;
///
/// let v = RPlusAlgebra::from_components(&[0.5]);
/// let w = v.scale(2.0);
/// assert!((w.value() - 1.0).abs() < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RPlusAlgebra(pub(crate) f64);

impl Add for RPlusAlgebra {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Add<&RPlusAlgebra> for RPlusAlgebra {
    type Output = RPlusAlgebra;
    fn add(self, rhs: &RPlusAlgebra) -> RPlusAlgebra {
        self + *rhs
    }
}

impl Add<RPlusAlgebra> for &RPlusAlgebra {
    type Output = RPlusAlgebra;
    fn add(self, rhs: RPlusAlgebra) -> RPlusAlgebra {
        *self + rhs
    }
}

impl Add<&RPlusAlgebra> for &RPlusAlgebra {
    type Output = RPlusAlgebra;
    fn add(self, rhs: &RPlusAlgebra) -> RPlusAlgebra {
        *self + *rhs
    }
}

impl Sub for RPlusAlgebra {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for RPlusAlgebra {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul<f64> for RPlusAlgebra {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Mul<RPlusAlgebra> for f64 {
    type Output = RPlusAlgebra;
    fn mul(self, rhs: RPlusAlgebra) -> RPlusAlgebra {
        rhs * self
    }
}

impl RPlusAlgebra {
    /// Create a new ℝ⁺ algebra element.
    ///
    /// The value represents an infinitesimal scaling factor.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Get the real value
    #[must_use]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl LieAlgebra for RPlusAlgebra {
    const DIM: usize = 1;

    fn zero() -> Self {
        Self(0.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn scale(&self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }

    fn norm(&self) -> f64 {
        self.0.abs()
    }

    fn basis_element(i: usize) -> Self {
        assert_eq!(i, 0, "ℝ⁺ algebra is 1-dimensional");
        Self(1.0)
    }

    fn from_components(components: &[f64]) -> Self {
        assert_eq!(components.len(), 1, "ℝ⁺ algebra has dimension 1");
        Self(components[0])
    }

    fn to_components(&self) -> Vec<f64> {
        vec![self.0]
    }

    // ℝ⁺ is abelian: [X, Y] = 0 for all X, Y.
    fn bracket(&self, _other: &Self) -> Self {
        Self::zero()
    }

    #[inline]
    fn inner(&self, other: &Self) -> f64 {
        self.0 * other.0
    }
}

// ============================================================================
// Lie Group: ℝ⁺ (positive reals under multiplication)
// ============================================================================

/// An element of ℝ⁺, the multiplicative group of positive reals
///
/// Represented as a positive real number x > 0.
///
/// # Representation
///
/// We store the value directly (not logarithm) for intuitive interpretation.
/// The logarithm is computed when needed for Lie algebra operations.
///
/// # Examples
///
/// ```
/// use lie_groups::{LieGroup, RPlus};
///
/// // Create elements
/// let g = RPlus::from_value(2.0);
/// let h = RPlus::from_value(3.0);
///
/// // Group multiplication
/// let product = g.compose(&h);
/// assert!((product.value() - 6.0).abs() < 1e-10);
///
/// // Inverse
/// let g_inv = g.inverse();
/// assert!((g_inv.value() - 0.5).abs() < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RPlus {
    /// The positive real value x > 0
    value: f64,
}

impl RPlus {
    /// Create ℝ⁺ element from a positive real number
    ///
    /// # Panics
    ///
    /// Panics if `value <= 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::RPlus;
    ///
    /// let g = RPlus::from_value(2.5);
    /// assert!((g.value() - 2.5).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn from_value(value: f64) -> Self {
        assert!(value > 0.0, "ℝ⁺ elements must be positive, got {}", value);
        Self { value }
    }

    /// Create ℝ⁺ element, clamping to positive range
    ///
    /// For robustness when value might be near zero or negative due to
    /// numerical errors. Clamps to a small positive value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::RPlus;
    ///
    /// let g = RPlus::from_value_clamped(-0.1);
    /// assert!(g.value() > 0.0);
    /// ```
    #[must_use]
    pub fn from_value_clamped(value: f64) -> Self {
        Self {
            value: value.max(1e-10),
        }
    }

    /// Get the positive real value
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Create from logarithm (exponential map)
    ///
    /// Given x ∈ ℝ (Lie algebra), returns eˣ ∈ ℝ⁺.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::RPlus;
    ///
    /// let g = RPlus::from_log(0.0);  // e⁰ = 1
    /// assert!((g.value() - 1.0).abs() < 1e-10);
    ///
    /// let h = RPlus::from_log(1.0);  // e¹ ≈ 2.718
    /// assert!((h.value() - std::f64::consts::E).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn from_log(log_value: f64) -> Self {
        Self {
            value: log_value.exp(),
        }
    }

    /// Scaling perturbation for optimization
    ///
    /// Returns a small scaling factor for gradient descent updates.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Step size in log-space
    #[must_use]
    pub fn scaling(magnitude: f64) -> Self {
        Self::from_log(magnitude)
    }

    /// Random ℝ⁺ element (log-normal distribution)
    ///
    /// Requires the `rand` feature (enabled by default).
    /// Samples from log-normal with given mean and std in log-space.
    ///
    /// # Panics
    ///
    /// Panics if `log_std` is negative or NaN.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn random<R: rand::Rng>(rng: &mut R, log_mean: f64, log_std: f64) -> Self {
        use rand::distributions::Distribution;
        use rand_distr::Normal;
        let normal =
            Normal::new(log_mean, log_std).expect("log_std must be non-negative and finite");
        Self::from_log(normal.sample(rng))
    }
}

impl approx::AbsDiffEq for RPlusAlgebra {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        1e-10
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        (self.0 - other.0).abs() < epsilon
    }
}

impl approx::RelativeEq for RPlusAlgebra {
    fn default_max_relative() -> Self::Epsilon {
        1e-10
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        approx::RelativeEq::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl fmt::Display for RPlusAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r+({:.4})", self.0)
    }
}

/// Group multiplication: g₁ · g₂ (real multiplication)
impl Mul<&RPlus> for &RPlus {
    type Output = RPlus;
    fn mul(self, rhs: &RPlus) -> RPlus {
        self.compose(rhs)
    }
}

impl Mul<&RPlus> for RPlus {
    type Output = RPlus;
    fn mul(self, rhs: &RPlus) -> RPlus {
        self.compose(rhs)
    }
}

impl Mul<RPlus> for RPlus {
    type Output = RPlus;
    fn mul(self, rhs: RPlus) -> RPlus {
        &self * &rhs
    }
}

impl MulAssign<&RPlus> for RPlus {
    fn mul_assign(&mut self, rhs: &RPlus) {
        *self = self.compose(rhs);
    }
}

impl std::iter::Product for RPlus {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_value(1.0), |acc, g| acc * g)
    }
}

impl<'a> std::iter::Product<&'a RPlus> for RPlus {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from_value(1.0), |acc, g| &acc * g)
    }
}

impl LieGroup for RPlus {
    const MATRIX_DIM: usize = 1;

    type Algebra = RPlusAlgebra;

    fn identity() -> Self {
        Self { value: 1.0 }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            value: self.value * other.value,
        }
    }

    fn inverse(&self) -> Self {
        Self {
            value: 1.0 / self.value,
        }
    }

    fn conjugate_transpose(&self) -> Self {
        // Convention: conjugate_transpose() returns g⁻¹ for consistency with
        // unitary matrix groups where g† = g⁻¹.
        self.inverse()
    }

    fn adjoint_action(&self, algebra_element: &RPlusAlgebra) -> RPlusAlgebra {
        // For abelian groups, Ad_g(X) = X
        *algebra_element
    }

    fn distance_to_identity(&self) -> f64 {
        // Distance in log-space: |log(x)|
        self.value.ln().abs()
    }

    fn exp(tangent: &RPlusAlgebra) -> Self {
        // exp: ℝ → ℝ⁺, exp(x) = eˣ
        Self::from_log(tangent.0)
    }

    fn log(&self) -> crate::error::LogResult<RPlusAlgebra> {
        // log: ℝ⁺ → ℝ, log(x) = ln(x)
        //
        // Unlike U(1), there's no branch cut ambiguity.
        // The logarithm is single-valued on ℝ⁺.
        Ok(RPlusAlgebra(self.value.ln()))
    }
}

/// Display implementation
impl std::fmt::Display for RPlus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ℝ⁺({:.4})", self.value)
    }
}

// ============================================================================
// Mathematical Property Implementations
// ============================================================================

// Note: ℝ⁺ is NOT compact (extends to infinity), so we don't implement Compact.

/// ℝ⁺ is abelian: a · b = b · a for all positive reals.
impl crate::Abelian for RPlus {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let e = RPlus::identity();
        assert!((e.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let a = RPlus::from_value(2.0);
        let b = RPlus::from_value(3.0);
        let product = a.compose(&b);
        assert!((product.value() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = RPlus::from_value(4.0);
        let a_inv = a.inverse();
        assert!((a_inv.value() - 0.25).abs() < 1e-10);

        // a * a⁻¹ = 1
        let product = a.compose(&a_inv);
        assert!((product.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let x = RPlusAlgebra(1.5);
        let g = RPlus::exp(&x);
        let x_back = g.log().unwrap();
        assert!((x_back.value() - x.value()).abs() < 1e-10);
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let g = RPlus::from_value(std::f64::consts::E);
        let x = g.log().unwrap();
        let g_back = RPlus::exp(&x);
        assert!((g_back.value() - g.value()).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_identity() {
        let e = RPlus::identity();
        assert!(e.distance_to_identity() < 1e-10);

        let g = RPlus::from_value(std::f64::consts::E);
        assert!((g.distance_to_identity() - 1.0).abs() < 1e-10);

        // Symmetric: distance(2) = distance(0.5)
        let a = RPlus::from_value(2.0);
        let b = RPlus::from_value(0.5);
        assert!((a.distance_to_identity() - b.distance_to_identity()).abs() < 1e-10);
    }

    #[test]
    fn test_algebra_operations() {
        let x = RPlusAlgebra(1.0);
        let y = RPlusAlgebra(2.0);

        let sum = x.add(&y);
        assert!((sum.value() - 3.0).abs() < 1e-10);

        let scaled = x.scale(3.0);
        assert!((scaled.value() - 3.0).abs() < 1e-10);

        let zero = RPlusAlgebra::zero();
        assert!(zero.value().abs() < 1e-10);
    }

    #[test]
    fn test_abelian_property() {
        let a = RPlus::from_value(2.0);
        let b = RPlus::from_value(3.0);

        let ab = a.compose(&b);
        let ba = b.compose(&a);

        assert!((ab.value() - ba.value()).abs() < 1e-10);
    }

    #[test]
    fn test_from_value_clamped() {
        // Negative values get clamped to positive
        let g = RPlus::from_value_clamped(-0.5);
        assert!(g.value() > 0.0);
        assert!(g.value() >= 1e-10);

        // Zero gets clamped
        let h = RPlus::from_value_clamped(0.0);
        assert!(h.value() > 0.0);

        // Positive values pass through unchanged
        let k = RPlus::from_value_clamped(2.5);
        assert!((k.value() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_scaling() {
        // Scaling with magnitude 0 gives identity
        let s0 = RPlus::scaling(0.0);
        assert!((s0.value() - 1.0).abs() < 1e-10);

        // Scaling with positive magnitude gives e^mag
        let s1 = RPlus::scaling(1.0);
        assert!((s1.value() - std::f64::consts::E).abs() < 1e-10);

        // Negative magnitude gives shrinking
        let sm1 = RPlus::scaling(-1.0);
        assert!((sm1.value() - 1.0 / std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Random samples should be positive
        for _ in 0..100 {
            let g = RPlus::random(&mut rng, 0.0, 1.0);
            assert!(g.value() > 0.0);
        }

        // Mean of log should be approximately log_mean
        let mut log_sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            let g = RPlus::random(&mut rng, 0.5, 0.1);
            log_sum += g.value().ln();
        }
        let log_mean = log_sum / n as f64;
        assert!(
            (log_mean - 0.5).abs() < 0.1,
            "Log mean should be approximately 0.5"
        );
    }

    #[test]
    fn test_adjoint() {
        // For abelian groups, adjoint = inverse
        let g = RPlus::from_value(3.0);
        let adj = g.conjugate_transpose();
        assert!(
            (adj.value() - 1.0 / 3.0).abs() < 1e-10,
            "Adjoint should equal inverse"
        );
    }

    #[test]
    fn test_adjoint_action() {
        // For abelian groups, Ad_g(X) = X
        let g = RPlus::from_value(5.0);
        let x = RPlusAlgebra(2.5);
        let result = g.adjoint_action(&x);
        assert!((result.value() - x.value()).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let g = RPlus::from_value(2.5);
        let s = format!("{}", g);
        assert!(s.contains("2.5"));
        assert!(s.contains("ℝ⁺"));
    }

    #[test]
    fn test_algebra_dim() {
        assert_eq!(RPlusAlgebra::DIM, 1);
    }

    #[test]
    fn test_algebra_basis_element() {
        let basis = RPlusAlgebra::basis_element(0);
        assert!((basis.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "1-dimensional")]
    fn test_algebra_basis_element_out_of_bounds() {
        let _ = RPlusAlgebra::basis_element(1);
    }

    #[test]
    fn test_algebra_from_to_components() {
        let x = RPlusAlgebra::from_components(&[3.5]);
        assert!((x.value() - 3.5).abs() < 1e-10);

        let comps = x.to_components();
        assert_eq!(comps.len(), 1);
        assert!((comps[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "dimension 1")]
    fn test_algebra_from_components_wrong_dim() {
        let _ = RPlusAlgebra::from_components(&[1.0, 2.0]);
    }

    #[test]
    fn test_algebra_norm() {
        let x = RPlusAlgebra(-3.0);
        assert!((x.norm() - 3.0).abs() < 1e-10);

        let y = RPlusAlgebra(2.5);
        assert!((y.norm() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_from_log() {
        // from_log(0) = e^0 = 1
        let g = RPlus::from_log(0.0);
        assert!((g.value() - 1.0).abs() < 1e-10);

        // from_log(1) = e^1 = e
        let h = RPlus::from_log(1.0);
        assert!((h.value() - std::f64::consts::E).abs() < 1e-10);

        // from_log(-1) = e^{-1} = 1/e
        let k = RPlus::from_log(-1.0);
        assert!((k.value() - 1.0 / std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_group_dim() {
        assert_eq!(RPlus::MATRIX_DIM, 1);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn test_from_value_panics_on_zero() {
        let _ = RPlus::from_value(0.0);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn test_from_value_panics_on_negative() {
        let _ = RPlus::from_value(-1.0);
    }
}
