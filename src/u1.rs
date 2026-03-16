//! U(1): The Circle Group
//!
//! This module implements U(1), the group of complex numbers with unit modulus.
//! U(1) appears as the gauge group of electromagnetism and in many other contexts.
//!
//! # Mathematical Background
//!
//! ## Definition
//!
//! ```text
//! U(1) = { e^{iθ} ∈ ℂ | θ ∈ ℝ } ≅ S¹
//! ```
//!
//! The circle group: complex numbers z with |z| = 1, isomorphic to the unit circle.
//!
//! ## Group Structure
//!
//! - **Multiplication**: e^{iθ₁} · e^{iθ₂} = e^{i(θ₁+θ₂)}
//! - **Identity**: e^{i·0} = 1
//! - **Inverse**: (e^{iθ})^{-1} = e^{-iθ}
//! - **Abelian**: e^{iθ₁} · e^{iθ₂} = e^{iθ₂} · e^{iθ₁}
//!
//! ## Lie Algebra
//!
//! ```text
//! u(1) = iℝ (purely imaginary numbers)
//! Exponential map: exp(ia) = e^{ia} for a ∈ ℝ
//! ```
//!
//! ## Logarithm and Branch Cuts
//!
//! The logarithm map log: U(1) → u(1) is **multivalued** - for any θ, the values
//! θ, θ + 2π, θ + 4π, ... all represent the same group element e^{iθ}.
//!
//! **Principal Branch**: We use **θ ∈ [0, 2π)** with branch cut at θ = 0.
//!
//! ### Why [0, 2π) instead of (-π, π]?
//!
//! | Choice | Branch Cut | Pros | Cons |
//! |--------|------------|------|------|
//! | **[0, 2π)** (ours) | At θ = 0 (positive real axis) | Natural for angles, matches `from_angle()` normalization | Non-standard in complex analysis |
//! | (-π, π] (std) | At θ = π (negative real axis) | Standard in complex analysis | Asymmetric around identity |
//!
//! We chose [0, 2π) for **consistency**: our internal representation normalizes
//! angles to [0, 2π) in `from_angle()`, so `log(g)` returns exactly the stored value.
//!
//! ### Discontinuity Example
//!
//! ```text
//! lim_{ε→0⁺} log(e^{i(2π-ε)}) = 2π - ε  →  2π
//! lim_{ε→0⁺} log(e^{iε})      = ε       →  0
//!
//! Jump discontinuity of 2π at θ = 0 (the identity).
//! ```
//!
//! ### Practical Impact
//!
//! - **Gauge fixing**: When computing holonomy defects `∫ log(g)`, discontinuities
//!   at branch cuts can cause artificial jumps in the objective function.
//! - **Unwrapping**: For smooth paths that cross θ = 0, consider phase unwrapping
//!   algorithms that track accumulated 2π crossings.
//! - **Optimization**: If optimizing over U(1) connections, use angle differences
//!   `Δθ = θ₂ - θ₁` which are smooth, rather than logarithms.
//!
//! ## Physical Interpretation
//!
//! ### Electromagnetism
//!
//! - **Gauge field**: Vector potential `A_μ(x)`
//! - **Field strength**: `F_μν` = ∂_μ `A_ν` - ∂_ν `A_μ` (electromagnetic field)
//! - **Gauge transformation**: `A_μ` → `A_μ` + ∂_μ λ for λ: M → ℝ
//! - **Connection**: U(x,y) = exp(i ∫_x^y `A_μ` dx^μ) ∈ U(1)
//!
//! ### Wilson Loops
//!
//! For a closed path C:
//! ```text
//! W_C = exp(i ∮_C A_μ dx^μ) = exp(i Φ)
//! ```
//! where Φ is the magnetic flux through C (Aharonov-Bohm phase).
//!
//! ## Applications
//!
//! 1. **Lattice QED**: Discretized electromagnetic field theory
//! 2. **Phase synchronization**: Kuramoto model, coupled oscillators
//! 3. **Superconductivity**: Order parameter ψ = |ψ|e^{iθ}, θ ∈ U(1)
//! 4. **XY model**: Classical spin system with continuous symmetry
//! 5. **Clock synchronization**: Distributed network of oscillators

use crate::traits::AntiHermitianByConstruction;
use crate::{LieAlgebra, LieGroup};
use num_complex::Complex;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

/// Lie algebra u(1) ≅ ℝ
///
/// The Lie algebra of U(1) consists of purely imaginary numbers i·a where a ∈ ℝ.
/// We represent this as a 1-dimensional real vector space.
///
/// # Mathematical Structure
///
/// Elements of u(1) have the form:
/// ```text
/// X = i·a where a ∈ ℝ
/// ```
///
/// The exponential map is simply:
/// ```text
/// exp(i·a) = e^{ia} ∈ U(1)
/// ```
///
/// # Examples
///
/// ```
/// use lie_groups::u1::U1Algebra;
/// use lie_groups::traits::LieAlgebra;
///
/// // Create algebra element
/// let v = U1Algebra::from_components(&[1.5]);
///
/// // Scale
/// let w = v.scale(2.0);
/// assert_eq!(w.value(), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct U1Algebra(pub(crate) f64);

impl Add for U1Algebra {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Add<&U1Algebra> for U1Algebra {
    type Output = U1Algebra;
    fn add(self, rhs: &U1Algebra) -> U1Algebra {
        self + *rhs
    }
}

impl Add<U1Algebra> for &U1Algebra {
    type Output = U1Algebra;
    fn add(self, rhs: U1Algebra) -> U1Algebra {
        *self + rhs
    }
}

impl Add<&U1Algebra> for &U1Algebra {
    type Output = U1Algebra;
    fn add(self, rhs: &U1Algebra) -> U1Algebra {
        *self + *rhs
    }
}

impl Sub for U1Algebra {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for U1Algebra {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul<f64> for U1Algebra {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Mul<U1Algebra> for f64 {
    type Output = U1Algebra;
    fn mul(self, rhs: U1Algebra) -> U1Algebra {
        rhs * self
    }
}

impl U1Algebra {
    /// Create a new u(1) algebra element.
    ///
    /// The value `a` represents the element `i·a ∈ u(1)`.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Get the real value a from the algebra element i·a
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl LieAlgebra for U1Algebra {
    const DIM: usize = 1;

    #[inline]
    fn zero() -> Self {
        Self(0.0)
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    fn scale(&self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }

    #[inline]
    fn norm(&self) -> f64 {
        self.0.abs()
    }

    #[inline]
    fn basis_element(i: usize) -> Self {
        assert_eq!(i, 0, "U(1) algebra is 1-dimensional");
        Self(1.0)
    }

    #[inline]
    fn from_components(components: &[f64]) -> Self {
        assert_eq!(components.len(), 1, "u(1) has dimension 1");
        Self(components[0])
    }

    #[inline]
    fn to_components(&self) -> Vec<f64> {
        vec![self.0]
    }

    // u(1) is abelian: [X, Y] = 0 for all X, Y.
    // Stated explicitly rather than inherited from the trait default,
    // so the algebraic property is visible at the definition site.
    #[inline]
    fn bracket(&self, _other: &Self) -> Self {
        Self::zero()
    }

    #[inline]
    fn inner(&self, other: &Self) -> f64 {
        self.0 * other.0
    }
}

/// An element of U(1), the circle group
///
/// Represented as a phase angle θ ∈ [0, 2π), corresponding to e^{iθ}.
///
/// # Representation
///
/// We use the angle representation rather than storing the complex number
/// directly to avoid floating-point accumulation errors and to make the
/// group structure (angle addition) explicit.
///
/// # Examples
///
/// ```rust
/// use lie_groups::{LieGroup, U1};
/// use std::f64::consts::PI;
///
/// // Create elements
/// let g = U1::from_angle(0.5);
/// let h = U1::from_angle(0.3);
///
/// // Group multiplication (angle addition)
/// let product = g.compose(&h);
/// assert!((product.angle() - 0.8).abs() < 1e-10);
///
/// // Inverse (angle negation)
/// let g_inv = g.inverse();
/// assert!((g_inv.angle() - (2.0 * PI - 0.5)).abs() < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct U1 {
    /// Phase angle θ ∈ [0, 2π)
    ///
    /// Represents the group element e^{iθ}
    theta: f64,
}

impl U1 {
    /// Create U(1) element from angle in radians
    ///
    /// Automatically normalizes to [0, 2π)
    ///
    /// # Arguments
    ///
    /// * `theta` - Phase angle in radians (any real number)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    ///
    /// let g = U1::from_angle(0.5);
    /// assert!((g.angle() - 0.5).abs() < 1e-10);
    ///
    /// // Normalization
    /// let h = U1::from_angle(2.0 * std::f64::consts::PI + 0.3);
    /// assert!((h.angle() - 0.3).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn from_angle(theta: f64) -> Self {
        Self {
            theta: theta.rem_euclid(2.0 * PI),
        }
    }

    /// Create U(1) element from complex number
    ///
    /// Extracts the phase angle from z = re^{iθ}, ignoring the magnitude.
    ///
    /// # Arguments
    ///
    /// * `z` - Complex number (does not need to have unit modulus)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    /// use num_complex::Complex;
    ///
    /// let z = Complex::new(0.0, 1.0);  // i
    /// let g = U1::from_complex(z);
    /// assert!((g.angle() - std::f64::consts::PI / 2.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn from_complex(z: Complex<f64>) -> Self {
        Self::from_angle(z.arg())
    }

    /// Get the phase angle θ ∈ [0, 2π)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    ///
    /// let g = U1::from_angle(1.5);
    /// assert!((g.angle() - 1.5).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn angle(&self) -> f64 {
        self.theta
    }

    /// Convert to complex number e^{iθ}
    ///
    /// Returns the complex number representation with unit modulus.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    ///
    /// let g = U1::from_angle(std::f64::consts::PI / 2.0);  // π/2
    /// let z = g.to_complex();
    ///
    /// assert!(z.re.abs() < 1e-10);  // cos(π/2) ≈ 0
    /// assert!((z.im - 1.0).abs() < 1e-10);  // sin(π/2) = 1
    /// assert!((z.norm() - 1.0).abs() < 1e-10);  // |z| = 1
    /// ```
    #[must_use]
    pub fn to_complex(&self) -> Complex<f64> {
        Complex::new(self.theta.cos(), self.theta.sin())
    }

    /// Trace of the 1×1 matrix representation
    ///
    /// For U(1), the trace is just the complex number itself: Tr(e^{iθ}) = e^{iθ}
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    ///
    /// let g = U1::from_angle(0.0);  // Identity
    /// let tr = g.trace_complex();
    ///
    /// assert!((tr.re - 1.0).abs() < 1e-10);
    /// assert!(tr.im.abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn trace_complex(&self) -> Complex<f64> {
        self.to_complex()
    }

    /// Rotation generators for lattice updates
    ///
    /// Returns a small U(1) element for MCMC proposals
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Step size in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    ///
    /// let perturbation = U1::rotation(0.1);
    /// assert!((perturbation.angle() - 0.1).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn rotation(magnitude: f64) -> Self {
        Self::from_angle(magnitude)
    }

    /// Random U(1) element uniformly distributed on the circle
    ///
    /// Requires the `rand` feature (enabled by default).
    /// Samples θ uniformly from [0, 2π).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::U1;
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// let g = U1::random(&mut rng);
    ///
    /// assert!(g.angle() >= 0.0 && g.angle() < 2.0 * std::f64::consts::PI);
    /// ```
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let dist = Uniform::new(0.0, 2.0 * PI);
        Self::from_angle(dist.sample(rng))
    }

    /// Random small perturbation for MCMC proposals
    ///
    /// Samples θ uniformly from [-`step_size`, +`step_size`]
    ///
    /// # Arguments
    ///
    /// * `step_size` - Maximum angle deviation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::{U1, LieGroup};
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// let delta = U1::random_small(0.1, &mut rng);
    ///
    /// // Should be close to identity
    /// assert!(delta.distance_to_identity() <= 0.1);
    /// ```
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn random_small<R: rand::Rng>(step_size: f64, rng: &mut R) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let dist = Uniform::new(-step_size, step_size);
        let angle = dist.sample(rng);
        Self::from_angle(angle)
    }
}

impl approx::AbsDiffEq for U1Algebra {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        1e-10
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        (self.0 - other.0).abs() < epsilon
    }
}

impl approx::RelativeEq for U1Algebra {
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

impl fmt::Display for U1Algebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "u(1)({:.4})", self.0)
    }
}

/// Group multiplication: g₁ · g₂ (phase addition)
impl Mul<&U1> for &U1 {
    type Output = U1;
    fn mul(self, rhs: &U1) -> U1 {
        self.compose(rhs)
    }
}

impl Mul<&U1> for U1 {
    type Output = U1;
    fn mul(self, rhs: &U1) -> U1 {
        self.compose(rhs)
    }
}

impl Mul<U1> for U1 {
    type Output = U1;
    fn mul(self, rhs: U1) -> U1 {
        &self * &rhs
    }
}

impl MulAssign<&U1> for U1 {
    fn mul_assign(&mut self, rhs: &U1) {
        *self = self.compose(rhs);
    }
}

impl std::iter::Product for U1 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_angle(0.0), |acc, g| acc * g)
    }
}

impl<'a> std::iter::Product<&'a U1> for U1 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from_angle(0.0), |acc, g| acc * g)
    }
}

impl LieGroup for U1 {
    const MATRIX_DIM: usize = 1;

    type Algebra = U1Algebra;

    fn identity() -> Self {
        Self { theta: 0.0 }
    }

    fn compose(&self, other: &Self) -> Self {
        Self::from_angle(self.theta + other.theta)
    }

    fn inverse(&self) -> Self {
        Self::from_angle(-self.theta)
    }

    fn conjugate_transpose(&self) -> Self {
        // For U(1) (abelian group), conjugate transpose = complex conjugate = inverse
        self.inverse()
    }

    fn adjoint_action(&self, algebra_element: &U1Algebra) -> U1Algebra {
        // For abelian groups, the adjoint representation is trivial:
        // Ad_g(X) = g X g⁻¹ = X (since everything commutes)
        //
        // Mathematically: For U(1), [g, X] = 0 for all g, X
        // Therefore: g X g⁻¹ = g g⁻¹ X = X
        *algebra_element
    }

    fn distance_to_identity(&self) -> f64 {
        // Shortest arc distance on circle
        let normalized = self.theta.rem_euclid(2.0 * PI);
        let dist = normalized.min(2.0 * PI - normalized);
        dist.abs()
    }

    fn exp(tangent: &U1Algebra) -> Self {
        // exp(i·a) = e^{ia}
        // Represented as angle a (mod 2π)
        Self::from_angle(tangent.0)
    }

    fn log(&self) -> crate::error::LogResult<U1Algebra> {
        // For U(1), log(e^{iθ}) = iθ
        //
        // **Principal Branch Choice**: θ ∈ [0, 2π) with branch cut at θ = 0
        //
        // See module documentation for detailed discussion of branch cuts.
        //
        // # Mathematical Properties
        //
        // - **Surjectivity**: Every algebra element a ∈ [0, 2π) has exp(ia) ∈ U(1)
        // - **Branch cut**: Discontinuity at θ = 0 (positive real axis, identity)
        // - **Consistency**: Returns exactly the normalized angle from `from_angle()`
        //
        // # Examples
        //
        // ```
        // use lie_groups::{LieGroup, U1};
        // use std::f64::consts::PI;
        //
        // // Identity
        // let e = U1::identity();
        // assert!(e.log().unwrap().value().abs() < 1e-14);
        //
        // // π/2
        // let g = U1::from_angle(PI / 2.0);
        // assert!((g.log().unwrap().value() - PI / 2.0).abs() < 1e-14);
        //
        // // Branch cut demonstration: values near 2π
        // let g1 = U1::from_angle(2.0 * PI - 0.01);  // Just before branch cut
        // let g2 = U1::from_angle(0.01);              // Just after branch cut
        //
        // // log values jump discontinuously
        // assert!(g1.log().unwrap().value() > 6.0);   // ≈ 2π - 0.01
        // assert!(g2.log().unwrap().value() < 0.1);   // ≈ 0.01
        // // Jump size ≈ 2π
        // ```
        //
        // # Design Rationale
        //
        // Unlike SU(2) or SU(3), where log() can fail for elements far from identity
        // (returning `Err`), U(1) log is **always defined** because:
        // 1. U(1) is 1-dimensional (no exponential map singularities)
        // 2. Our angle representation is already the logarithm
        // 3. The covering map exp: ℝ → U(1) is a local diffeomorphism everywhere
        //
        // However, users should be aware of the **discontinuity** at θ = 0 when
        // using log in optimization or when computing angle differences.
        Ok(U1Algebra(self.theta))
    }
}

/// Display implementation shows the phase angle
impl std::fmt::Display for U1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "U1(θ={:.4})", self.theta)
    }
}

// ============================================================================
// Mathematical Property Implementations for U(1)
// ============================================================================

use crate::traits::{Abelian, Compact};

/// U(1) is compact.
///
/// The circle group {e^{iθ} : θ ∈ ℝ} is diffeomorphic to S¹.
/// All elements have unit modulus: |e^{iθ}| = 1.
///
/// # Topological Structure
///
/// U(1) ≅ SO(2) ≅ S¹ (the unit circle)
/// - Closed and bounded in ℂ
/// - Every sequence has a convergent subsequence (Bolzano-Weierstrass)
/// - Admits a finite Haar measure
///
/// # Physical Significance
///
/// Compactness of U(1) ensures:
/// - Well-defined Yang-Mills action
/// - Completely reducible representations
/// - Quantization of electric charge
impl Compact for U1 {}

/// U(1) is abelian.
///
/// Complex multiplication commutes: e^{iα} · e^{iβ} = e^{iβ} · e^{iα}
///
/// # Mathematical Structure
///
/// For all g, h ∈ U(1):
/// ```text
/// g · h = e^{i(α+β)} = e^{i(β+α)} = h · g
/// ```
///
/// This makes U(1) the structure group for **Maxwell's electromagnetism**.
///
/// # Type Safety Example
///
/// ```ignore
/// // This function requires an abelian gauge group
/// fn compute_chern_number<G: Abelian>(conn: &NetworkConnection<G>) -> i32 {
///     // Chern class computation requires commutativity
/// }
///
/// // ✅ Compiles: U(1) is abelian
/// let u1_conn = NetworkConnection::<U1>::new(graph);
/// compute_chern_number(&u1_conn);
///
/// // ❌ Won't compile: SU(2) is not abelian
/// let su2_conn = NetworkConnection::<SU2>::new(graph);
/// // compute_chern_number(&su2_conn);  // Compile error!
/// ```
///
/// # Physical Significance
///
/// - Gauge transformations commute (no self-interaction)
/// - Field strength is linear: F = dA (no `[A,A]` term)
/// - Maxwell's equations are linear PDEs
/// - Superposition principle holds
impl Abelian for U1 {}

// Note: U(1) is NOT Simple or SemiSimple
// - Abelian groups (except trivial group) are not simple
// - They contain proper normal subgroups
// - The center Z(U(1)) = U(1) itself

// ============================================================================
// Algebra Marker Traits
// ============================================================================

/// u(1) algebra elements are anti-Hermitian by construction.
///
/// The representation `U1Algebra(f64)` stores a real number `a` representing
/// the purely imaginary element `ia ∈ iℝ ⊂ ℂ`. Since `(ia)* = -ia`, these
/// are anti-Hermitian (as 1×1 complex matrices).
///
/// # Note: NOT `TracelessByConstruction`
///
/// Unlike su(n), the u(1) algebra is NOT traceless. For a 1×1 matrix `[ia]`,
/// the trace is ia ≠ 0 (for a ≠ 0). This is why U(1) ≠ SU(1).
///
/// The theorem `det(exp(X)) = exp(tr(X))` still holds, but gives:
/// - det(exp(ia)) = exp(ia) ≠ 1 in general
/// - This is correct: U(1) elements have |det| = 1, not det = 1
impl AntiHermitianByConstruction for U1Algebra {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_from_angle() {
        let g = U1::from_angle(0.5);
        assert!((g.angle() - 0.5).abs() < 1e-10);

        // Test normalization
        let h = U1::from_angle(2.0 * PI + 0.3);
        assert!((h.angle() - 0.3).abs() < 1e-10);

        // Negative angles
        let k = U1::from_angle(-0.5);
        assert!((k.angle() - (2.0 * PI - 0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_to_complex() {
        // Identity
        let e = U1::identity();
        let z = e.to_complex();
        assert!((z.re - 1.0).abs() < 1e-10);
        assert!(z.im.abs() < 1e-10);

        // π/2
        let g = U1::from_angle(PI / 2.0);
        let z = g.to_complex();
        assert!(z.re.abs() < 1e-10);
        assert!((z.im - 1.0).abs() < 1e-10);

        // All should have unit modulus
        let h = U1::from_angle(1.234);
        assert!((h.to_complex().norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_group_identity() {
        let g = U1::from_angle(1.5);
        let e = U1::identity();

        let g_e = g.compose(&e);
        let e_g = e.compose(&g);

        assert!((g_e.angle() - g.angle()).abs() < 1e-10);
        assert!((e_g.angle() - g.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_group_inverse() {
        let g = U1::from_angle(1.2);
        let g_inv = g.inverse();
        let product = g.compose(&g_inv);

        assert!(product.is_near_identity(1e-10));

        // Double inverse
        let g_inv_inv = g_inv.inverse();
        assert!((g_inv_inv.angle() - g.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_group_associativity() {
        let g1 = U1::from_angle(0.5);
        let g2 = U1::from_angle(1.2);
        let g3 = U1::from_angle(0.8);

        let left = g1.compose(&g2.compose(&g3));
        let right = g1.compose(&g2).compose(&g3);

        assert!((left.angle() - right.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_commutativity() {
        // U(1) is abelian
        let g = U1::from_angle(0.7);
        let h = U1::from_angle(1.3);

        let gh = g.compose(&h);
        let hg = h.compose(&g);

        assert!((gh.angle() - hg.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_identity() {
        let e = U1::identity();
        assert!(e.distance_to_identity().abs() < 1e-10);

        // Small angle
        let g1 = U1::from_angle(0.1);
        assert!((g1.distance_to_identity() - 0.1).abs() < 1e-10);

        // Angle > π should take shorter arc
        let g2 = U1::from_angle(1.9 * PI);
        assert!((g2.distance_to_identity() - 0.1 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_distance_symmetry() {
        let g = U1::from_angle(1.5);
        let h = U1::from_angle(0.8);

        let d_gh = g.distance(&h);
        let d_hg = h.distance(&g);

        assert!((d_gh - d_hg).abs() < 1e-10);
    }

    #[test]
    fn test_adjoint_equals_inverse() {
        // For abelian U(1), adjoint = inverse
        let g = U1::from_angle(1.7);
        let adj = g.conjugate_transpose();
        let inv = g.inverse();

        assert!((adj.angle() - inv.angle()).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random_distribution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Generate many random elements
        let samples: Vec<U1> = (0..1000).map(|_| U1::random(&mut rng)).collect();

        // Check all are in [0, 2π)
        for g in &samples {
            assert!(g.angle() >= 0.0 && g.angle() < 2.0 * PI);
        }

        // Check distribution is roughly uniform (basic test)
        let mean_angle = samples.iter().map(super::U1::angle).sum::<f64>() / samples.len() as f64;
        assert!((mean_angle - PI).abs() < 0.2); // Should be near π
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random_small() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let step_size = 0.1;

        let samples: Vec<U1> = (0..100)
            .map(|_| U1::random_small(step_size, &mut rng))
            .collect();

        // Most should be close to identity
        let close_to_identity = samples
            .iter()
            .filter(|g| g.distance_to_identity() < 0.3)
            .count();

        assert!(close_to_identity > 80); // At least 80% within 3σ
    }

    #[test]
    fn test_trace() {
        let g = U1::from_angle(PI / 3.0);
        let tr = g.trace_complex();

        // Tr(e^{iθ}) = e^{iθ}
        let expected = g.to_complex();
        assert!((tr.re - expected.re).abs() < 1e-10);
        assert!((tr.im - expected.im).abs() < 1e-10);
    }

    // ========================================================================
    // Logarithm Map Tests
    // ========================================================================

    #[test]
    fn test_log_identity() {
        use crate::traits::LieGroup;

        let e = U1::identity();
        let log_e = e.log().unwrap();

        // log(identity) = 0
        assert!(log_e.norm() < 1e-14);
    }

    #[test]
    fn test_log_exp_roundtrip() {
        use crate::traits::LieGroup;

        // For small algebra elements, log(exp(X)) ≈ X
        let x = U1Algebra(0.5);
        let g = U1::exp(&x);
        let x_recovered = g.log().unwrap();

        // Should match original
        assert!((x_recovered.0 - x.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        use crate::traits::LieGroup;

        // For group elements near identity, exp(log(g)) ≈ g
        let g = U1::from_angle(0.7);
        let x = g.log().unwrap();
        let g_recovered = U1::exp(&x);

        // Should match original
        assert!((g_recovered.angle() - g.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_log_branch_cut_at_zero() {
        use crate::traits::LieGroup;

        // Test the branch cut at θ = 0 (identity)
        // Values just before 2π should log to ~2π
        // Values just after 0 should log to ~0
        // Demonstrating the discontinuity

        let eps = 0.01;

        // Just before the branch cut (θ ≈ 2π - ε)
        let g_before = U1::from_angle(2.0 * PI - eps);
        let log_before = g_before.log().unwrap().value();

        // Just after the branch cut (θ ≈ 0 + ε)
        let g_after = U1::from_angle(eps);
        let log_after = g_after.log().unwrap().value();

        // Verify we're on the correct sides
        assert!(
            (log_before - (2.0 * PI - eps)).abs() < 1e-10,
            "Expected log(e^{{i(2π-ε)}}) ≈ 2π-ε, got {}",
            log_before
        );
        assert!(
            (log_after - eps).abs() < 1e-10,
            "Expected log(e^{{iε}}) ≈ ε, got {}",
            log_after
        );

        // The jump discontinuity is ~2π
        let jump = log_before - log_after;
        assert!(
            (jump - 2.0 * PI).abs() < 0.1,
            "Expected discontinuity ≈ 2π, got {}",
            jump
        );
    }

    #[test]
    fn test_log_principal_branch_coverage() {
        use crate::traits::LieGroup;

        // Verify log returns values in [0, 2π) for all inputs
        let test_angles = vec![
            0.0,
            PI / 4.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
            2.0 * PI - 0.001,
        ];

        for theta in test_angles {
            let g = U1::from_angle(theta);
            let log_value = g.log().unwrap().value();

            // Should be in principal branch [0, 2π)
            assert!(
                (0.0..2.0 * PI).contains(&log_value),
                "log value {} outside principal branch [0, 2π) for θ = {}",
                log_value,
                theta
            );

            // Should match the stored angle
            assert!(
                (log_value - g.angle()).abs() < 1e-14,
                "log inconsistent with angle() for θ = {}",
                theta
            );
        }
    }

    // ========================================================================
    // Property-Based Tests for Group Axioms
    // ========================================================================
    //
    // These tests use proptest to verify that U(1) satisfies the
    // mathematical axioms of a Lie group for randomly generated elements.
    //
    // U(1) is **abelian**, so we also test commutativity.
    //
    // Run with: cargo test --features property-tests

    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    /// Strategy for generating arbitrary U(1) elements.
    ///
    /// We generate U(1) elements by sampling angles uniformly from [0, 2π).
    #[cfg(feature = "proptest")]
    fn arb_u1() -> impl Strategy<Value = U1> {
        (0.0..2.0 * PI).prop_map(U1::from_angle)
    }

    #[cfg(feature = "proptest")]
    proptest! {
        /// **Group Axiom 1: Identity Element**
        ///
        /// For all g ∈ U(1):
        /// - e · g = g (left identity)
        /// - g · e = g (right identity)
        #[test]
        fn prop_identity_axiom(g in arb_u1()) {
            let e = U1::identity();

            // Left identity: e · g = g
            let left = e.compose(&g);
            prop_assert!(
                (left.angle() - g.angle()).abs() < 1e-10,
                "Left identity failed: e·g != g"
            );

            // Right identity: g · e = g
            let right = g.compose(&e);
            prop_assert!(
                (right.angle() - g.angle()).abs() < 1e-10,
                "Right identity failed: g·e != g"
            );
        }

        /// **Group Axiom 2: Inverse Element**
        ///
        /// For all g ∈ U(1):
        /// - g · g⁻¹ = e (right inverse)
        /// - g⁻¹ · g = e (left inverse)
        #[test]
        fn prop_inverse_axiom(g in arb_u1()) {
            let g_inv = g.inverse();

            // Right inverse: g · g⁻¹ = e
            let right_product = g.compose(&g_inv);
            prop_assert!(
                right_product.is_near_identity(1e-10),
                "Right inverse failed: g·g⁻¹ != e, distance = {}",
                right_product.distance_to_identity()
            );

            // Left inverse: g⁻¹ · g = e
            let left_product = g_inv.compose(&g);
            prop_assert!(
                left_product.is_near_identity(1e-10),
                "Left inverse failed: g⁻¹·g != e, distance = {}",
                left_product.distance_to_identity()
            );
        }

        /// **Group Axiom 3: Associativity**
        ///
        /// For all g₁, g₂, g₃ ∈ U(1):
        /// - (g₁ · g₂) · g₃ = g₁ · (g₂ · g₃)
        #[test]
        fn prop_associativity(g1 in arb_u1(), g2 in arb_u1(), g3 in arb_u1()) {
            // Left association: (g₁ · g₂) · g₃
            let left_assoc = g1.compose(&g2).compose(&g3);

            // Right association: g₁ · (g₂ · g₃)
            let right_assoc = g1.compose(&g2.compose(&g3));

            prop_assert!(
                (left_assoc.angle() - right_assoc.angle()).abs() < 1e-10,
                "Associativity failed: (g₁·g₂)·g₃ != g₁·(g₂·g₃)"
            );
        }

        /// **Abelian Property: Commutativity**
        ///
        /// For all g, h ∈ U(1):
        /// - g · h = h · g
        ///
        /// U(1) is abelian (commutative), which is NOT true for SU(2).
        /// This is why U(1) is used for electromagnetism (no self-interaction).
        #[test]
        fn prop_commutativity(g in arb_u1(), h in arb_u1()) {
            let gh = g.compose(&h);
            let hg = h.compose(&g);

            prop_assert!(
                (gh.angle() - hg.angle()).abs() < 1e-10,
                "Commutativity failed: g·h != h·g, g·h = {}, h·g = {}",
                gh.angle(),
                hg.angle()
            );
        }

        /// **Exponential Map Property**
        ///
        /// For U(1), exp: u(1) → U(1) is the exponential map.
        /// We verify that exp(a) · exp(b) = exp(a + b) for scalars a, b ∈ ℝ.
        ///
        /// This is the Baker-Campbell-Hausdorff formula specialized to the
        /// abelian case (where it simplifies dramatically).
        #[test]
        fn prop_exponential_map_homomorphism(a in -PI..PI, b in -PI..PI) {
            use crate::traits::LieGroup;

            let exp_a = U1::exp(&U1Algebra(a));
            let exp_b = U1::exp(&U1Algebra(b));
            let exp_sum = U1::exp(&U1Algebra(a + b));

            let product = exp_a.compose(&exp_b);

            prop_assert!(
                product.distance(&exp_sum) < 1e-10,
                "Exponential map homomorphism failed: exp(a)·exp(b) != exp(a+b)"
            );
        }

        /// **Compactness: Angles Wrap Around**
        ///
        /// U(1) is compact (isomorphic to S¹, the circle).
        /// This means angles wrap around modulo 2π.
        #[test]
        fn prop_compactness_angle_wrapping(theta in -10.0 * PI..10.0 * PI) {
            let g = U1::from_angle(theta);

            // All angles should be normalized to [0, 2π)
            prop_assert!(
                g.angle() >= 0.0 && g.angle() < 2.0 * PI,
                "Angle not normalized: θ = {}",
                g.angle()
            );

            // Adding 2πn should give the same element
            let g_plus_2pi = U1::from_angle(theta + 2.0 * PI);
            prop_assert!(
                (g.angle() - g_plus_2pi.angle()).abs() < 1e-10,
                "Adding 2π changed the element"
            );
        }
    }
}
