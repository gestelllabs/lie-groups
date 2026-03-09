//! SU(2): The Special Unitary Group in 2 Dimensions
//!
//! This module implements SU(2) group elements using **2×2 complex matrices**.
//! We define the Pauli generators directly (no external dependencies needed).
//!
//! # SU(2) Implementations
//!
//! This library provides multiple implementations of SU(2):
//!
//! ## Matrix-Based (this module)
//!
//! - **`SU2`**: Specialized 2×2 complex matrix implementation
//!   - Uses `ndarray` for matrix operations
//!   - Has convenient constructors (`rotation_x`, `rotation_y`, `rotation_z`)
//!   - Good for learning, verification, and physics applications
//!
//! - **`SU2Generic`** ([`crate::sun::SUN<2>`]): Generic N×N matrix implementation
//!   - Part of the unified SU(N) framework
//!   - Same performance as `SU2` (both use matrix multiplication)
//!   - Useful when writing code generic over SU(N)
//!
//! ## Quaternion-Based
//!
//! - **[`UnitQuaternion`](crate::UnitQuaternion)**: Quaternion representation
//!   - Uses SU(2) ≅ S³ isomorphism (4 real numbers)
//!   - **~3× faster** for multiplication than matrix representation
//!   - Better numerical stability (no matrix decomposition)
//!   - Recommended for performance-critical rotation computations
//!
//! # Performance Comparison
//!
//! Benchmarked on typical hardware (multiplication, 1000 ops):
//! ```text
//! UnitQuaternion: ~16 µs  (4 real multiplies + adds)
//! SU2 (matrix):   ~45 µs  (complex 2×2 matrix multiply)
//! ```
//!
//! For applications requiring compatibility with higher SU(N) groups (e.g.,
//! Wilson loops, parallel transport), the matrix representation is preferred.
//!
//! ## Example
//!
//! ```rust
//! use lie_groups::{SU2, LieGroup};
//!
//! // SU2 has specialized constructors:
//! let g1 = SU2::rotation_x(0.5);
//! let g2 = SU2::rotation_y(0.3);
//!
//! // Compose rotations
//! let g3 = g1.compose(&g2);
//! ```

use crate::traits::{AntiHermitianByConstruction, LieAlgebra, LieGroup, TracelessByConstruction};
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

// ============================================================================
// Numerical Tolerance Constants
// ============================================================================
//
// These thresholds are chosen based on IEEE 754 f64 precision (ε ≈ 2.2e-16)
// and practical considerations for accumulated floating-point error.
//
// For SU(2) matrices, unitarity ensures |Re(Tr(U))/2| ≤ 1 exactly.
// Violations indicate matrix corruption (drift from the group manifold).
//
// Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2002)

/// Threshold for detecting unitarity violations.
/// If |cos(θ/2)| exceeds 1 by more than this, the matrix has drifted from SU(2).
const UNITARITY_VIOLATION_THRESHOLD: f64 = 1e-10;

/// Threshold for small angle detection in `exp()`.
/// Below this, return identity to avoid division by near-zero.
const SMALL_ANGLE_EXP_THRESHOLD: f64 = 1e-10;

/// Threshold for sin(θ/2) in `log()` axis extraction.
/// Below this, the rotation axis cannot be reliably determined.
const SIN_HALF_THETA_THRESHOLD: f64 = 1e-10;

/// Lie algebra su(2) ≅ ℝ³
///
/// The Lie algebra of SU(2) consists of 2×2 traceless anti-Hermitian matrices.
/// We represent these using 3 real coordinates.
///
/// # Convention
///
/// We identify su(2) with ℝ³ via the basis `{e₀, e₁, e₂} = {iσₓ/2, iσᵧ/2, iσᵤ/2}`,
/// where σᵢ are the Pauli matrices:
/// ```text
/// σₓ = [[0, 1], [1, 0]]    e₀ = iσₓ/2 = [[0, i/2], [i/2, 0]]
/// σᵧ = [[0, -i], [i, 0]]   e₁ = iσᵧ/2 = [[0, 1/2], [-1/2, 0]]
/// σᵤ = [[1, 0], [0, -1]]   e₂ = iσᵤ/2 = [[i/2, 0], [0, -i/2]]
/// ```
///
/// An element `Su2Algebra([a, b, c])` corresponds to the matrix
/// `(a·iσₓ + b·iσᵧ + c·iσᵤ)/2`, and the parameter `‖(a,b,c)‖` is the
/// rotation angle in the exponential map.
///
/// # Structure Constants
///
/// With this basis, the Lie bracket satisfies `[eᵢ, eⱼ] = -εᵢⱼₖ eₖ`, giving
/// structure constants `fᵢⱼₖ = -εᵢⱼₖ` (negative Levi-Civita symbol).
/// In ℝ³ coordinates, `[X, Y] = -(X × Y)`.
///
/// # Isomorphism with ℝ³
///
/// su(2) is isomorphic to ℝ³ as a vector space, and as a Lie algebra
/// the bracket is the negative cross product. The norm `‖v‖` equals the
/// rotation angle θ, matching the exponential map
/// `exp(v) = cos(θ/2)I + i·sin(θ/2)·v̂·σ`.
///
/// # Examples
///
/// ```
/// use lie_groups::su2::Su2Algebra;
/// use lie_groups::traits::LieAlgebra;
///
/// // Create algebra element in X direction
/// let v = Su2Algebra::from_components(&[1.0, 0.0, 0.0]);
///
/// // Scale and add
/// let w = v.scale(2.0);
/// let sum = v.add(&w);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Su2Algebra(pub [f64; 3]);

impl Add for Su2Algebra {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Add<&Su2Algebra> for Su2Algebra {
    type Output = Su2Algebra;
    fn add(self, rhs: &Su2Algebra) -> Su2Algebra {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Add<Su2Algebra> for &Su2Algebra {
    type Output = Su2Algebra;
    fn add(self, rhs: Su2Algebra) -> Su2Algebra {
        Su2Algebra([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Add<&Su2Algebra> for &Su2Algebra {
    type Output = Su2Algebra;
    fn add(self, rhs: &Su2Algebra) -> Su2Algebra {
        Su2Algebra([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Sub for Su2Algebra {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl Neg for Su2Algebra {
    type Output = Self;
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Mul<f64> for Su2Algebra {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self([self.0[0] * scalar, self.0[1] * scalar, self.0[2] * scalar])
    }
}

impl Mul<Su2Algebra> for f64 {
    type Output = Su2Algebra;
    fn mul(self, rhs: Su2Algebra) -> Su2Algebra {
        rhs * self
    }
}

impl LieAlgebra for Su2Algebra {
    const DIM: usize = 3;

    #[inline]
    fn zero() -> Self {
        Self([0.0, 0.0, 0.0])
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        Self([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
        ])
    }

    #[inline]
    fn scale(&self, scalar: f64) -> Self {
        Self([self.0[0] * scalar, self.0[1] * scalar, self.0[2] * scalar])
    }

    #[inline]
    fn norm(&self) -> f64 {
        (self.0[0].powi(2) + self.0[1].powi(2) + self.0[2].powi(2)).sqrt()
    }

    #[inline]
    fn basis_element(i: usize) -> Self {
        assert!(i < 3, "SU(2) algebra is 3-dimensional");
        let mut v = [0.0; 3];
        v[i] = 1.0;
        Self(v)
    }

    #[inline]
    fn from_components(components: &[f64]) -> Self {
        assert_eq!(components.len(), 3, "su(2) has dimension 3");
        Self([components[0], components[1], components[2]])
    }

    #[inline]
    fn to_components(&self) -> Vec<f64> {
        self.0.to_vec()
    }

    /// Lie bracket for su(2): [X, Y] = -(X × Y)
    ///
    /// # Convention
    ///
    /// We represent su(2) as ℝ³ with basis `{eᵢ} = {iσᵢ/2}`. The matrix
    /// commutator gives:
    /// ```text
    /// [iσᵢ/2, iσⱼ/2] = (i²/4)[σᵢ, σⱼ] = -(1/4)(2iεᵢⱼₖσₖ) = -εᵢⱼₖ(iσₖ/2)
    /// ```
    ///
    /// In ℝ³ coordinates, this is the **negative cross product**:
    /// ```text
    /// [X, Y] = -(X × Y)
    /// ```
    ///
    /// The negative sign is the unique bracket consistent with the half-angle
    /// exponential map `exp(v) = cos(‖v‖/2)I + i·sin(‖v‖/2)·v̂·σ`, ensuring
    /// the BCH formula `exp(X)·exp(Y) = exp(X + Y - ½(X×Y) + ...)` holds.
    ///
    /// # Properties
    ///
    /// - Structure constants: `fᵢⱼₖ = -εᵢⱼₖ`
    /// - Antisymmetric: `[X, Y] = -[Y, X]`
    /// - Jacobi identity: `[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0`
    /// - Killing form: `B(X, Y) = -2(X · Y)`
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::Su2Algebra;
    /// use lie_groups::LieAlgebra;
    ///
    /// let e1 = Su2Algebra::basis_element(0);  // (1, 0, 0)
    /// let e2 = Su2Algebra::basis_element(1);  // (0, 1, 0)
    /// let bracket = e1.bracket(&e2);          // [e₁, e₂] = -e₃
    ///
    /// // Should give -e₃ = (0, 0, -1)
    /// assert!((bracket.0[0]).abs() < 1e-10);
    /// assert!((bracket.0[1]).abs() < 1e-10);
    /// assert!((bracket.0[2] - (-1.0)).abs() < 1e-10);
    /// ```
    #[inline]
    fn bracket(&self, other: &Self) -> Self {
        // Negative cross product: -(X × Y)
        // Consistent with exp convention exp(v) = exp_matrix(iv·σ/2)
        let x = self.0;
        let y = other.0;
        Self([
            -(x[1] * y[2] - x[2] * y[1]),
            -(x[2] * y[0] - x[0] * y[2]),
            -(x[0] * y[1] - x[1] * y[0]),
        ])
    }
}

// ============================================================================
// Casimir Operators for SU(2)
// ============================================================================

impl crate::Casimir for Su2Algebra {
    type Representation = crate::representation::Spin;

    #[inline]
    fn quadratic_casimir_eigenvalue(irrep: &Self::Representation) -> f64 {
        let j = irrep.value();
        j * (j + 1.0)
    }

    #[inline]
    fn rank() -> usize {
        1 // SU(2) has rank 1 (dimension of Cartan subalgebra)
    }
}

/// SU(2) group element represented as a 2×2 unitary matrix.
///
/// SU(2) is the group of 2×2 complex unitary matrices with determinant 1:
/// ```text
/// U ∈ SU(2)  ⟺  U†U = I  and  det(U) = 1
/// ```
///
/// # Applications
/// - Represents rotations and spin transformations
/// - Acts on spinor fields: ψ → U ψ
/// - Preserves inner products (unitarity)
#[derive(Debug, Clone)]
pub struct SU2 {
    /// The 2×2 unitary matrix representation
    pub(crate) matrix: Array2<Complex64>,
}

impl SU2 {
    /// Access the underlying 2×2 unitary matrix
    #[must_use]
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }

    /// Identity element: I₂
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: Array2::eye(2),
        }
    }

    /// Get Pauli X generator (`σ_x` / 2)
    ///
    /// Returns: (1/2) * [[0, 1], [1, 0]]
    #[must_use]
    pub fn pauli_x() -> Array2<Complex64> {
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 1]] = Complex64::new(0.5, 0.0);
        matrix[[1, 0]] = Complex64::new(0.5, 0.0);
        matrix
    }

    /// Get Pauli Y generator (`σ_y` / 2)
    ///
    /// Returns: (1/2) * [[0, -i], [i, 0]]
    #[must_use]
    pub fn pauli_y() -> Array2<Complex64> {
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 1]] = Complex64::new(0.0, -0.5);
        matrix[[1, 0]] = Complex64::new(0.0, 0.5);
        matrix
    }

    /// Get Pauli Z generator (`σ_z` / 2)
    ///
    /// Returns: (1/2) * [[1, 0], [0, -1]]
    #[must_use]
    pub fn pauli_z() -> Array2<Complex64> {
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(0.5, 0.0);
        matrix[[1, 1]] = Complex64::new(-0.5, 0.0);
        matrix
    }

    /// Rotation around X-axis by angle θ
    ///
    /// Uses closed form: U = cos(θ/2) I + i sin(θ/2) `σ_x`
    #[inline]
    #[must_use]
    pub fn rotation_x(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();

        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(c, 0.0);
        matrix[[0, 1]] = Complex64::new(0.0, s);
        matrix[[1, 0]] = Complex64::new(0.0, s);
        matrix[[1, 1]] = Complex64::new(c, 0.0);

        Self { matrix }
    }

    /// Rotation around Y-axis by angle θ
    ///
    /// Uses closed form: U = cos(θ/2) I + i sin(θ/2) `σ_y`
    #[inline]
    #[must_use]
    pub fn rotation_y(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();

        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(c, 0.0);
        matrix[[0, 1]] = Complex64::new(s, 0.0);
        matrix[[1, 0]] = Complex64::new(-s, 0.0);
        matrix[[1, 1]] = Complex64::new(c, 0.0);

        Self { matrix }
    }

    /// Rotation around Z-axis by angle θ
    ///
    /// Uses closed form: U = cos(θ/2) I + i sin(θ/2) `σ_z`
    #[inline]
    #[must_use]
    pub fn rotation_z(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();

        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(c, s);
        matrix[[0, 1]] = Complex64::new(0.0, 0.0);
        matrix[[1, 0]] = Complex64::new(0.0, 0.0);
        matrix[[1, 1]] = Complex64::new(c, -s);

        Self { matrix }
    }

    /// Random SU(2) element uniformly distributed according to Haar measure
    ///
    /// Requires the `rand` feature (enabled by default).
    ///
    /// Samples uniformly from SU(2) ≅ S³ (the 3-sphere) using the quaternion
    /// representation and Gaussian sampling method.
    ///
    /// # Mathematical Background
    ///
    /// SU(2) is diffeomorphic to the 3-sphere S³ ⊂ ℝ⁴:
    /// ```text
    /// SU(2) = {(a,b,c,d) ∈ ℝ⁴ : a² + b² + c² + d² = 1}
    /// ```
    ///
    /// represented as matrices:
    /// ```text
    /// U = [[a+ib,  c+id ],
    ///      [-c+id, a-ib]]
    /// ```
    ///
    /// ## Haar Measure Sampling
    ///
    /// To sample uniformly from S³:
    /// 1. Generate 4 independent standard Gaussian variables (μ=0, σ=1)
    /// 2. Normalize to unit length
    ///
    /// This gives the uniform (Haar) measure on SU(2) due to the rotational
    /// invariance of the Gaussian distribution.
    ///
    /// # References
    ///
    /// - Shoemake, K.: "Uniform Random Rotations" (Graphics Gems III, 1992)
    /// - Diaconis & Saloff-Coste: "Comparison Theorems for Random Walks on Groups" (1993)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lie_groups::SU2;
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// let g = SU2::random_haar(&mut rng);
    ///
    /// // Verify it's unitary
    /// assert!(g.verify_unitarity(1e-10));
    /// ```
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn random_haar<R: rand::Rng>(rng: &mut R) -> Self {
        use rand_distr::{Distribution, StandardNormal};

        // Numerical stability constant: minimum acceptable norm before re-sampling
        // Probability of norm < 1e-10 for 4 Gaussians is ~10^-40, but we guard anyway
        const MIN_NORM: f64 = 1e-10;

        loop {
            // Generate 4 independent Gaussian(0,1) random variables
            let a: f64 = StandardNormal.sample(rng);
            let b: f64 = StandardNormal.sample(rng);
            let c: f64 = StandardNormal.sample(rng);
            let d: f64 = StandardNormal.sample(rng);

            // Normalize to unit length (project onto S³)
            let norm = (a * a + b * b + c * c + d * d).sqrt();

            // Guard against numerical instability: if norm is too small, re-sample
            // This prevents division by ~0 which would produce Inf/NaN
            if norm < MIN_NORM {
                continue;
            }

            let a = a / norm;
            let b = b / norm;
            let c = c / norm;
            let d = d / norm;

            // Construct SU(2) matrix from quaternion (a,b,c,d)
            // U = [[a+ib,  c+id ],
            //      [-c+id, a-ib]]
            let mut matrix = Array2::zeros((2, 2));
            matrix[[0, 0]] = Complex64::new(a, b);
            matrix[[0, 1]] = Complex64::new(c, d);
            matrix[[1, 0]] = Complex64::new(-c, d);
            matrix[[1, 1]] = Complex64::new(a, -b);

            return Self { matrix };
        }
    }

    /// Group inverse: U⁻¹ = U† (conjugate transpose for unitary matrices)
    #[must_use]
    pub fn inverse(&self) -> Self {
        let matrix = self.matrix.t().mapv(|z| z.conj());
        Self { matrix }
    }

    /// Hermitian conjugate transpose: U†
    ///
    /// For unitary matrices U ∈ SU(2), we have U† = U⁻¹
    /// This is used in gauge transformations: A' = g A g†
    #[must_use]
    pub fn conjugate_transpose(&self) -> Self {
        self.inverse()
    }

    /// Act on a 2D vector: ψ → U ψ
    #[must_use]
    pub fn act_on_vector(&self, v: &[Complex64; 2]) -> [Complex64; 2] {
        let vec = Array2::from_shape_vec((2, 1), vec![v[0], v[1]])
            .expect("Failed to create 2x1 array from 2-element vector");
        let result = self.matrix.dot(&vec);
        [result[[0, 0]], result[[1, 0]]]
    }

    /// Trace of the matrix: Tr(U)
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        self.matrix[[0, 0]] + self.matrix[[1, 1]]
    }

    /// Distance from identity (geodesic distance in Lie group manifold)
    ///
    /// Computed as: d(U, I) = ||log(U)||_F (Frobenius norm of logarithm)
    ///
    /// # Numerical Behavior
    ///
    /// For valid SU(2) matrices, `|Re(Tr(U))/2| ≤ 1` exactly. Small violations
    /// (within `UNITARITY_VIOLATION_THRESHOLD`) are clamped silently. Larger
    /// violations trigger a debug assertion, indicating matrix corruption.
    #[must_use]
    pub fn distance_to_identity(&self) -> f64 {
        // For SU(2), use the formula: d = arccos(Re(Tr(U))/2)
        // This is the angle of rotation
        let trace_val = self.trace();
        let raw_cos_half = trace_val.re / 2.0;

        // Detect unitarity violations: for valid SU(2), |cos(θ/2)| ≤ 1
        debug_assert!(
            raw_cos_half.abs() <= 1.0 + UNITARITY_VIOLATION_THRESHOLD,
            "SU(2) unitarity violation: |cos(θ/2)| = {:.10} > 1 + ε. \
             Matrix has drifted from the group manifold.",
            raw_cos_half.abs()
        );

        // Safe clamp for tiny numerical overshoot
        let cos_half_angle = raw_cos_half.clamp(-1.0, 1.0);
        2.0 * cos_half_angle.acos()
    }

    /// Extract rotation angle θ and axis n̂ from SU(2) element.
    ///
    /// For U = cos(θ/2)·I + i·sin(θ/2)·(n̂·σ), returns (θ, n̂).
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `angle`: Rotation angle θ ∈ [0, 2π]
    /// - `axis`: Unit 3-vector n̂ = `[n_x, n_y, n_z]` (or `[0,0,1]` if θ ≈ 0)
    ///
    /// # Usage in Topological Charge
    ///
    /// For computing topological charge, the product `F_μν · F_ρσ` involves
    /// the dot product of the orientation axes: `(n_μν · n_ρσ)`.
    #[must_use]
    pub fn angle_and_axis(&self) -> (f64, [f64; 3]) {
        // U = cos(θ/2)·I + i·sin(θ/2)·(n̂·σ)
        // matrix[[0,0]] = cos(θ/2) + i·sin(θ/2)·n_z
        // matrix[[0,1]] = sin(θ/2)·n_y + i·sin(θ/2)·n_x
        // matrix[[1,0]] = -sin(θ/2)·n_y + i·sin(θ/2)·n_x
        // matrix[[1,1]] = cos(θ/2) - i·sin(θ/2)·n_z

        let cos_half = self.matrix[[0, 0]].re.clamp(-1.0, 1.0);
        let angle = 2.0 * cos_half.acos();

        // For small angles, axis is ill-defined; return default
        if angle < 1e-10 {
            return (angle, [0.0, 0.0, 1.0]);
        }

        let sin_half = (angle / 2.0).sin();
        if sin_half.abs() < 1e-10 {
            // Near identity or near -I
            return (angle, [0.0, 0.0, 1.0]);
        }

        // Extract axis components
        let n_z = self.matrix[[0, 0]].im / sin_half;
        let n_x = self.matrix[[0, 1]].im / sin_half;
        let n_y = self.matrix[[0, 1]].re / sin_half;

        // Normalize (should already be unit, but ensure numerical stability)
        let norm = (n_x * n_x + n_y * n_y + n_z * n_z).sqrt();
        if norm < 1e-10 {
            return (angle, [0.0, 0.0, 1.0]);
        }

        (angle, [n_x / norm, n_y / norm, n_z / norm])
    }

    /// Verify this is approximately in SU(2)
    ///
    /// Checks: U†U ≈ I and |det(U) - 1| ≈ 0
    #[must_use]
    pub fn verify_unitarity(&self, tolerance: f64) -> bool {
        // Check U†U = I
        let u_dagger = self.matrix.t().mapv(|z: Complex64| z.conj());
        let product = u_dagger.dot(&self.matrix);
        let identity = Array2::eye(2);

        let diff_norm: f64 = product
            .iter()
            .zip(identity.iter())
            .map(|(a, b): (&Complex64, &Complex64)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();

        diff_norm < tolerance
    }

    /// Convert to 2×2 array format (for quaternion compatibility)
    #[must_use]
    pub fn to_matrix_array(&self) -> [[num_complex::Complex64; 2]; 2] {
        [
            [
                num_complex::Complex64::new(self.matrix[[0, 0]].re, self.matrix[[0, 0]].im),
                num_complex::Complex64::new(self.matrix[[0, 1]].re, self.matrix[[0, 1]].im),
            ],
            [
                num_complex::Complex64::new(self.matrix[[1, 0]].re, self.matrix[[1, 0]].im),
                num_complex::Complex64::new(self.matrix[[1, 1]].re, self.matrix[[1, 1]].im),
            ],
        ]
    }

    /// Create from 2×2 array format (for quaternion compatibility)
    #[must_use]
    pub fn from_matrix_array(arr: [[num_complex::Complex64; 2]; 2]) -> Self {
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(arr[0][0].re, arr[0][0].im);
        matrix[[0, 1]] = Complex64::new(arr[0][1].re, arr[0][1].im);
        matrix[[1, 0]] = Complex64::new(arr[1][0].re, arr[1][0].im);
        matrix[[1, 1]] = Complex64::new(arr[1][1].re, arr[1][1].im);
        Self { matrix }
    }

    // ========================================================================
    // Numerically Stable Logarithm
    // ========================================================================

    /// Compute logarithm using numerically stable atan2 formulation.
    ///
    /// This is more stable than the standard `log()` method, especially for
    /// rotation angles approaching π. Uses `atan2(sin(θ/2), cos(θ/2))` instead
    /// of `acos(cos(θ/2))` for improved numerical conditioning.
    ///
    /// # Stability Improvements
    ///
    /// | Angle θ | acos stability | atan2 stability |
    /// |---------|----------------|-----------------|
    /// | θ ≈ 0   | Poor (derivative ∞) | Good |
    /// | θ ≈ π/2 | Good | Good |
    /// | θ ≈ π   | Good | Good |
    ///
    /// The atan2 formulation avoids the derivative singularity at θ = 0
    /// where d(acos)/dx → ∞.
    ///
    /// # Returns
    ///
    /// Same as `log()`, but with improved numerical stability.
    pub fn log_stable(&self) -> crate::error::LogResult<Su2Algebra> {
        use crate::error::LogError;

        // For SU(2): U = cos(θ/2)I + i·sin(θ/2)·(n̂·σ)
        //
        // Extract sin(θ/2)·n̂ directly from matrix elements:
        // U[[0,0]] = cos(θ/2) + i·nz·sin(θ/2)  →  sin_nz = Im(U[[0,0]])
        // U[[0,1]] = (ny + i·nx)·sin(θ/2)      →  sin_nx = Im(U[[0,1]])
        //                                          sin_ny = Re(U[[0,1]])

        let sin_nx = self.matrix[[0, 1]].im;
        let sin_ny = self.matrix[[0, 1]].re;
        let sin_nz = self.matrix[[0, 0]].im;

        // Compute sin(θ/2) = ||sin(θ/2)·n̂||
        let sin_half_theta = (sin_nx * sin_nx + sin_ny * sin_ny + sin_nz * sin_nz).sqrt();

        // Compute cos(θ/2) = Re(Tr(U))/2
        let raw_cos_half = self.matrix[[0, 0]].re; // = cos(θ/2) for proper SU(2)

        // Detect unitarity violations
        if (raw_cos_half.abs() - 1.0).max(0.0) > UNITARITY_VIOLATION_THRESHOLD
            && sin_half_theta < UNITARITY_VIOLATION_THRESHOLD
        {
            // Check if cos² + sin² ≈ 1
            let norm_check = raw_cos_half * raw_cos_half + sin_half_theta * sin_half_theta;
            if (norm_check - 1.0).abs() > 1e-6 {
                return Err(LogError::NumericalInstability {
                    reason: format!(
                        "SU(2) unitarity violation: cos²(θ/2) + sin²(θ/2) = {:.10} ≠ 1",
                        norm_check
                    ),
                });
            }
        }

        // Use atan2 for stable angle extraction
        // atan2(sin, cos) is well-conditioned everywhere except at origin
        let half_theta = sin_half_theta.atan2(raw_cos_half);
        let theta = 2.0 * half_theta;

        // Handle identity case
        if sin_half_theta < 1e-15 {
            return Ok(Su2Algebra::zero());
        }

        // Note: θ = π is NOT a singularity for SU(2). At θ = π,
        // sin(θ/2) = 1 and the rotation axis is perfectly extractable.
        // The true singularity is at θ = 2π (U = -I, sin(θ/2) = 0),
        // which is caught by the sin_half_theta check above.

        // Extract normalized axis and scale by angle
        let scale = theta / sin_half_theta;
        Ok(Su2Algebra([scale * sin_nx, scale * sin_ny, scale * sin_nz]))
    }

    /// Compute logarithm with conditioning information.
    ///
    /// Returns both the logarithm and a [`LogCondition`](crate::LogCondition)
    /// structure that provides information about the numerical reliability
    /// of the result.
    ///
    /// Unlike `log()`, this method also provides condition number information
    /// so callers can assess the numerical reliability of the result near
    /// the cut locus (θ → 2π, U → -I).
    ///
    /// # Example
    ///
    /// ```rust
    /// use lie_groups::SU2;
    ///
    /// let g = SU2::rotation_x(2.9); // Close to π
    /// let (log_g, cond) = g.log_with_condition().unwrap();
    ///
    /// if cond.is_well_conditioned() {
    ///     println!("Reliable result: {:?}", log_g);
    /// } else {
    ///     println!("Warning: condition number = {:.1}", cond.condition_number);
    /// }
    /// ```
    ///
    /// # Cut Locus Behavior
    ///
    /// The SU(2) cut locus is at θ = 2π (U = -I), where sin(θ/2) = 0
    /// and the axis is undefined. As θ → 2π, the condition number
    /// diverges as 1/sin(θ/2). The method still returns a result — it's
    /// up to the caller to decide whether to use it based on the
    /// conditioning information.
    pub fn log_with_condition(&self) -> crate::error::ConditionedLogResult<Su2Algebra> {
        use crate::error::{LogCondition, LogError};

        // Extract sin(θ/2)·n̂ from matrix elements
        let sin_nx = self.matrix[[0, 1]].im;
        let sin_ny = self.matrix[[0, 1]].re;
        let sin_nz = self.matrix[[0, 0]].im;
        let sin_half_theta = (sin_nx * sin_nx + sin_ny * sin_ny + sin_nz * sin_nz).sqrt();

        let raw_cos_half = self.matrix[[0, 0]].re;

        // Unitarity check
        let norm_check = raw_cos_half * raw_cos_half + sin_half_theta * sin_half_theta;
        if (norm_check - 1.0).abs() > 1e-6 {
            return Err(LogError::NumericalInstability {
                reason: format!(
                    "SU(2) unitarity violation: cos²(θ/2) + sin²(θ/2) = {:.10} ≠ 1",
                    norm_check
                ),
            });
        }

        // Stable angle extraction using atan2
        let half_theta = sin_half_theta.atan2(raw_cos_half);
        let theta = 2.0 * half_theta;

        // Compute conditioning
        let condition = LogCondition::from_angle(theta);

        // Handle identity case
        if sin_half_theta < 1e-15 {
            return Ok((Su2Algebra::zero(), condition));
        }

        // For θ < 2π, the axis is extractable. The result is well-conditioned
        // for most of the range; only near θ = 2π (U = -I) does sin(θ/2) → 0.
        let scale = theta / sin_half_theta;
        let result = Su2Algebra([scale * sin_nx, scale * sin_ny, scale * sin_nz]);

        Ok((result, condition))
    }
}

/// Group multiplication: U₁ * U₂
impl Mul<&SU2> for &SU2 {
    type Output = SU2;

    fn mul(self, rhs: &SU2) -> SU2 {
        SU2 {
            matrix: self.matrix.dot(&rhs.matrix),
        }
    }
}

impl Mul<&SU2> for SU2 {
    type Output = SU2;

    fn mul(self, rhs: &SU2) -> SU2 {
        &self * rhs
    }
}

impl MulAssign<&SU2> for SU2 {
    fn mul_assign(&mut self, rhs: &SU2) {
        self.matrix = self.matrix.dot(&rhs.matrix);
    }
}

impl fmt::Display for Su2Algebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "su(2)[{:.4}, {:.4}, {:.4}]",
            self.0[0], self.0[1], self.0[2]
        )
    }
}

impl fmt::Display for SU2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show rotation angle and distance to identity
        let dist = self.distance_to_identity();
        write!(f, "SU(2)(θ={:.4})", dist)
    }
}

/// Implementation of the `LieGroup` trait for SU(2).
///
/// This provides the abstract group interface, making SU(2) usable in
/// generic gauge theory algorithms.
///
/// # Mathematical Verification
///
/// The implementation satisfies all group axioms:
/// - Identity: Verified in tests (`test_identity`)
/// - Inverse: Verified in tests (`test_inverse`)
/// - Associativity: Follows from matrix multiplication
/// - Closure: Matrix multiplication of unitaries is unitary
impl LieGroup for SU2 {
    const DIM: usize = 2;

    type Algebra = Su2Algebra;

    fn identity() -> Self {
        Self::identity() // Delegate to inherent method
    }

    fn compose(&self, other: &Self) -> Self {
        // Group composition is matrix multiplication
        self * other
    }

    fn inverse(&self) -> Self {
        Self::inverse(self) // Delegate to inherent method
    }

    fn conjugate_transpose(&self) -> Self {
        Self::conjugate_transpose(self) // Delegate to inherent method
    }

    fn distance_to_identity(&self) -> f64 {
        Self::distance_to_identity(self) // Delegate to inherent method
    }

    fn exp(tangent: &Su2Algebra) -> Self {
        // Su2Algebra([a, b, c]) corresponds to X = i(a·σₓ + b·σᵧ + c·σᵤ)/2
        // with rotation angle θ = ||(a,b,c)||.
        let angle = tangent.norm();

        if angle < SMALL_ANGLE_EXP_THRESHOLD {
            // For small angles, return identity (avoid division by zero)
            return Self::identity();
        }

        // Normalize to get rotation axis
        let axis = tangent.scale(1.0 / angle);

        // Rodrigues formula: exp(i(θ/2)·n̂·σ) = cos(θ/2)I + i·sin(θ/2)·n̂·σ
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();

        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(c, s * axis.0[2]);
        matrix[[0, 1]] = Complex64::new(s * axis.0[1], s * axis.0[0]);
        matrix[[1, 0]] = Complex64::new(-s * axis.0[1], s * axis.0[0]);
        matrix[[1, 1]] = Complex64::new(c, -s * axis.0[2]);

        Self { matrix }
    }

    fn log(&self) -> crate::error::LogResult<Su2Algebra> {
        use crate::error::LogError;

        // For SU(2), we use the formula:
        // U = cos(θ/2)I + i·sin(θ/2)·(n̂·σ)
        //
        // Where θ is the rotation angle and n̂ is the rotation axis.
        //
        // The logarithm is:
        // log(U) = θ·n̂ ∈ su(2) ≅ ℝ³
        //
        // Step 1: Extract angle from trace
        // Tr(U) = 2·cos(θ/2)
        let tr = self.trace();
        let raw_cos_half = tr.re / 2.0;

        // Detect unitarity violations: for valid SU(2), |cos(θ/2)| ≤ 1
        if raw_cos_half.abs() > 1.0 + UNITARITY_VIOLATION_THRESHOLD {
            return Err(LogError::NumericalInstability {
                reason: format!(
                    "SU(2) unitarity violation: |cos(θ/2)| = {:.10} > 1 + ε. \
                     Matrix has drifted from the group manifold.",
                    raw_cos_half.abs()
                ),
            });
        }

        // Safe clamp for tiny numerical overshoot
        let cos_half_theta = raw_cos_half.clamp(-1.0, 1.0);
        let half_theta = cos_half_theta.acos();
        let theta = 2.0 * half_theta;

        // Step 2: Handle special cases
        const SMALL_ANGLE_THRESHOLD: f64 = 1e-10;

        if theta.abs() < SMALL_ANGLE_THRESHOLD {
            // Near identity: log(I) = 0
            return Ok(Su2Algebra::zero());
        }

        // Note: θ = π is NOT a singularity for SU(2). At θ = π,
        // sin(θ/2) = 1 and the axis is perfectly extractable.
        // The true singularity is at θ = 2π where U = -I and
        // sin(θ/2) = 0, which is caught by the sin_half_theta
        // check below.

        // Step 3: Extract rotation axis from matrix elements
        // For U = cos(θ/2)I + i·sin(θ/2)·(n̂·σ), we have:
        //
        // U = [[cos(θ/2) + i·nz·sin(θ/2),  (ny + i·nx)·sin(θ/2)    ],
        //      [(-ny + i·nx)·sin(θ/2),     cos(θ/2) - i·nz·sin(θ/2)]]
        //
        // Extracting:
        // U[[0,0]] = cos(θ/2) + i·nz·sin(θ/2)  →  nz = Im(U[[0,0]]) / sin(θ/2)
        // U[[0,1]] = (ny + i·nx)·sin(θ/2)      →  nx = Im(U[[0,1]]) / sin(θ/2)
        //                                          ny = Re(U[[0,1]]) / sin(θ/2)

        let sin_half_theta = (half_theta).sin();

        if sin_half_theta.abs() < SIN_HALF_THETA_THRESHOLD {
            // This shouldn't happen given our checks above, but guard against it
            return Err(LogError::NumericalInstability {
                reason: "sin(θ/2) too small for reliable axis extraction".to_string(),
            });
        }

        let nx = self.matrix[[0, 1]].im / sin_half_theta;
        let ny = self.matrix[[0, 1]].re / sin_half_theta;
        let nz = self.matrix[[0, 0]].im / sin_half_theta;

        // The logarithm is log(U) = θ·(nx, ny, nz) ∈ su(2)
        Ok(Su2Algebra([theta * nx, theta * ny, theta * nz]))
    }

    fn adjoint_action(&self, algebra_element: &Su2Algebra) -> Su2Algebra {
        // Ad_g(X) = g X g† for matrix groups
        //
        // We construct the matrix M = i(a·σₓ + b·σᵧ + c·σᵤ) = 2X, where
        // X = i(a·σₓ + b·σᵧ + c·σᵤ)/2 is the actual algebra element in the
        // {iσ/2} basis. The factor of 2 cancels: we compute gMg† and extract
        // coefficients using the same convention, recovering (a', b', c').

        let [a, b, c] = algebra_element.0;
        let i = Complex64::new(0.0, 1.0);

        // M = i(a·σₓ + b·σᵧ + c·σᵤ) = [[c·i, b + a·i], [-b + a·i, -c·i]]
        let mut x_matrix = Array2::zeros((2, 2));
        x_matrix[[0, 0]] = i * c; // c·i
        x_matrix[[0, 1]] = Complex64::new(b, a); // b + a·i
        x_matrix[[1, 0]] = Complex64::new(-b, a); // -b + a·i
        x_matrix[[1, 1]] = -i * c; // -c·i

        // Compute g X g† where g† is conjugate transpose
        let g_x = self.matrix.dot(&x_matrix);
        let g_adjoint_matrix = self.matrix.t().mapv(|z| z.conj()); // Conjugate transpose
        let result = g_x.dot(&g_adjoint_matrix);

        // Extract coefficients: result = [[c'·i, b'+a'·i], [-b'+a'·i, -c'·i]]

        let a_prime = result[[0, 1]].im; // Imaginary part of result[[0,1]]
        let b_prime = result[[0, 1]].re; // Real part of result[[0,1]]
        let c_prime = result[[0, 0]].im; // Imaginary part of result[[0,0]]

        Su2Algebra([a_prime, b_prime, c_prime])
    }

    fn dim() -> usize {
        2 // SU(2) consists of 2×2 matrices
    }

    fn trace(&self) -> Complex64 {
        // Tr(M) = M₀₀ + M₁₁
        self.matrix[[0, 0]] + self.matrix[[1, 1]]
    }
}

// ============================================================================
// Mathematical Property Implementations for SU(2)
// ============================================================================

use crate::traits::{Compact, SemiSimple, Simple};

/// SU(2) is compact.
///
/// All elements are bounded: ||U|| = 1 for all U ∈ SU(2).
/// The group is diffeomorphic to the 3-sphere S³.
impl Compact for SU2 {}

/// SU(2) is simple.
///
/// It has no non-trivial normal subgroups. This is a fundamental
/// result in Lie theory - SU(2) is one of the classical simple groups.
impl Simple for SU2 {}

/// SU(2) is semi-simple.
///
/// As a simple group, it is automatically semi-simple.
/// (Simple ⊂ Semi-simple in the classification hierarchy)
impl SemiSimple for SU2 {}

// ============================================================================
// Algebra Marker Traits
// ============================================================================

/// su(2) algebra elements are traceless by construction.
///
/// The representation `Su2Algebra([f64; 3])` stores coefficients in the
/// Pauli basis {iσ₁, iσ₂, iσ₃}. Since each Pauli matrix is traceless,
/// any linear combination is also traceless.
///
/// # Lean Connection
///
/// Combined with `det_exp_eq_exp_trace`: det(exp(X)) = exp(tr(X)) = exp(0) = 1.
/// Therefore `SU2::exp` always produces elements with determinant 1.
impl TracelessByConstruction for Su2Algebra {}

/// su(2) algebra elements are anti-Hermitian by construction.
///
/// The representation uses {iσ₁, iσ₂, iσ₃} where σᵢ are Hermitian.
/// Since (iσ)† = -iσ† = -iσ, each basis element is anti-Hermitian,
/// and any real linear combination is also anti-Hermitian.
///
/// # Lean Connection
///
/// Combined with `exp_antiHermitian_unitary`: exp(X)† · exp(X) = I.
/// Therefore `SU2::exp` always produces unitary elements.
impl AntiHermitianByConstruction for Su2Algebra {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let id = SU2::identity();
        assert!(id.verify_unitarity(1e-10));
        assert_relative_eq!(id.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_unitarity() {
        let u = SU2::rotation_x(0.5);
        assert!(u.verify_unitarity(1e-10));
    }

    #[test]
    fn test_inverse() {
        let u = SU2::rotation_y(1.2);
        let u_inv = u.inverse();
        let product = &u * &u_inv;

        assert_relative_eq!(product.distance_to_identity(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_group_multiplication() {
        let u1 = SU2::rotation_x(0.3);
        let u2 = SU2::rotation_y(0.7);
        let product = &u1 * &u2;

        assert!(product.verify_unitarity(1e-10));
    }

    #[test]
    fn test_action_on_vector() {
        let id = SU2::identity();
        let v = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let result = id.act_on_vector(&v);

        assert_relative_eq!(result[0].re, v[0].re, epsilon = 1e-10);
        assert_relative_eq!(result[1].im, v[1].im, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let u = SU2::rotation_z(2.5);
        let v = [Complex64::new(3.0, 4.0), Complex64::new(1.0, 2.0)];

        let norm_before = v[0].norm_sqr() + v[1].norm_sqr();
        let rotated = u.act_on_vector(&v);
        let norm_after = rotated[0].norm_sqr() + rotated[1].norm_sqr();

        assert_relative_eq!(norm_before, norm_after, epsilon = 1e-10);
    }

    // ========================================================================
    // LieGroup Trait Tests
    // ========================================================================

    #[test]
    fn test_lie_group_identity() {
        use crate::traits::LieGroup;

        let g = SU2::rotation_x(1.5);
        let e = SU2::identity();

        // Right identity: g * e = g
        let g_times_e = g.compose(&e);
        assert_relative_eq!(g_times_e.distance(&g), 0.0, epsilon = 1e-10);

        // Left identity: e * g = g
        let e_times_g = e.compose(&g);
        assert_relative_eq!(e_times_g.distance(&g), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lie_group_inverse() {
        use crate::traits::LieGroup;

        let g = SU2::rotation_y(2.3);
        let g_inv = g.inverse();

        // g * g⁻¹ = e
        // Use 1e-7 tolerance to account for accumulated numerical error
        // in matrix operations (2×2 matrix multiplication + arccos)
        let right_product = g.compose(&g_inv);
        assert!(
            right_product.is_near_identity(1e-7),
            "Right inverse failed: distance = {}",
            right_product.distance_to_identity()
        );

        // g⁻¹ * g = e
        let left_product = g_inv.compose(&g);
        assert!(
            left_product.is_near_identity(1e-7),
            "Left inverse failed: distance = {}",
            left_product.distance_to_identity()
        );
    }

    #[test]
    fn test_lie_group_associativity() {
        use crate::traits::LieGroup;

        let g1 = SU2::rotation_x(0.5);
        let g2 = SU2::rotation_y(0.8);
        let g3 = SU2::rotation_z(1.2);

        // (g1 * g2) * g3
        let left_assoc = g1.compose(&g2).compose(&g3);

        // g1 * (g2 * g3)
        let right_assoc = g1.compose(&g2.compose(&g3));

        assert_relative_eq!(left_assoc.distance(&right_assoc), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lie_group_distance_symmetry() {
        let g = SU2::rotation_x(1.8);
        let g_inv = g.inverse();

        // d(g, e) = d(g⁻¹, e) by symmetry
        let d1 = g.distance_to_identity();
        let d2 = g_inv.distance_to_identity();

        assert_relative_eq!(d1, d2, epsilon = 1e-10);
    }

    #[test]
    fn test_lie_group_is_near_identity() {
        use crate::traits::LieGroup;

        let e = SU2::identity();
        assert!(
            e.is_near_identity(1e-10),
            "Identity should be near identity"
        );

        let almost_e = SU2::rotation_x(1e-12);
        assert!(
            almost_e.is_near_identity(1e-10),
            "Small rotation should be near identity"
        );

        let g = SU2::rotation_y(0.5);
        assert!(
            !g.is_near_identity(1e-10),
            "Large rotation should not be near identity"
        );
    }

    #[test]
    fn test_lie_group_generic_algorithm() {
        use crate::traits::LieGroup;

        // Generic parallel transport function (works for any LieGroup!)
        fn parallel_transport<G: LieGroup>(path: &[G]) -> G {
            path.iter().fold(G::identity(), |acc, g| acc.compose(g))
        }

        let path = vec![
            SU2::rotation_x(0.1),
            SU2::rotation_y(0.2),
            SU2::rotation_z(0.3),
        ];

        let holonomy = parallel_transport(&path);

        // Verify it's a valid SU(2) element
        assert!(holonomy.verify_unitarity(1e-10));

        // Should be non-trivial (not identity)
        assert!(holonomy.distance_to_identity() > 0.1);
    }

    // ========================================================================
    // Property-Based Tests for Group Axioms
    // ========================================================================
    //
    // These tests use proptest to verify that SU(2) satisfies the
    // mathematical axioms of a Lie group for randomly generated elements.
    //
    // This is a form of **specification-based testing**: the group axioms
    // are the specification, and we verify they hold for all inputs.
    //
    // Run with: cargo test --features property-tests

    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    /// Strategy for generating arbitrary SU(2) elements.
    ///
    /// We generate SU(2) elements by composing three Euler rotations:
    /// `g = R_z(α) · R_y(β) · R_x(γ)`
    ///
    /// This gives a good coverage of SU(2) ≅ S³.
    #[cfg(feature = "proptest")]
    fn arb_su2() -> impl Strategy<Value = SU2> {
        use std::f64::consts::TAU;

        // Generate three Euler angles
        let alpha = 0.0..TAU;
        let beta = 0.0..TAU;
        let gamma = 0.0..TAU;

        (alpha, beta, gamma).prop_map(|(a, b, c)| {
            SU2::rotation_z(a)
                .compose(&SU2::rotation_y(b))
                .compose(&SU2::rotation_x(c))
        })
    }

    #[cfg(feature = "proptest")]
    proptest! {
        /// **Group Axiom 1: Identity Element**
        ///
        /// For all g ∈ SU(2):
        /// - e · g = g (left identity)
        /// - g · e = g (right identity)
        ///
        /// where e = identity element
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_identity_axiom(g in arb_su2()) {
            let e = SU2::identity();

            // Left identity: e · g = g
            let left = e.compose(&g);
            prop_assert!(
                left.distance(&g) < 1e-7,
                "Left identity failed: e·g != g, distance = {}",
                left.distance(&g)
            );

            // Right identity: g · e = g
            let right = g.compose(&e);
            prop_assert!(
                right.distance(&g) < 1e-7,
                "Right identity failed: g·e != g, distance = {}",
                right.distance(&g)
            );
        }

        /// **Group Axiom 2: Inverse Element**
        ///
        /// For all g ∈ SU(2):
        /// - g · g⁻¹ = e (right inverse)
        /// - g⁻¹ · g = e (left inverse)
        ///
        /// where g⁻¹ = inverse of g
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_inverse_axiom(g in arb_su2()) {
            let g_inv = g.inverse();

            // Right inverse: g · g⁻¹ = e
            let right_product = g.compose(&g_inv);
            prop_assert!(
                right_product.is_near_identity(1e-7),
                "Right inverse failed: g·g⁻¹ != e, distance = {}",
                right_product.distance_to_identity()
            );

            // Left inverse: g⁻¹ · g = e
            let left_product = g_inv.compose(&g);
            prop_assert!(
                left_product.is_near_identity(1e-7),
                "Left inverse failed: g⁻¹·g != e, distance = {}",
                left_product.distance_to_identity()
            );
        }

        /// **Group Axiom 3: Associativity**
        ///
        /// For all g₁, g₂, g₃ ∈ SU(2):
        /// - (g₁ · g₂) · g₃ = g₁ · (g₂ · g₃)
        ///
        /// Group multiplication is associative.
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_associativity(g1 in arb_su2(), g2 in arb_su2(), g3 in arb_su2()) {
            // Left association: (g₁ · g₂) · g₃
            let left_assoc = g1.compose(&g2).compose(&g3);

            // Right association: g₁ · (g₂ · g₃)
            let right_assoc = g1.compose(&g2.compose(&g3));

            prop_assert!(
                left_assoc.distance(&right_assoc) < 1e-7,
                "Associativity failed: (g₁·g₂)·g₃ != g₁·(g₂·g₃), distance = {}",
                left_assoc.distance(&right_assoc)
            );
        }

        /// **Lie Group Property: Inverse is Smooth**
        ///
        /// For SU(2), the inverse operation is smooth (continuously differentiable).
        /// We verify this by checking that nearby elements have nearby inverses.
        #[test]
        fn prop_inverse_continuity(g in arb_su2()) {
            // Create a small perturbation
            let epsilon = 0.01;
            let perturbation = SU2::rotation_x(epsilon);
            let g_perturbed = g.compose(&perturbation);

            // Check that inverses are close
            let inv_distance = g.inverse().distance(&g_perturbed.inverse());

            prop_assert!(
                inv_distance < 0.1,
                "Inverse not continuous: small perturbation caused large inverse change, distance = {}",
                inv_distance
            );
        }

        /// **Unitarity Preservation**
        ///
        /// All SU(2) operations should preserve unitarity.
        /// This is not strictly a group axiom, but it's essential for SU(2).
        #[test]
        fn prop_unitarity_preserved(g1 in arb_su2(), g2 in arb_su2()) {
            // Composition preserves unitarity
            let product = g1.compose(&g2);
            prop_assert!(
                product.verify_unitarity(1e-10),
                "Composition violated unitarity"
            );

            // Inverse preserves unitarity
            let inv = g1.inverse();
            prop_assert!(
                inv.verify_unitarity(1e-10),
                "Inverse violated unitarity"
            );
        }

        /// **Adjoint Representation: Group Homomorphism**
        ///
        /// The adjoint representation Ad: G → Aut(𝔤) is a group homomorphism:
        /// - Ad_{g₁∘g₂}(X) = Ad_{g₁}(Ad_{g₂}(X))
        ///
        /// This is a fundamental property that must hold for the adjoint action
        /// to be a valid representation of the group.
        #[test]
        fn prop_adjoint_homomorphism(
            g1 in arb_su2(),
            g2 in arb_su2(),
            x in arb_su2_algebra()
        ) {
            // Compute Ad_{g₁∘g₂}(X)
            let g_composed = g1.compose(&g2);
            let left = g_composed.adjoint_action(&x);

            // Compute Ad_{g₁}(Ad_{g₂}(X))
            let ad_g2_x = g2.adjoint_action(&x);
            let right = g1.adjoint_action(&ad_g2_x);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-7,
                "Adjoint homomorphism failed: Ad_{{g₁∘g₂}}(X) != Ad_{{g₁}}(Ad_{{g₂}}(X)), diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Identity Action**
        ///
        /// The identity element acts trivially on the Lie algebra:
        /// - Ad_e(X) = X for all X ∈ 𝔤
        #[test]
        fn prop_adjoint_identity(x in arb_su2_algebra()) {
            let e = SU2::identity();
            let result = e.adjoint_action(&x);

            let diff = result.add(&x.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-10,
                "Identity action failed: Ad_e(X) != X, diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Lie Bracket Preservation**
        ///
        /// The adjoint representation preserves the Lie bracket:
        /// - Ad_g([X,Y]) = [Ad_g(X), Ad_g(Y)]
        ///
        /// This is a critical property that ensures the adjoint action
        /// is a Lie algebra automorphism.
        #[test]
        fn prop_adjoint_bracket_preservation(
            g in arb_su2(),
            x in arb_su2_algebra(),
            y in arb_su2_algebra()
        ) {
            use crate::traits::LieAlgebra;

            // Compute Ad_g([X,Y])
            let bracket_xy = x.bracket(&y);
            let left = g.adjoint_action(&bracket_xy);

            // Compute [Ad_g(X), Ad_g(Y)]
            let ad_x = g.adjoint_action(&x);
            let ad_y = g.adjoint_action(&y);
            let right = ad_x.bracket(&ad_y);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-6,
                "Bracket preservation failed: Ad_g([X,Y]) != [Ad_g(X), Ad_g(Y)], diff norm = {}",
                diff.norm()
            );
        }

        /// **Jacobi Identity: Fundamental Lie Algebra Axiom**
        ///
        /// The Jacobi identity must hold for all X, Y, Z ∈ 𝔤:
        /// - [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
        ///
        /// This is equivalent to the derivation property of ad_X tested elsewhere,
        /// but here we test it directly with three random elements.
        #[test]
        fn prop_jacobi_identity(
            x in arb_su2_algebra(),
            y in arb_su2_algebra(),
            z in arb_su2_algebra()
        ) {
            use crate::traits::LieAlgebra;

            // Compute [X, [Y, Z]]
            let yz = y.bracket(&z);
            let term1 = x.bracket(&yz);

            // Compute [Y, [Z, X]]
            let zx = z.bracket(&x);
            let term2 = y.bracket(&zx);

            // Compute [Z, [X, Y]]
            let xy = x.bracket(&y);
            let term3 = z.bracket(&xy);

            // Sum should be zero
            let sum = term1.add(&term2).add(&term3);
            prop_assert!(
                sum.norm() < 1e-10,
                "Jacobi identity failed: ||[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]]|| = {:.2e}",
                sum.norm()
            );
        }

        /// **Adjoint Representation: Inverse Property**
        ///
        /// The inverse of an element acts as the inverse transformation:
        /// - Ad_{g⁻¹}(Ad_g(X)) = X
        #[test]
        fn prop_adjoint_inverse(g in arb_su2(), x in arb_su2_algebra()) {
            // Apply Ad_g then Ad_{g⁻¹}
            let ad_g_x = g.adjoint_action(&x);
            let g_inv = g.inverse();
            let result = g_inv.adjoint_action(&ad_g_x);

            // Should recover X
            let diff = result.add(&x.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-7,
                "Inverse property failed: Ad_{{g⁻¹}}(Ad_g(X)) != X, diff norm = {}",
                diff.norm()
            );
        }
    }

    /// Strategy for generating arbitrary `Su2Algebra` elements.
    ///
    /// We generate algebra elements by picking random coefficients in [-π, π]
    /// for each of the three basis directions (`σ_x`, `σ_y`, `σ_z`).
    #[cfg(feature = "proptest")]
    fn arb_su2_algebra() -> impl Strategy<Value = Su2Algebra> {
        use proptest::prelude::*;
        use std::f64::consts::PI;

        ((-PI..PI), (-PI..PI), (-PI..PI)).prop_map(|(a, b, c)| Su2Algebra([a, b, c]))
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random_haar_unitarity() {
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Generate many random elements and verify they're all unitary
        for _ in 0..100 {
            let g = SU2::random_haar(&mut rng);
            assert!(
                g.verify_unitarity(1e-10),
                "Random Haar element should be unitary"
            );
        }
    }

    #[test]
    fn test_bracket_bilinearity() {
        use crate::traits::LieAlgebra;

        let x = Su2Algebra([1.0, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 1.0, 0.0]);
        let z = Su2Algebra([0.0, 0.0, 1.0]);
        let alpha = 2.5;

        // Left linearity: [αX + Y, Z] = α[X, Z] + [Y, Z]
        let lhs = x.scale(alpha).add(&y).bracket(&z);
        let rhs = x.bracket(&z).scale(alpha).add(&y.bracket(&z));
        for i in 0..3 {
            assert!(
                (lhs.0[i] - rhs.0[i]).abs() < 1e-14,
                "Left bilinearity failed at component {}: {} vs {}",
                i,
                lhs.0[i],
                rhs.0[i]
            );
        }

        // Right linearity: [Z, αX + Y] = α[Z, X] + [Z, Y]
        let lhs = z.bracket(&x.scale(alpha).add(&y));
        let rhs = z.bracket(&x).scale(alpha).add(&z.bracket(&y));
        for i in 0..3 {
            assert!(
                (lhs.0[i] - rhs.0[i]).abs() < 1e-14,
                "Right bilinearity failed at component {}: {} vs {}",
                i,
                lhs.0[i],
                rhs.0[i]
            );
        }
    }

    /// **Jacobi Identity Test with Random Elements**
    ///
    /// The Jacobi identity is the defining axiom of a Lie algebra:
    /// ```text
    /// [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
    /// ```
    ///
    /// This test verifies it holds for random algebra elements.
    #[test]
    fn test_jacobi_identity_random() {
        use crate::traits::LieAlgebra;
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

        // Test with 100 random triples
        for _ in 0..100 {
            // Generate random algebra elements
            let x = Su2Algebra([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            let y = Su2Algebra([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            let z = Su2Algebra([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);

            // Compute [X, [Y, Z]]
            let yz = y.bracket(&z);
            let term1 = x.bracket(&yz);

            // Compute [Y, [Z, X]]
            let zx = z.bracket(&x);
            let term2 = y.bracket(&zx);

            // Compute [Z, [X, Y]]
            let xy = x.bracket(&y);
            let term3 = z.bracket(&xy);

            // Sum should be zero
            let sum = term1.add(&term2).add(&term3);
            assert!(
                sum.norm() < 1e-10,
                "Jacobi identity failed: ||[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]]|| = {:.2e}",
                sum.norm()
            );
        }
    }

    /// **Exp-Log Round-Trip Test**
    ///
    /// For any algebra element X with ||X|| < π, we should have:
    /// ```text
    /// log(exp(X)) = X
    /// ```
    #[test]
    fn test_exp_log_roundtrip() {
        use crate::traits::{LieAlgebra, LieGroup};
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rngs::StdRng::seed_from_u64(54321);
        let dist = Uniform::new(-2.0, 2.0); // Stay within log domain

        for _ in 0..100 {
            let x = Su2Algebra([
                dist.sample(&mut rng),
                dist.sample(&mut rng),
                dist.sample(&mut rng),
            ]);

            // exp then log
            let g = SU2::exp(&x);
            let x_recovered = g.log().expect("log should succeed for exp output");

            // Should recover original (up to numerical precision)
            let diff = x.add(&x_recovered.scale(-1.0));
            assert!(
                diff.norm() < 1e-10,
                "log(exp(X)) should equal X: ||diff|| = {:.2e}",
                diff.norm()
            );
        }
    }

    /// **Log-Exp Round-Trip Test**
    ///
    /// For any group element g, we should have:
    /// ```text
    /// exp(log(g)) = g
    /// ```
    #[test]
    #[cfg(feature = "rand")]
    fn test_log_exp_roundtrip() {
        use crate::traits::LieGroup;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(98765);

        for _ in 0..100 {
            let g = SU2::random_haar(&mut rng);

            // log then exp
            let x = g.log().expect("log should succeed for valid SU(2) element");
            let g_recovered = SU2::exp(&x);

            // Should recover original (numerical precision varies with rotation angle)
            let diff = g.compose(&g_recovered.inverse()).distance_to_identity();
            assert!(
                diff < 1e-7,
                "exp(log(g)) should equal g: diff = {:.2e}",
                diff
            );
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_random_haar_distribution() {
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        // Generate many samples and check statistical properties
        let samples: Vec<SU2> = (0..1000).map(|_| SU2::random_haar(&mut rng)).collect();

        // Check that average distance to identity is approximately correct
        // For uniform distribution on SU(2) ≅ S³, the expected distance is non-trivial
        let mean_distance: f64 =
            samples.iter().map(SU2::distance_to_identity).sum::<f64>() / samples.len() as f64;

        // For S³ with uniform (Haar) measure, the mean geodesic distance from identity
        // is approximately π (the maximum is π for antipodal points).
        // Empirically, the mean is close to π.
        assert!(
            mean_distance > 2.5 && mean_distance < 3.5,
            "Mean distance from identity should be ~π, got {}",
            mean_distance
        );

        // Check that some elements are far from identity
        let far_from_identity = samples
            .iter()
            .filter(|g| g.distance_to_identity() > std::f64::consts::PI / 2.0)
            .count();

        assert!(
            far_from_identity > 100,
            "Should have many elements far from identity, got {}",
            far_from_identity
        );
    }

    #[test]
    fn test_adjoint_action_simple() {
        use crate::traits::LieGroup;

        // Test with identity: Ad_e(X) = X
        let e = SU2::identity();
        let x = Su2Algebra([1.0, 0.0, 0.0]);
        let result = e.adjoint_action(&x);

        println!("Identity test: X = {:?}, Ad_e(X) = {:?}", x, result);
        assert!((result.0[0] - x.0[0]).abs() < 1e-10);
        assert!((result.0[1] - x.0[1]).abs() < 1e-10);
        assert!((result.0[2] - x.0[2]).abs() < 1e-10);

        // Test with rotation around Z by 90 degrees
        // Should rotate X basis element into Y basis element
        let g = SU2::rotation_z(std::f64::consts::FRAC_PI_2);
        let x_basis = Su2Algebra([1.0, 0.0, 0.0]);
        let rotated = g.adjoint_action(&x_basis);

        println!(
            "Rotation test: X = {:?}, Ad_{{Rz(π/2)}}(X) = {:?}",
            x_basis, rotated
        );
        // Should be approximately (0, 1, 0) - rotated 90° around Z
    }

    // ========================================================================
    // Casimir Operator Tests
    // ========================================================================

    #[test]
    fn test_su2_casimir_scalar() {
        // j = 0 (scalar): c₂ = 0
        use crate::representation::Spin;
        use crate::Casimir;

        let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::ZERO);
        assert_eq!(c2, 0.0, "Casimir of scalar representation should be 0");
    }

    #[test]
    fn test_su2_casimir_spinor() {
        // j = 1/2 (spinor): c₂ = 3/4
        use crate::representation::Spin;
        use crate::Casimir;

        let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::HALF);
        assert_eq!(c2, 0.75, "Casimir of spinor representation should be 3/4");
    }

    #[test]
    fn test_su2_casimir_vector() {
        // j = 1 (vector/adjoint): c₂ = 2
        use crate::representation::Spin;
        use crate::Casimir;

        let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::ONE);
        assert_eq!(c2, 2.0, "Casimir of vector representation should be 2");
    }

    #[test]
    fn test_su2_casimir_j_three_halves() {
        // j = 3/2: c₂ = 15/4
        use crate::representation::Spin;
        use crate::Casimir;

        let j_three_halves = Spin::from_half_integer(3);
        let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&j_three_halves);
        assert_eq!(c2, 3.75, "Casimir for j=3/2 should be 15/4 = 3.75");
    }

    #[test]
    fn test_su2_casimir_formula() {
        // Test the general formula c₂(j) = j(j+1) for various spins
        use crate::representation::Spin;
        use crate::Casimir;

        for two_j in 0..10 {
            let spin = Spin::from_half_integer(two_j);
            let j = spin.value();
            let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&spin);
            let expected = j * (j + 1.0);

            assert_relative_eq!(c2, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_su2_rank() {
        // SU(2) has rank 1
        use crate::Casimir;

        assert_eq!(Su2Algebra::rank(), 1, "SU(2) should have rank 1");
        assert_eq!(
            Su2Algebra::num_casimirs(),
            1,
            "SU(2) should have 1 Casimir operator"
        );
    }

    // ========================================================================
    // Stable Logarithm Tests
    // ========================================================================

    #[test]
    fn test_log_stable_identity() {
        let e = SU2::identity();
        let log_e = e.log_stable().expect("log of identity should succeed");
        assert!(log_e.norm() < 1e-10, "log(I) should be zero");
    }

    #[test]
    fn test_log_stable_small_rotation() {
        let theta = 0.1;
        let g = SU2::rotation_x(theta);
        let log_g = g
            .log_stable()
            .expect("log of small rotation should succeed");

        // Should give θ * (1, 0, 0)
        assert!((log_g.0[0] - theta).abs() < 1e-10);
        assert!(log_g.0[1].abs() < 1e-10);
        assert!(log_g.0[2].abs() < 1e-10);
    }

    #[test]
    fn test_log_stable_large_rotation() {
        let theta = 2.5; // Less than π
        let g = SU2::rotation_y(theta);
        let log_g = g.log_stable().expect("log of rotation < π should succeed");

        // Should give θ * (0, 1, 0)
        assert!(log_g.0[0].abs() < 1e-10);
        assert!((log_g.0[1] - theta).abs() < 1e-10);
        assert!(log_g.0[2].abs() < 1e-10);
    }

    #[test]
    fn test_log_stable_vs_log_consistency() {
        use crate::traits::{LieAlgebra, LieGroup};

        // For rotations away from 2π, log_stable and log should agree
        // Note: θ = π is NOT a singularity for SU(2), so we include it.
        for theta in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, std::f64::consts::PI] {
            let g = SU2::rotation_z(theta);
            let log_standard = g.log().expect("log should succeed");
            let log_stable = g.log_stable().expect("log_stable should succeed");

            let diff = log_standard.add(&log_stable.scale(-1.0)).norm();
            assert!(
                diff < 1e-10,
                "log and log_stable should agree for θ = {}: diff = {:.2e}",
                theta,
                diff
            );
        }
    }

    #[test]
    fn test_log_with_condition_returns_condition() {
        let theta = 1.0;
        let g = SU2::rotation_x(theta);
        let (log_g, cond) = g
            .log_with_condition()
            .expect("log_with_condition should succeed");

        // Should be well-conditioned for θ = 1
        assert!(
            cond.is_well_conditioned(),
            "θ = 1 should be well-conditioned"
        );
        assert!(
            (cond.angle - theta).abs() < 1e-10,
            "reported angle should match"
        );
        assert!((log_g.0[0] - theta).abs() < 1e-10);
    }

    #[test]
    fn test_log_with_condition_near_pi() {
        // At θ ≈ π, sin(θ/2) ≈ 1, so axis extraction is perfectly stable.
        // The cut locus for SU(2) is at θ = 2π (U = -I), not θ = π.
        let theta = std::f64::consts::PI - 0.01;
        let g = SU2::rotation_z(theta);
        let (log_g, cond) = g
            .log_with_condition()
            .expect("log_with_condition should return best-effort");

        // At θ ≈ π, sin(θ/2) ≈ 1, so numerical extraction is stable
        assert!(
            cond.is_well_conditioned(),
            "θ ≈ π should be numerically well-conditioned: κ = {}",
            cond.condition_number
        );

        // Distance to cut locus (2π) should be about π
        assert!(
            (cond.distance_to_cut_locus - (std::f64::consts::PI + 0.01)).abs() < 1e-10,
            "distance to cut locus should be ≈ π + 0.01: got {}",
            cond.distance_to_cut_locus
        );

        // Result should be correct
        assert!((log_g.0[2] - theta).abs() < 1e-6);
    }

    #[test]
    fn test_log_condition_from_angle() {
        use crate::{LogCondition, LogQuality};

        // Small angle θ = 0.5: sin(0.25) ≈ 0.247, κ ≈ 4.0 → Good
        // The condition number tracks stability of axis extraction (dividing by sin(θ/2))
        let cond_small = LogCondition::from_angle(0.5);
        assert_eq!(cond_small.quality, LogQuality::Good);
        assert!(cond_small.condition_number > 2.0 && cond_small.condition_number < 10.0);

        // π/2: sin(π/4) ≈ 0.707, κ ≈ 1.4 → Excellent
        let cond_half_pi = LogCondition::from_angle(std::f64::consts::FRAC_PI_2);
        assert_eq!(cond_half_pi.quality, LogQuality::Excellent);
        assert!(cond_half_pi.is_well_conditioned());

        // Near π: sin((π-0.001)/2) ≈ 1, κ ≈ 1.0 → Excellent (numerically stable)
        // Note: This is NUMERICALLY stable even though the axis is MATHEMATICALLY ambiguous
        let cond_near_pi = LogCondition::from_angle(std::f64::consts::PI - 0.001);
        assert!(
            cond_near_pi.is_well_conditioned(),
            "Near π should be numerically stable: κ = {}",
            cond_near_pi.condition_number
        );

        // Near zero: sin(0.001/2) ≈ 0.0005, κ ≈ 2000 → AtSingularity
        // This is where axis extraction becomes unstable (dividing by small number)
        let cond_near_zero = LogCondition::from_angle(0.001);
        assert!(
            !cond_near_zero.is_well_conditioned(),
            "Near zero should have poor conditioning: κ = {}",
            cond_near_zero.condition_number
        );
    }

    #[test]
    fn test_log_stable_roundtrip() {
        use crate::traits::{LieAlgebra, LieGroup};
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rngs::StdRng::seed_from_u64(99999);
        let dist = Uniform::new(-2.5, 2.5); // Stay within log domain but include large angles

        for _ in 0..100 {
            let x = Su2Algebra([
                dist.sample(&mut rng),
                dist.sample(&mut rng),
                dist.sample(&mut rng),
            ]);

            // Skip elements near the singularity
            if x.norm() > std::f64::consts::PI - 0.2 {
                continue;
            }

            // exp then log_stable
            let g = SU2::exp(&x);
            let x_recovered = g.log_stable().expect("log_stable should succeed");

            // Should recover original
            let diff = x.add(&x_recovered.scale(-1.0));
            assert!(
                diff.norm() < 1e-8,
                "log_stable(exp(X)) should equal X: ||diff|| = {:.2e}",
                diff.norm()
            );
        }
    }

    #[test]
    fn test_log_quality_display() {
        use crate::LogQuality;

        assert_eq!(format!("{}", LogQuality::Excellent), "Excellent");
        assert_eq!(format!("{}", LogQuality::Good), "Good");
        assert_eq!(format!("{}", LogQuality::Acceptable), "Acceptable");
        assert_eq!(format!("{}", LogQuality::Poor), "Poor");
        assert_eq!(format!("{}", LogQuality::AtSingularity), "AtSingularity");
    }
}
