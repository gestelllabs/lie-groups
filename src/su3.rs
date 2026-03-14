//! Lie group SU(3) - Special unitary 3×3 group
//!
//! SU(3) is the group of 3×3 complex unitary matrices with determinant 1.
//! It is the gauge group of quantum chromodynamics (QCD).
//!
//! # Mathematical Structure
//!
//! ```text
//! SU(3) = { U ∈ ℂ³ˣ³ | U† U = I, det(U) = 1 }
//! ```
//!
//! # Lie Algebra
//!
//! The Lie algebra su(3) consists of 3×3 traceless anti-Hermitian matrices:
//! ```text
//! su(3) = { X ∈ ℂ³ˣ³ | X† = -X, Tr(X) = 0 }
//! ```
//!
//! This is 8-dimensional, with basis given by the Gell-Mann matrices.
//!
//! # Gell-Mann Matrices
//!
//! The 8 generators λ₁, ..., λ₈ are:
//! ```text
//! λ₁ = [[0,1,0],[1,0,0],[0,0,0]]           (like Pauli σₓ in 1-2 block)
//! λ₂ = [[0,-i,0],[i,0,0],[0,0,0]]          (like Pauli σᵧ in 1-2 block)
//! λ₃ = [[1,0,0],[0,-1,0],[0,0,0]]          (like Pauli σᵤ in 1-2 block)
//! λ₄ = [[0,0,1],[0,0,0],[1,0,0]]           (1-3 mixing)
//! λ₅ = [[0,0,-i],[0,0,0],[i,0,0]]          (1-3 mixing)
//! λ₆ = [[0,0,0],[0,0,1],[0,1,0]]           (2-3 mixing)
//! λ₇ = [[0,0,0],[0,0,-i],[0,i,0]]          (2-3 mixing)
//! λ₈ = (1/√3)[[1,0,0],[0,1,0],[0,0,-2]]    (diagonal, traceless)
//! ```
//!
//! # Color Charge
//!
//! In QCD, SU(3) acts on the 3-dimensional color space (red, green, blue).
//! Quarks transform in the fundamental representation, gluons in the adjoint.
//!
//! # Scaling-and-Squaring Algorithm
//!
//! The matrix exponential uses scaling-and-squaring (Higham, "Functions of Matrices"):
//!
//! ```text
//! exp(X) = [exp(X/2^k)]^{2^k}
//! ```
//!
//! ## Threshold Choice: ||X/2^k|| ≤ 0.5
//!
//! We choose k such that the scaled matrix has Frobenius norm ≤ 0.5. This threshold
//! balances accuracy and efficiency:
//!
//! **Convergence Analysis:**
//! The Taylor series exp(Y) = I + Y + Y²/2! + ... has remainder bounded by:
//! ```text
//! ||exp(Y) - Σₙ Yⁿ/n!|| ≤ ||Y||^{N+1} / (N+1)! × exp(||Y||)
//! ```
//!
//! For ||Y|| ≤ 0.5 and N = 15 terms:
//! - Truncation error: 0.5^16 / 16! × e^0.5 ≈ 4 × 10^{-18}
//! - Well below IEEE 754 f64 machine epsilon (2.2 × 10^{-16})
//!
//! **Why not smaller?** Using ||Y|| ≤ 0.1 requires more squarings (k larger),
//! and each squaring accumulates rounding error. The choice 0.5 minimizes total error.
//!
//! **Why not larger?** For ||Y|| > 1, the series converges slowly and requires
//! many more terms, increasing both computation and accumulated rounding error.
//!
//! **Reference:** Al-Mohy & Higham, "A New Scaling and Squaring Algorithm for the
//! Matrix Exponential" (2010), SIAM J. Matrix Anal. Appl.

use crate::traits::{AntiHermitianByConstruction, LieAlgebra, LieGroup, TracelessByConstruction};
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

/// Lie algebra su(3) - 8-dimensional space of traceless anti-Hermitian 3×3 matrices
///
/// Elements are represented by 8 real coefficients [a₁, a₂, ..., a₈] corresponding
/// to the linear combination:
/// ```text
/// X = i·∑ⱼ aⱼ·λⱼ
/// ```
/// where λⱼ are the Gell-Mann matrices (j = 1..8).
///
/// # Basis Elements
///
/// The 8 Gell-Mann matrices form a basis for su(3). They satisfy:
/// - Hermitian: λⱼ† = λⱼ
/// - Traceless: Tr(λⱼ) = 0
/// - Normalized: Tr(λⱼ λₖ) = 2δⱼₖ
///
/// # Examples
///
/// ```
/// use lie_groups::su3::Su3Algebra;
/// use lie_groups::traits::LieAlgebra;
///
/// // First basis element (λ₁)
/// let e1 = Su3Algebra::basis_element(0);
/// assert_eq!(*e1.components(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Su3Algebra(pub(crate) [f64; 8]);

impl Su3Algebra {
    /// Create a new su(3) algebra element from components.
    ///
    /// The 8 components correspond to coefficients in the Gell-Mann basis:
    /// `X = i·∑ⱼ aⱼ·λⱼ`
    #[must_use]
    pub fn new(components: [f64; 8]) -> Self {
        Self(components)
    }

    /// Returns the components as a fixed-size array reference.
    #[must_use]
    pub fn components(&self) -> &[f64; 8] {
        &self.0
    }
}

impl Add for Su3Algebra {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut r = [0.0; 8];
        for i in 0..8 {
            r[i] = self.0[i] + rhs.0[i];
        }
        Self(r)
    }
}

impl Add<&Su3Algebra> for Su3Algebra {
    type Output = Su3Algebra;
    fn add(self, rhs: &Su3Algebra) -> Su3Algebra {
        self + *rhs
    }
}

impl Add<Su3Algebra> for &Su3Algebra {
    type Output = Su3Algebra;
    fn add(self, rhs: Su3Algebra) -> Su3Algebra {
        *self + rhs
    }
}

impl Add<&Su3Algebra> for &Su3Algebra {
    type Output = Su3Algebra;
    fn add(self, rhs: &Su3Algebra) -> Su3Algebra {
        *self + *rhs
    }
}

impl Sub for Su3Algebra {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut r = [0.0; 8];
        for i in 0..8 {
            r[i] = self.0[i] - rhs.0[i];
        }
        Self(r)
    }
}

impl Neg for Su3Algebra {
    type Output = Self;
    fn neg(self) -> Self {
        let mut r = [0.0; 8];
        for i in 0..8 {
            r[i] = -self.0[i];
        }
        Self(r)
    }
}

impl Mul<f64> for Su3Algebra {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        let mut r = [0.0; 8];
        for i in 0..8 {
            r[i] = self.0[i] * scalar;
        }
        Self(r)
    }
}

impl Mul<Su3Algebra> for f64 {
    type Output = Su3Algebra;
    fn mul(self, rhs: Su3Algebra) -> Su3Algebra {
        rhs * self
    }
}

impl LieAlgebra for Su3Algebra {
    const DIM: usize = 8;

    fn zero() -> Self {
        Self([0.0; 8])
    }

    fn add(&self, other: &Self) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.0[i] + other.0[i];
        }
        Self(result)
    }

    fn scale(&self, scalar: f64) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.0[i] * scalar;
        }
        Self(result)
    }

    fn norm(&self) -> f64 {
        self.0.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    }

    fn basis_element(i: usize) -> Self {
        assert!(i < 8, "SU(3) algebra is 8-dimensional");
        let mut coeffs = [0.0; 8];
        coeffs[i] = 1.0;
        Self(coeffs)
    }

    fn from_components(components: &[f64]) -> Self {
        assert_eq!(components.len(), 8, "su(3) has dimension 8");
        let mut coeffs = [0.0; 8];
        coeffs.copy_from_slice(components);
        Self(coeffs)
    }

    fn to_components(&self) -> Vec<f64> {
        self.0.to_vec()
    }

    /// Lie bracket for su(3): [X, Y] = XY - YX
    ///
    /// Computed using structure constants fᵢⱼₖ for O(1) performance.
    ///
    /// # Mathematical Formula
    ///
    /// For X = i·∑ᵢ aᵢ·(λᵢ/2) and Y = i·∑ⱼ bⱼ·(λⱼ/2):
    /// ```text
    /// [X, Y] = -i·∑ₖ (∑ᵢⱼ aᵢ·bⱼ·fᵢⱼₖ)·(λₖ/2)
    /// ```
    ///
    /// # Performance
    ///
    /// Uses pre-computed table of non-zero structure constants.
    /// - Old implementation: O(512) iterations with conditional checks
    /// - New implementation: O(54) direct lookups
    /// - Speedup: ~10× fewer operations
    ///
    /// # Complexity
    ///
    /// O(1) - constant time (54 multiply-adds)
    ///
    /// # Properties
    ///
    /// - Antisymmetric: [X, Y] = -[Y, X]
    /// - Jacobi identity: [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
    fn bracket(&self, other: &Self) -> Self {
        // Pre-computed non-zero structure constants f_ijk
        // SU(3) has 9 fundamental non-zero f values, each with 6 permutations = 54 total
        // Format: (i, j, k, f_ijk)
        const SQRT3_HALF: f64 = 0.866_025_403_784_438_6;

        // All 54 non-zero structure constant entries
        // Grouped by fundamental: 3 even perms (+f) + 3 odd perms (-f)
        #[rustfmt::skip]
        const STRUCTURE_CONSTANTS: [(usize, usize, usize, f64); 54] = [
            // f_012 = 1
            (0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
            (1, 0, 2, -1.0), (0, 2, 1, -1.0), (2, 1, 0, -1.0),
            // f_036 = 0.5
            (0, 3, 6, 0.5), (3, 6, 0, 0.5), (6, 0, 3, 0.5),
            (3, 0, 6, -0.5), (0, 6, 3, -0.5), (6, 3, 0, -0.5),
            // f_045 = -0.5
            (0, 4, 5, -0.5), (4, 5, 0, -0.5), (5, 0, 4, -0.5),
            (4, 0, 5, 0.5), (0, 5, 4, 0.5), (5, 4, 0, 0.5),
            // f_135 = 0.5
            (1, 3, 5, 0.5), (3, 5, 1, 0.5), (5, 1, 3, 0.5),
            (3, 1, 5, -0.5), (1, 5, 3, -0.5), (5, 3, 1, -0.5),
            // f_146 = 0.5
            (1, 4, 6, 0.5), (4, 6, 1, 0.5), (6, 1, 4, 0.5),
            (4, 1, 6, -0.5), (1, 6, 4, -0.5), (6, 4, 1, -0.5),
            // f_234 = 0.5
            (2, 3, 4, 0.5), (3, 4, 2, 0.5), (4, 2, 3, 0.5),
            (3, 2, 4, -0.5), (2, 4, 3, -0.5), (4, 3, 2, -0.5),
            // f_256 = -0.5
            (2, 5, 6, -0.5), (5, 6, 2, -0.5), (6, 2, 5, -0.5),
            (5, 2, 6, 0.5), (2, 6, 5, 0.5), (6, 5, 2, 0.5),
            // f_347 = √3/2
            (3, 4, 7, SQRT3_HALF), (4, 7, 3, SQRT3_HALF), (7, 3, 4, SQRT3_HALF),
            (4, 3, 7, -SQRT3_HALF), (3, 7, 4, -SQRT3_HALF), (7, 4, 3, -SQRT3_HALF),
            // f_567 = √3/2
            (5, 6, 7, SQRT3_HALF), (6, 7, 5, SQRT3_HALF), (7, 5, 6, SQRT3_HALF),
            (6, 5, 7, -SQRT3_HALF), (5, 7, 6, -SQRT3_HALF), (7, 6, 5, -SQRT3_HALF),
        ];

        let mut result = [0.0; 8];

        for &(i, j, k, f) in &STRUCTURE_CONSTANTS {
            result[k] += self.0[i] * other.0[j] * f;
        }

        // Apply -1 factor from [X,Y] = -i Σ f_ijk X_i Y_j (λ_k/2)
        // With T_k = iλ_k/2, the bracket coefficient is -Σ f_ijk x_i y_j
        for r in &mut result {
            *r *= -1.0;
        }

        Self(result)
    }

    #[inline]
    fn inner(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..8 {
            sum += self.0[i] * other.0[i];
        }
        sum
    }
}

// ============================================================================
// Casimir Operators for SU(3)
// ============================================================================

impl crate::Casimir for Su3Algebra {
    type Representation = crate::Su3Irrep;

    fn quadratic_casimir_eigenvalue(irrep: &Self::Representation) -> f64 {
        let p = irrep.p as f64;
        let q = irrep.q as f64;

        // c₂(p,q) = (1/3)(p² + q² + pq + 3p + 3q)
        (p * p + q * q + p * q + 3.0 * p + 3.0 * q) / 3.0
    }

    /// Cubic Casimir eigenvalue for SU(3).
    ///
    /// For representation (p, q), the cubic Casimir eigenvalue is:
    /// ```text
    /// c₃(p,q) = (1/18)(p - q)(2p + q + 3)(p + 2q + 3)
    /// ```
    ///
    /// # Properties
    ///
    /// - **Conjugation**: c₃(p,q) = -c₃(q,p). Conjugate representations have opposite c₃.
    /// - **Self-conjugate representations**: c₃ = 0 when p = q (e.g., adjoint (1,1)).
    /// - **Physical interpretation**: In QCD, distinguishes quarks (positive c₃) from
    ///   antiquarks (negative c₃) beyond the quadratic Casimir.
    ///
    /// # Examples
    ///
    /// ```text
    /// (0,0): c₃ = 0         (trivial)
    /// (1,0): c₃ = 10/9      (fundamental)
    /// (0,1): c₃ = -10/9     (antifundamental)
    /// (1,1): c₃ = 0         (adjoint, self-conjugate)
    /// (2,0): c₃ = 70/9      (symmetric tensor)
    /// ```
    ///
    /// # Reference
    ///
    /// Georgi, "Lie Algebras in Particle Physics" (1999), Chapter 7.
    fn higher_casimir_eigenvalues(irrep: &Self::Representation) -> Vec<f64> {
        let p = irrep.p as f64;
        let q = irrep.q as f64;

        // c₃(p,q) = (1/18)(p - q)(2p + q + 3)(p + 2q + 3)
        let c3 = (p - q) * (2.0 * p + q + 3.0) * (p + 2.0 * q + 3.0) / 18.0;

        vec![c3]
    }

    fn rank() -> usize {
        2 // SU(3) has rank 2 (dimension of Cartan subalgebra)
    }
}

impl Su3Algebra {
    /// Convert algebra element to 3×3 anti-Hermitian matrix
    ///
    /// Returns X = i·∑ⱼ aⱼ·(λⱼ/2) where λⱼ are Gell-Mann matrices.
    /// Convention: tr(Tₐ†Tᵦ) = ½δₐᵦ where Tₐ = iλₐ/2.
    #[must_use]
    pub fn to_matrix(&self) -> Array2<Complex64> {
        let [a1, a2, a3, a4, a5, a6, a7, a8] = self.0;
        let i = Complex64::new(0.0, 1.0);
        let sqrt3_inv = 1.0 / 3_f64.sqrt();

        // Build i·∑ⱼ aⱼ·λⱼ (then apply /2 at the end)
        let mut matrix = Array2::zeros((3, 3));

        // λ₁ = [[0,1,0],[1,0,0],[0,0,0]]
        matrix[[0, 1]] += i * a1;
        matrix[[1, 0]] += i * a1;

        // λ₂ = [[0,-i,0],[i,0,0],[0,0,0]]
        matrix[[0, 1]] += i * Complex64::new(0.0, -a2); // i·(-i·a₂) = a₂
        matrix[[1, 0]] += i * Complex64::new(0.0, a2); // i·(i·a₂) = -a₂

        // λ₃ = [[1,0,0],[0,-1,0],[0,0,0]]
        matrix[[0, 0]] += i * a3;
        matrix[[1, 1]] += -i * a3;

        // λ₄ = [[0,0,1],[0,0,0],[1,0,0]]
        matrix[[0, 2]] += i * a4;
        matrix[[2, 0]] += i * a4;

        // λ₅ = [[0,0,-i],[0,0,0],[i,0,0]]
        matrix[[0, 2]] += i * Complex64::new(0.0, -a5);
        matrix[[2, 0]] += i * Complex64::new(0.0, a5);

        // λ₆ = [[0,0,0],[0,0,1],[0,1,0]]
        matrix[[1, 2]] += i * a6;
        matrix[[2, 1]] += i * a6;

        // λ₇ = [[0,0,0],[0,0,-i],[0,i,0]]
        matrix[[1, 2]] += i * Complex64::new(0.0, -a7);
        matrix[[2, 1]] += i * Complex64::new(0.0, a7);

        // λ₈ = (1/√3)·[[1,0,0],[0,1,0],[0,0,-2]]
        matrix[[0, 0]] += i * a8 * sqrt3_inv;
        matrix[[1, 1]] += i * a8 * sqrt3_inv;
        matrix[[2, 2]] += -i * a8 * sqrt3_inv * 2.0;

        // Apply /2 for tr(Tₐ†Tᵦ) = ½δₐᵦ convention
        matrix.mapv_inplace(|z| z * 0.5);
        matrix
    }

    /// Extract algebra element from 3×3 anti-Hermitian matrix
    ///
    /// Inverse of `to_matrix()`. Uses the normalization Tr(λⱼ λₖ) = 2δⱼₖ.
    ///
    /// Given X = i·∑ⱼ aⱼ·(λⱼ/2), we have Tr(X·λⱼ) = i·aⱼ, so aⱼ = -i·Tr(X·λⱼ).
    #[must_use]
    pub fn from_matrix(matrix: &Array2<Complex64>) -> Self {
        let i = Complex64::new(0.0, 1.0);
        let neg_i = -i;

        let mut coeffs = [0.0; 8];

        // For each Gell-Mann matrix, compute aⱼ = -i·Tr(X·λⱼ)
        for j in 0..8 {
            let lambda_j = Self::gell_mann_matrix(j);
            let product = matrix.dot(&lambda_j);

            // Compute trace
            let trace = product[[0, 0]] + product[[1, 1]] + product[[2, 2]];

            // Extract coefficient: aⱼ = -i·Tr(X·λⱼ)
            coeffs[j] = (neg_i * trace).re;
        }

        Self(coeffs)
    }

    /// Get the j-th Gell-Mann matrix (j = 0..7 for λ₁..λ₈)
    ///
    /// Returns the Hermitian Gell-Mann matrix λⱼ (not i·λⱼ).
    #[must_use]
    pub fn gell_mann_matrix(j: usize) -> Array2<Complex64> {
        assert!(j < 8, "Gell-Mann matrices are indexed 0..7");

        let mut matrix = Array2::zeros((3, 3));
        let i = Complex64::new(0.0, 1.0);
        let sqrt3_inv = 1.0 / 3_f64.sqrt();

        match j {
            0 => {
                // λ₁ = [[0,1,0],[1,0,0],[0,0,0]]
                matrix[[0, 1]] = Complex64::new(1.0, 0.0);
                matrix[[1, 0]] = Complex64::new(1.0, 0.0);
            }
            1 => {
                // λ₂ = [[0,-i,0],[i,0,0],[0,0,0]]
                matrix[[0, 1]] = -i;
                matrix[[1, 0]] = i;
            }
            2 => {
                // λ₃ = [[1,0,0],[0,-1,0],[0,0,0]]
                matrix[[0, 0]] = Complex64::new(1.0, 0.0);
                matrix[[1, 1]] = Complex64::new(-1.0, 0.0);
            }
            3 => {
                // λ₄ = [[0,0,1],[0,0,0],[1,0,0]]
                matrix[[0, 2]] = Complex64::new(1.0, 0.0);
                matrix[[2, 0]] = Complex64::new(1.0, 0.0);
            }
            4 => {
                // λ₅ = [[0,0,-i],[0,0,0],[i,0,0]]
                matrix[[0, 2]] = -i;
                matrix[[2, 0]] = i;
            }
            5 => {
                // λ₆ = [[0,0,0],[0,0,1],[0,1,0]]
                matrix[[1, 2]] = Complex64::new(1.0, 0.0);
                matrix[[2, 1]] = Complex64::new(1.0, 0.0);
            }
            6 => {
                // λ₇ = [[0,0,0],[0,0,-i],[0,i,0]]
                matrix[[1, 2]] = -i;
                matrix[[2, 1]] = i;
            }
            7 => {
                // λ₈ = (1/√3)·[[1,0,0],[0,1,0],[0,0,-2]]
                matrix[[0, 0]] = Complex64::new(sqrt3_inv, 0.0);
                matrix[[1, 1]] = Complex64::new(sqrt3_inv, 0.0);
                matrix[[2, 2]] = Complex64::new(-2.0 * sqrt3_inv, 0.0);
            }
            _ => unreachable!(),
        }

        matrix
    }

    /// SU(3) structure constants `f_ijk`
    ///
    /// Returns the structure constant `f_ijk` satisfying [λᵢ, λⱼ] = 2i·∑ₖ fᵢⱼₖ·λₖ
    /// where λᵢ are the Gell-Mann matrices (0-indexed: 0..7 for λ₁..λ₈).
    ///
    /// # Properties
    ///
    /// - Totally antisymmetric: `f_ijk` = -`f_jik` = -`f_ikj` = `f_jki`
    /// - Most entries are zero (only ~24 non-zero out of 512)
    ///
    /// # Performance
    ///
    /// This is a compile-time constant lookup (O(1)) compared to O(n³) matrix multiplication.
    ///
    /// # Note
    ///
    /// This function is kept for documentation and testing purposes.
    /// The `bracket()` method uses a pre-computed table for better performance.
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    fn structure_constant(i: usize, j: usize, k: usize) -> f64 {
        const SQRT3_HALF: f64 = 0.866_025_403_784_438_6; // √3/2

        // f_ijk is totally antisymmetric, so f_iik = f_iii = 0
        if i == j || i == k || j == k {
            return 0.0;
        }

        // Use antisymmetry to canonicalize to i < j
        if i > j {
            return -Self::structure_constant(j, i, k);
        }

        // Now i < j, check all non-zero structure constants
        // (Using 0-based indexing where λ₁→0, λ₂→1, ..., λ₈→7)
        match (i, j, k) {
            // f₀₁₂ = 1  ([λ₁, λ₂] = 2i·λ₃)
            (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
            (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,

            // f₀₃₆ = 1/2  ([λ₁, λ₄] = i·λ₇)
            (0, 3, 6) | (3, 6, 0) | (6, 0, 3) => 0.5,
            (0, 6, 3) | (6, 3, 0) | (3, 0, 6) => -0.5,

            // f₀₄₅ = -1/2  ([λ₁, λ₅] = -i·λ₆)
            (0, 4, 5) | (4, 5, 0) | (5, 0, 4) => -0.5,
            (0, 5, 4) | (5, 4, 0) | (4, 0, 5) => 0.5,

            // f₁₃₅ = 1/2  ([λ₂, λ₄] = i·λ₆)
            (1, 3, 5) | (3, 5, 1) | (5, 1, 3) => 0.5,
            (1, 5, 3) | (5, 3, 1) | (3, 1, 5) => -0.5,

            // f₁₄₆ = 1/2  ([λ₂, λ₅] = i·λ₇)
            (1, 4, 6) | (4, 6, 1) | (6, 1, 4) => 0.5,
            (1, 6, 4) | (6, 4, 1) | (4, 1, 6) => -0.5,

            // f₂₃₄ = 1/2  ([λ₃, λ₄] = i·λ₅)
            (2, 3, 4) | (3, 4, 2) | (4, 2, 3) => 0.5,
            (2, 4, 3) | (4, 3, 2) | (3, 2, 4) => -0.5,

            // f₂₅₆ = -1/2  ([λ₃, λ₆] = -i·λ₇)
            (2, 5, 6) | (5, 6, 2) | (6, 2, 5) => -0.5,
            (2, 6, 5) | (6, 5, 2) | (5, 2, 6) => 0.5,

            // f₃₄₇ = √3/2  ([λ₄, λ₅] = i√3·λ₈)
            (3, 4, 7) | (4, 7, 3) | (7, 3, 4) => SQRT3_HALF,
            (3, 7, 4) | (7, 4, 3) | (4, 3, 7) => -SQRT3_HALF,

            // f₅₆₇ = √3/2  ([λ₆, λ₇] = i√3·λ₈)
            (5, 6, 7) | (6, 7, 5) | (7, 5, 6) => SQRT3_HALF,
            (5, 7, 6) | (7, 6, 5) | (6, 5, 7) => -SQRT3_HALF,

            // All other combinations are zero
            _ => 0.0,
        }
    }
}

/// SU(3) group element - 3×3 complex unitary matrix with determinant 1
///
/// Represents a color rotation in QCD.
///
/// # Representation
///
/// We use `Array2<Complex64>` from ndarray to represent the 3×3 unitary matrix.
///
/// # Constraints
///
/// - Unitarity: U† U = I
/// - Determinant: det(U) = 1
///
/// # Examples
///
/// ```
/// use lie_groups::su3::SU3;
/// use lie_groups::traits::LieGroup;
///
/// let id = SU3::identity();
/// assert!(id.verify_unitarity(1e-10));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SU3 {
    /// 3×3 complex unitary matrix
    pub(crate) matrix: Array2<Complex64>,
}

impl SU3 {
    /// Access the underlying 3×3 unitary matrix
    #[must_use]
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }

    /// Identity element
    #[must_use]
    pub fn identity() -> Self {
        let mut matrix = Array2::zeros((3, 3));
        matrix[[0, 0]] = Complex64::new(1.0, 0.0);
        matrix[[1, 1]] = Complex64::new(1.0, 0.0);
        matrix[[2, 2]] = Complex64::new(1.0, 0.0);
        Self { matrix }
    }

    /// Verify unitarity: U† U = I
    #[must_use]
    pub fn verify_unitarity(&self, tolerance: f64) -> bool {
        let adjoint = self.matrix.t().mapv(|z| z.conj());
        let product = adjoint.dot(&self.matrix);

        let mut identity = Array2::zeros((3, 3));
        identity[[0, 0]] = Complex64::new(1.0, 0.0);
        identity[[1, 1]] = Complex64::new(1.0, 0.0);
        identity[[2, 2]] = Complex64::new(1.0, 0.0);

        let diff = product - identity;
        let norm: f64 = diff
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt();

        norm < tolerance
    }

    /// Matrix inverse (equals conjugate transpose for unitary matrices)
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.t().mapv(|z| z.conj()),
        }
    }

    /// Conjugate transpose: U†
    #[must_use]
    pub fn conjugate_transpose(&self) -> Self {
        self.inverse()
    }

    /// Trace of the matrix: Tr(U)
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        self.matrix[[0, 0]] + self.matrix[[1, 1]] + self.matrix[[2, 2]]
    }

    /// Distance from identity element
    #[must_use]
    pub fn distance_to_identity(&self) -> f64 {
        // Use Frobenius norm: ||U - I||_F
        let mut identity = Array2::zeros((3, 3));
        identity[[0, 0]] = Complex64::new(1.0, 0.0);
        identity[[1, 1]] = Complex64::new(1.0, 0.0);
        identity[[2, 2]] = Complex64::new(1.0, 0.0);

        let diff = &self.matrix - &identity;
        diff.iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt()
    }

    /// Gram-Schmidt reorthogonalization for SU(3) matrices
    ///
    /// Projects a potentially corrupted matrix back onto the SU(3) manifold
    /// using Gram-Schmidt orthogonalization followed by determinant correction.
    ///
    /// # Algorithm
    ///
    /// 1. Orthogonalize columns using Modified Gram-Schmidt (MGS)
    /// 2. Normalize to ensure unitarity
    /// 3. Adjust phase to ensure det(U) = 1
    ///
    /// This avoids the log-exp round-trip that would cause infinite recursion
    /// when called from within `exp()`.
    ///
    /// # Numerical Stability
    ///
    /// Uses Modified Gram-Schmidt (not Classical GS) for better numerical stability.
    /// Key difference: projections are computed against already-orthonormalized
    /// vectors (`result.column(k)`) rather than original columns, and each
    /// projection is subtracted immediately before the next is computed.
    /// This provides O(ε) backward error vs O(κε) for Classical GS,
    /// where κ is the condition number.
    ///
    /// # Numerical Considerations (Tao priority)
    ///
    /// Uses **scale-relative threshold** for detecting linear dependence:
    /// ```text
    /// threshold = max(ε_mach × ||A||_F, ε_abs)
    /// ```
    /// where `ε_mach` ≈ 2.2e-16 (machine epsilon) and `ε_abs` = 1e-14 (absolute floor).
    ///
    /// This ensures correct behavior for matrices of any scale:
    /// - For ||A|| ~ 1: threshold ≈ 1e-14 (absolute dominates)
    /// - For ||A|| ~ 1e-8: threshold ≈ 2e-24 (relative dominates)
    /// - For ||A|| ~ 1e8: threshold ≈ 2e-8 (relative dominates)
    ///
    /// Reference: Björck, "Numerical Methods for Least Squares Problems" (1996)
    /// Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2002)
    #[must_use]
    fn gram_schmidt_project(matrix: Array2<Complex64>) -> Array2<Complex64> {
        let mut result: Array2<Complex64> = Array2::zeros((3, 3));

        // Compute Frobenius norm for scale-relative threshold
        let matrix_norm: f64 = matrix
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt();

        // Scale-relative threshold: max(ε_mach × ||A||, ε_abs)
        // This prevents false positives for small-scale matrices and
        // false negatives for large-scale matrices
        const MACHINE_EPSILON: f64 = 2.2e-16;
        const ABSOLUTE_FLOOR: f64 = 1e-14;
        let relative_threshold = MACHINE_EPSILON * matrix_norm;
        let threshold = relative_threshold.max(ABSOLUTE_FLOOR);

        // Modified Gram-Schmidt on columns
        for j in 0..3 {
            let mut col = matrix.column(j).to_owned();

            // Subtract projections onto previous columns
            for k in 0..j {
                let prev_col = result.column(k);
                let proj: Complex64 = prev_col
                    .iter()
                    .zip(col.iter())
                    .map(|(p, c)| p.conj() * c)
                    .sum();
                for i in 0..3 {
                    col[i] -= proj * prev_col[i];
                }
            }

            // Normalize
            let norm: f64 = col
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum::<f64>()
                .sqrt();

            // Detect linear dependence using scale-relative threshold
            debug_assert!(
                norm > threshold,
                "Gram-Schmidt: column {} is linearly dependent (norm = {:.2e}, threshold = {:.2e}). \
                 Input matrix is rank-deficient.",
                j,
                norm,
                threshold
            );

            if norm > threshold {
                for i in 0..3 {
                    result[[i, j]] = col[i] / norm;
                }
            }
            // Note: if norm ≤ threshold, column remains zero → det will be ~0 → identity fallback
        }

        // Ensure det = 1 (SU(N) condition)
        // Compute determinant and divide by its phase
        let det = result[[0, 0]]
            * (result[[1, 1]] * result[[2, 2]] - result[[1, 2]] * result[[2, 1]])
            - result[[0, 1]] * (result[[1, 0]] * result[[2, 2]] - result[[1, 2]] * result[[2, 0]])
            + result[[0, 2]] * (result[[1, 0]] * result[[2, 1]] - result[[1, 1]] * result[[2, 0]]);

        // Guard against zero determinant (degenerate matrix)
        // Use same scale-relative threshold for consistency
        let det_norm = det.norm();
        if det_norm < threshold {
            // Matrix is degenerate; return identity as fallback
            // This can occur if input was already corrupted
            return Array2::eye(3);
        }

        let det_phase = det / det_norm;

        // Multiply by det_phase^{-1/3} to preserve volume while fixing det
        let correction = (det_phase.conj()).powf(1.0 / 3.0);
        result.mapv_inplace(|z| z * correction);

        result
    }

    /// Compute matrix exponential using Taylor series
    ///
    /// Assumes ||matrix|| is small (≤ 0.5) for rapid convergence.
    /// This is a helper method for the scaling-and-squaring algorithm.
    ///
    /// # Arguments
    ///
    /// - `matrix`: Anti-Hermitian 3×3 matrix
    /// - `max_terms`: Maximum number of Taylor series terms
    ///
    /// # Returns
    ///
    /// exp(matrix) ∈ SU(3)
    #[must_use]
    fn exp_taylor(matrix: &Array2<Complex64>, max_terms: usize) -> Self {
        let mut result = Array2::zeros((3, 3));
        result[[0, 0]] = Complex64::new(1.0, 0.0);
        result[[1, 1]] = Complex64::new(1.0, 0.0);
        result[[2, 2]] = Complex64::new(1.0, 0.0);

        let mut term = matrix.clone();
        let mut factorial = 1.0;

        for n in 1..=max_terms {
            factorial *= n as f64;
            result += &term.mapv(|z| z / factorial);

            // Early termination if term is negligible
            let term_norm: f64 = term
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum::<f64>()
                .sqrt();

            if term_norm / factorial < 1e-14 {
                break;
            }

            if n < max_terms {
                term = term.dot(matrix);
            }
        }

        Self { matrix: result }
    }

    /// Compute matrix square root using Denman-Beavers iteration.
    ///
    /// For a unitary matrix U, this converges to U^{1/2} (principal square root).
    ///
    /// # Algorithm
    ///
    /// Y₀ = U, Z₀ = I
    /// Yₙ₊₁ = (Yₙ + Zₙ⁻¹)/2
    /// Zₙ₊₁ = (Zₙ + Yₙ⁻¹)/2
    ///
    /// Converges quadratically to Y = U^{1/2}, Z = U^{-1/2}.
    ///
    /// # Reference
    ///
    /// Higham, "Functions of Matrices", Ch. 6.
    fn matrix_sqrt_db(u: &Array2<Complex64>) -> Array2<Complex64> {
        use nalgebra::{Complex as NaComplex, Matrix3};

        // Convert ndarray to nalgebra for inversion
        fn to_nalgebra(a: &Array2<Complex64>) -> Matrix3<NaComplex<f64>> {
            Matrix3::from_fn(|i, j| NaComplex::new(a[[i, j]].re, a[[i, j]].im))
        }

        fn to_ndarray(m: &Matrix3<NaComplex<f64>>) -> Array2<Complex64> {
            Array2::from_shape_fn((3, 3), |(i, j)| Complex64::new(m[(i, j)].re, m[(i, j)].im))
        }

        let mut y = to_nalgebra(u);
        let mut z = Matrix3::<NaComplex<f64>>::identity();

        const MAX_ITERS: usize = 20;
        const TOL: f64 = 1e-14;

        for _ in 0..MAX_ITERS {
            // Compute actual inverses
            let y_inv = y.try_inverse().unwrap_or(y.adjoint()); // Fallback to adjoint if singular
            let z_inv = z.try_inverse().unwrap_or(z.adjoint());

            let y_new = (y + z_inv).scale(0.5);
            let z_new = (z + y_inv).scale(0.5);

            // Check convergence
            let diff: f64 = (y_new - y).norm();

            y = y_new;
            z = z_new;

            if diff < TOL {
                break;
            }
        }

        to_ndarray(&y)
    }
}

impl approx::AbsDiffEq for Su3Algebra {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        1e-10
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}

impl approx::RelativeEq for Su3Algebra {
    fn default_max_relative() -> Self::Epsilon {
        1e-10
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| approx::RelativeEq::relative_eq(a, b, epsilon, max_relative))
    }
}

impl fmt::Display for Su3Algebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "su(3)[")?;
        for (i, c) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", c)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for SU3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dist = self.distance_to_identity();
        write!(f, "SU(3)(d={:.4})", dist)
    }
}

/// Group multiplication: U₁ · U₂
impl Mul<&SU3> for &SU3 {
    type Output = SU3;
    fn mul(self, rhs: &SU3) -> SU3 {
        SU3 {
            matrix: self.matrix.dot(&rhs.matrix),
        }
    }
}

impl Mul<&SU3> for SU3 {
    type Output = SU3;
    fn mul(self, rhs: &SU3) -> SU3 {
        &self * rhs
    }
}

impl MulAssign<&SU3> for SU3 {
    fn mul_assign(&mut self, rhs: &SU3) {
        self.matrix = self.matrix.dot(&rhs.matrix);
    }
}

impl LieGroup for SU3 {
    const MATRIX_DIM: usize = 3;

    type Algebra = Su3Algebra;

    fn identity() -> Self {
        Self::identity()
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            matrix: self.matrix.dot(&other.matrix),
        }
    }

    fn inverse(&self) -> Self {
        Self::inverse(self)
    }

    fn conjugate_transpose(&self) -> Self {
        Self::conjugate_transpose(self)
    }

    fn adjoint_action(&self, algebra_element: &Su3Algebra) -> Su3Algebra {
        // Ad_g(X) = g X g† for matrix groups
        let x_matrix = algebra_element.to_matrix();
        let g_x = self.matrix.dot(&x_matrix);
        let g_adjoint_matrix = self.matrix.t().mapv(|z| z.conj());
        let result = g_x.dot(&g_adjoint_matrix);

        Su3Algebra::from_matrix(&result)
    }

    fn distance_to_identity(&self) -> f64 {
        Self::distance_to_identity(self)
    }

    fn exp(tangent: &Su3Algebra) -> Self {
        // Matrix exponential using scaling-and-squaring algorithm
        //
        // Algorithm: exp(X) = [exp(X/2^k)]^(2^k)
        // where k is chosen so ||X/2^k|| is small enough for Taylor series convergence
        //
        // This ensures accurate computation even for large ||X||

        let x_matrix = tangent.to_matrix();

        // Compute matrix norm (Frobenius norm)
        let norm: f64 = x_matrix
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt();

        // Choose scaling parameter k such that ||X/2^k|| ≤ 0.5
        // This ensures rapid Taylor series convergence
        let k = if norm > 0.5 {
            (norm / 0.5).log2().ceil() as usize
        } else {
            0
        };

        // Scale the matrix: Y = X / 2^k
        let scale_factor = 1.0 / (1_u64 << k) as f64;
        let scaled_matrix = x_matrix.mapv(|z| z * scale_factor);

        // Compute exp(Y) using Taylor series (converges rapidly for ||Y|| ≤ 0.5)
        let exp_scaled = SU3::exp_taylor(&scaled_matrix, 15);

        // Square k times to recover exp(X) = [exp(Y)]^(2^k)
        //
        // Numerical stability (Tao priority): Reorthogonalize after EVERY squaring.
        //
        // Rationale (Higham & Al-Mohy, 2010):
        // - Each matrix multiplication accumulates O(nε) orthogonality loss
        // - After k squarings without reorthogonalization: O(2^k × nε) error
        // - For k=10, this is O(1000ε) ≈ 1e-13, approaching catastrophic loss
        //
        // Previous code reorthogonalized every 4 squarings, which is insufficient
        // for large k or ill-conditioned intermediate results. The cost of
        // Gram-Schmidt (O(n³) = O(27) for 3×3) is negligible compared to
        // the matrix multiply (also O(n³)).
        let mut result = exp_scaled.matrix;
        for _ in 0..k {
            result = result.dot(&result);
            // Reorthogonalize after every squaring to maintain manifold constraint
            result = Self::gram_schmidt_project(result);
        }

        // Result is already orthogonalized from final loop iteration
        Self { matrix: result }
    }

    fn log(&self) -> crate::error::LogResult<Su3Algebra> {
        use crate::error::LogError;

        // Matrix logarithm for SU(3) using inverse scaling-squaring algorithm.
        //
        // Algorithm (Higham, "Functions of Matrices", Ch. 11):
        // 1. Take square roots until ||U^{1/2^k} - I|| < 0.5
        // 2. Use Taylor series for log(I + X) with ||X|| < 0.5 (fast convergence)
        // 3. Scale back: log(U) = 2^k × log(U^{1/2^k})
        //
        // This achieves ~1e-10 accuracy vs ~1e-2 for direct Taylor series.

        // Check distance from identity
        let dist = self.distance_to_identity();
        const MAX_DISTANCE: f64 = 2.0; // Maximum distance for principal branch

        if dist > MAX_DISTANCE {
            return Err(LogError::NotNearIdentity {
                distance: dist,
                threshold: MAX_DISTANCE,
            });
        }

        // Check if at identity
        if dist < 1e-14 {
            return Ok(Su3Algebra::zero());
        }

        // Phase 1: Inverse scaling via matrix square roots
        // Take square roots until ||U - I|| < 0.5 for rapid Taylor convergence
        let mut current = self.matrix.clone();
        let mut num_sqrts = 0;
        const MAX_SQRTS: usize = 32; // Prevent infinite loop
        const TARGET_NORM: f64 = 0.5; // Taylor converges rapidly for ||X|| < 0.5

        let identity: Array2<Complex64> = Array2::eye(3);

        while num_sqrts < MAX_SQRTS {
            let x_matrix = &current - &identity;
            let x_norm: f64 = x_matrix
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum::<f64>()
                .sqrt();

            if x_norm < TARGET_NORM {
                break;
            }

            // Compute matrix square root using Denman-Beavers iteration
            current = Self::matrix_sqrt_db(&current);
            num_sqrts += 1;
        }

        // Phase 2: Taylor series for log(I + X) with ||X|| < 0.5
        let x_matrix = &current - &identity;

        // Taylor series: log(I + X) = Σ_{n=1}^∞ (-1)^{n+1} X^n / n
        let mut log_matrix = x_matrix.clone();
        let mut x_power = x_matrix.clone();

        // With ||X|| < 0.5, 30 terms gives ~0.5^30/30 ≈ 3e-11 truncation error
        const N_TERMS: usize = 30;

        for n in 2..=N_TERMS {
            x_power = x_power.dot(&x_matrix);
            let coefficient = (-1.0_f64).powi(n as i32 + 1) / n as f64;
            log_matrix = log_matrix + x_power.mapv(|z| z * coefficient);
        }

        // Phase 3: Scale back: log(U) = 2^k × log(U^{1/2^k})
        let scale_factor = (1_u64 << num_sqrts) as f64;
        log_matrix = log_matrix.mapv(|z| z * scale_factor);

        // Convert result to algebra element
        Ok(Su3Algebra::from_matrix(&log_matrix))
    }
}

// ============================================================================
// Mathematical Property Implementations
// ============================================================================

use crate::traits::{Compact, SemiSimple, Simple};

/// SU(3) is compact
///
/// All elements are bounded: ||U|| = 1 for all U ∈ SU(3).
impl Compact for SU3 {}

/// SU(3) is simple
///
/// It has no non-trivial normal subgroups (except center ℤ₃).
impl Simple for SU3 {}

/// SU(3) is semi-simple
impl SemiSimple for SU3 {}

// ============================================================================
// Algebra Marker Traits
// ============================================================================

/// su(3) algebra elements are traceless by construction.
///
/// The representation `Su3Algebra::new([f64; 8])` stores coefficients in the
/// Gell-Mann basis {iλ₁, ..., iλ₈}. All Gell-Mann matrices are traceless.
impl TracelessByConstruction for Su3Algebra {}

/// su(3) algebra elements are anti-Hermitian by construction.
///
/// The representation uses {iλⱼ} where λⱼ are Hermitian Gell-Mann matrices.
impl AntiHermitianByConstruction for Su3Algebra {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let id = SU3::identity();
        assert!(id.verify_unitarity(1e-10));
        assert_relative_eq!(id.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_algebra_dimension() {
        assert_eq!(Su3Algebra::DIM, 8);
    }

    #[test]
    fn test_gell_mann_hermiticity() {
        // Verify that Gell-Mann matrices are Hermitian: λ† = λ
        for j in 0..8 {
            let lambda = Su3Algebra::gell_mann_matrix(j);
            let adjoint = lambda.t().mapv(|z| z.conj());

            for i in 0..3 {
                for k in 0..3 {
                    assert_relative_eq!(lambda[[i, k]].re, adjoint[[i, k]].re, epsilon = 1e-10);
                    assert_relative_eq!(lambda[[i, k]].im, adjoint[[i, k]].im, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gell_mann_traceless() {
        // Verify that Gell-Mann matrices are traceless
        for j in 0..8 {
            let lambda = Su3Algebra::gell_mann_matrix(j);
            let trace = lambda[[0, 0]] + lambda[[1, 1]] + lambda[[2, 2]];
            assert_relative_eq!(trace.re, 0.0, epsilon = 1e-10);
            assert_relative_eq!(trace.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matrix_roundtrip() {
        // Test that to_matrix() and from_matrix() are inverses
        let algebra = Su3Algebra([1.0, 2.0, 3.0, 0.5, -0.5, 1.5, -1.5, 0.3]);
        let matrix = algebra.to_matrix();
        let recovered = Su3Algebra::from_matrix(&matrix);

        for i in 0..8 {
            assert_relative_eq!(algebra.0[i], recovered.0[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse() {
        use crate::traits::LieGroup;

        let g = SU3::exp(&Su3Algebra([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));
        let g_inv = g.inverse();
        let product = g.compose(&g_inv);

        assert_relative_eq!(product.distance_to_identity(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_adjoint_identity() {
        use crate::traits::LieGroup;

        let e = SU3::identity();
        let x = Su3Algebra([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = e.adjoint_action(&x);

        for i in 0..8 {
            assert_relative_eq!(result.0[i], x.0[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_structure_constants_bracket() {
        use crate::traits::LieAlgebra;

        // [T₁, T₂] = -T₃ where Tₐ = iλₐ/2
        let t1 = Su3Algebra::basis_element(0);
        let t2 = Su3Algebra::basis_element(1);
        let bracket = t1.bracket(&t2);

        // Component 2 (T₃) should be -1, all others zero
        assert_relative_eq!(bracket.0[2], -1.0, epsilon = 1e-10);
        for i in [0, 1, 3, 4, 5, 6, 7] {
            assert_relative_eq!(bracket.0[i], 0.0, epsilon = 1e-10);
        }

        // Antisymmetry: [T₂, T₁] = -[T₁, T₂]
        let bracket_reversed = t2.bracket(&t1);
        for i in 0..8 {
            assert_relative_eq!(bracket.0[i], -bracket_reversed.0[i], epsilon = 1e-10);
        }

        // [T₄, T₅]: coefficient c₈ = -f₃₄₇ = -√3/2
        let t4 = Su3Algebra::basis_element(3);
        let t5 = Su3Algebra::basis_element(4);
        let bracket_45 = t4.bracket(&t5);
        let expected_c8 = -(3.0_f64.sqrt() / 2.0);
        assert_relative_eq!(bracket_45.0[7], expected_c8, epsilon = 1e-10);
    }

    #[test]
    fn test_bracket_jacobi_identity() {
        use crate::traits::LieAlgebra;

        let x = Su3Algebra::basis_element(0);
        let y = Su3Algebra::basis_element(3);
        let z = Su3Algebra::basis_element(7);

        // [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
        let t1 = x.bracket(&y.bracket(&z));
        let t2 = y.bracket(&z.bracket(&x));
        let t3 = z.bracket(&x.bracket(&y));
        let sum = t1.add(&t2).add(&t3);

        assert!(
            sum.norm() < 1e-10,
            "Jacobi identity violated for SU(3): ||sum|| = {:.2e}",
            sum.norm()
        );
    }

    #[test]
    fn test_bracket_bilinearity() {
        use crate::traits::LieAlgebra;

        let x = Su3Algebra::basis_element(0);
        let y = Su3Algebra::basis_element(2);
        let z = Su3Algebra::basis_element(5);
        let alpha = 3.7;

        // [αX + Y, Z] = α[X, Z] + [Y, Z]
        let lhs = x.scale(alpha).add(&y).bracket(&z);
        let rhs = x.bracket(&z).scale(alpha).add(&y.bracket(&z));
        for i in 0..8 {
            assert_relative_eq!(lhs.0[i], rhs.0[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_exp_large_algebra_element() {
        use crate::traits::LieGroup;

        // Test with large algebra element (||X|| > 1)
        // Old Taylor series would fail, scaling-and-squaring handles this
        let large_algebra = Su3Algebra([2.0, 1.5, -1.8, 0.9, -1.2, 1.1, -0.8, 1.3]);
        let norm = large_algebra.norm();

        // Verify this is actually large
        assert!(norm > 1.0, "Test requires ||X|| > 1, got {}", norm);

        // Compute exponential (should not panic or produce NaN)
        let g = SU3::exp(&large_algebra);

        // Verify unitarity is preserved
        assert!(
            g.verify_unitarity(1e-8),
            "Unitarity violated for large algebra element"
        );

        // Verify g is not identity (non-trivial rotation)
        assert!(g.distance_to_identity() > 0.1);
    }

    #[test]
    fn test_exp_very_small_algebra_element() {
        use crate::traits::LieGroup;

        // Test with very small algebra element
        let small_algebra = Su3Algebra([1e-8, 2e-8, -1e-8, 3e-9, -2e-9, 1e-9, -5e-10, 2e-10]);

        let g = SU3::exp(&small_algebra);

        // Should be very close to identity
        assert!(g.distance_to_identity() < 1e-7);
        assert!(g.verify_unitarity(1e-12));
    }

    #[test]
    fn test_exp_scaling_correctness() {
        use crate::traits::LieGroup;

        // Verify exp(2X) = exp(X)^2 (approximately)
        let algebra = Su3Algebra([0.5, 0.3, -0.4, 0.2, -0.3, 0.1, -0.2, 0.25]);

        let exp_x = SU3::exp(&algebra);
        let exp_2x = SU3::exp(&algebra.scale(2.0));
        let exp_x_squared = exp_x.compose(&exp_x);

        // These should be approximately equal
        let distance = exp_2x.distance(&exp_x_squared);
        assert!(
            distance < 1e-6,
            "exp(2X) should equal exp(X)^2, distance = {}",
            distance
        );
    }

    // ========================================================================
    // Property-Based Tests for Group Axioms
    // ========================================================================
    //
    // These tests use proptest to verify that SU(3) satisfies the
    // mathematical axioms of a Lie group for randomly generated elements.
    //
    // Run with: cargo test --features nightly

    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    /// Strategy for generating arbitrary SU(3) elements.
    ///
    /// We generate SU(3) elements via the exponential map from random
    /// algebra elements. Using smaller range (-0.5..0.5) for better
    /// convergence of the Taylor series exponential.
    #[cfg(feature = "proptest")]
    fn arb_su3() -> impl Strategy<Value = SU3> {
        // Use smaller range for Taylor series convergence
        let range = -0.5_f64..0.5_f64;

        (
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range,
        )
            .prop_map(|(a1, a2, a3, a4, a5, a6, a7, a8)| {
                let algebra = Su3Algebra([a1, a2, a3, a4, a5, a6, a7, a8]);
                SU3::exp(&algebra)
            })
    }

    /// Strategy for generating arbitrary `Su3Algebra` elements.
    #[cfg(feature = "proptest")]
    fn arb_su3_algebra() -> impl Strategy<Value = Su3Algebra> {
        let range = -0.5_f64..0.5_f64;

        (
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range.clone(),
            range,
        )
            .prop_map(|(a1, a2, a3, a4, a5, a6, a7, a8)| {
                Su3Algebra([a1, a2, a3, a4, a5, a6, a7, a8])
            })
    }

    #[cfg(feature = "proptest")]
    proptest! {
        /// **Group Axiom 1: Identity Element**
        ///
        /// For all U ∈ SU(3):
        /// - I · U = U (left identity)
        /// - U · I = U (right identity)
        ///
        /// where I = identity element
        #[test]
        fn prop_identity_axiom(u in arb_su3()) {
            let e = SU3::identity();

            // Left identity: I · U = U
            let left = e.compose(&u);
            prop_assert!(
                left.distance(&u) < 1e-6,
                "Left identity failed: I·U != U, distance = {}",
                left.distance(&u)
            );

            // Right identity: U · I = U
            let right = u.compose(&e);
            prop_assert!(
                right.distance(&u) < 1e-6,
                "Right identity failed: U·I != U, distance = {}",
                right.distance(&u)
            );
        }

        /// **Group Axiom 2: Inverse Element**
        ///
        /// For all U ∈ SU(3):
        /// - U · U† = I (right inverse)
        /// - U† · U = I (left inverse)
        ///
        /// where U† = conjugate transpose
        #[test]
        fn prop_inverse_axiom(u in arb_su3()) {
            let u_inv = u.inverse();

            // Right inverse: U · U† = I
            let right_product = u.compose(&u_inv);
            prop_assert!(
                right_product.is_near_identity(1e-6),
                "Right inverse failed: U·U† != I, distance = {}",
                right_product.distance_to_identity()
            );

            // Left inverse: U† · U = I
            let left_product = u_inv.compose(&u);
            prop_assert!(
                left_product.is_near_identity(1e-6),
                "Left inverse failed: U†·U != I, distance = {}",
                left_product.distance_to_identity()
            );
        }

        /// **Group Axiom 3: Associativity**
        ///
        /// For all U₁, U₂, U₃ ∈ SU(3):
        /// - (U₁ · U₂) · U₃ = U₁ · (U₂ · U₃)
        ///
        /// Group multiplication is associative.
        #[test]
        fn prop_associativity(u1 in arb_su3(), u2 in arb_su3(), u3 in arb_su3()) {
            // Left association: (U₁ · U₂) · U₃
            let left_assoc = u1.compose(&u2).compose(&u3);

            // Right association: U₁ · (U₂ · U₃)
            let right_assoc = u1.compose(&u2.compose(&u3));

            prop_assert!(
                left_assoc.distance(&right_assoc) < 1e-6,
                "Associativity failed: (U₁·U₂)·U₃ != U₁·(U₂·U₃), distance = {}",
                left_assoc.distance(&right_assoc)
            );
        }

        /// **Lie Group Property: Inverse is Smooth**
        ///
        /// For SU(3), the inverse operation is smooth (continuously differentiable).
        /// We verify this by checking that nearby elements have nearby inverses.
        #[test]
        fn prop_inverse_continuity(u in arb_su3()) {
            // Create a small perturbation
            let epsilon = 0.01;
            let perturbation = SU3::exp(&Su3Algebra([epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
            let u_perturbed = u.compose(&perturbation);

            // Check that inverses are close
            let inv_distance = u.inverse().distance(&u_perturbed.inverse());

            prop_assert!(
                inv_distance < 0.1,
                "Inverse not continuous: small perturbation caused large inverse change, distance = {}",
                inv_distance
            );
        }

        /// **Unitarity Preservation**
        ///
        /// All SU(3) operations should preserve unitarity.
        /// This is not strictly a group axiom, but it's essential for SU(3).
        #[test]
        fn prop_unitarity_preserved(u1 in arb_su3(), u2 in arb_su3()) {
            // Composition preserves unitarity
            let product = u1.compose(&u2);
            prop_assert!(
                product.verify_unitarity(1e-10),
                "Composition violated unitarity"
            );

            // Inverse preserves unitarity
            let inv = u1.inverse();
            prop_assert!(
                inv.verify_unitarity(1e-10),
                "Inverse violated unitarity"
            );
        }

        /// **Adjoint Representation: Group Homomorphism**
        ///
        /// The adjoint representation Ad: G → Aut(𝔤) is a group homomorphism:
        /// - Ad_{U₁∘U₂}(X) = Ad_{U₁}(Ad_{U₂}(X))
        ///
        /// This is a fundamental property that must hold for the adjoint action
        /// to be a valid representation of the group.
        #[test]
        fn prop_adjoint_homomorphism(
            u1 in arb_su3(),
            u2 in arb_su3(),
            x in arb_su3_algebra()
        ) {
            // Compute Ad_{U₁∘U₂}(X)
            let u_composed = u1.compose(&u2);
            let left = u_composed.adjoint_action(&x);

            // Compute Ad_{U₁}(Ad_{U₂}(X))
            let ad_u2_x = u2.adjoint_action(&x);
            let right = u1.adjoint_action(&ad_u2_x);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-6,
                "Adjoint homomorphism failed: Ad_{{U₁∘U₂}}(X) != Ad_{{U₁}}(Ad_{{U₂}}(X)), diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Identity Action**
        ///
        /// The identity element acts trivially on the Lie algebra:
        /// - Ad_I(X) = X for all X ∈ su(3)
        #[test]
        fn prop_adjoint_identity(x in arb_su3_algebra()) {
            let e = SU3::identity();
            let result = e.adjoint_action(&x);

            let diff = result.add(&x.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-10,
                "Identity action failed: Ad_I(X) != X, diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Lie Bracket Preservation**
        ///
        /// The adjoint representation preserves the Lie bracket:
        /// - Ad_U([X,Y]) = [Ad_U(X), Ad_U(Y)]
        ///
        /// This is a critical property that ensures the adjoint action
        /// is a Lie algebra automorphism.
        #[test]
        fn prop_adjoint_bracket_preservation(
            u in arb_su3(),
            x in arb_su3_algebra(),
            y in arb_su3_algebra()
        ) {
            use crate::traits::LieAlgebra;

            // Compute Ad_U([X,Y])
            let bracket_xy = x.bracket(&y);
            let left = u.adjoint_action(&bracket_xy);

            // Compute [Ad_U(X), Ad_U(Y)]
            let ad_x = u.adjoint_action(&x);
            let ad_y = u.adjoint_action(&y);
            let right = ad_x.bracket(&ad_y);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-5,
                "Bracket preservation failed: Ad_U([X,Y]) != [Ad_U(X), Ad_U(Y)], diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Inverse Property**
        ///
        /// The inverse of an element acts as the inverse transformation:
        /// - Ad_{U†}(Ad_U(X)) = X
        #[test]
        fn prop_adjoint_inverse(u in arb_su3(), x in arb_su3_algebra()) {
            // Apply Ad_U then Ad_{U†}
            let ad_u_x = u.adjoint_action(&x);
            let u_inv = u.inverse();
            let result = u_inv.adjoint_action(&ad_u_x);

            // Should recover X
            let diff = result.add(&x.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-6,
                "Inverse property failed: Ad_{{U†}}(Ad_U(X)) != X, diff norm = {}",
                diff.norm()
            );
        }
    }

    // ========================================================================
    // Exp/Log Round-Trip Tests
    // ========================================================================

    /// **Exp-Log Round-Trip Test for SU(3)**
    ///
    /// For any algebra element X with small norm, we should have:
    /// ```text
    /// log(exp(X)) ≈ X
    /// ```
    #[test]
    fn test_exp_log_roundtrip() {
        use crate::traits::{LieAlgebra, LieGroup};
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rngs::StdRng::seed_from_u64(11111);
        let dist = Uniform::new(-0.5, 0.5); // Small values for stable exp/log

        for _ in 0..50 {
            let mut coeffs = [0.0; 8];
            for coeff in &mut coeffs {
                *coeff = dist.sample(&mut rng);
            }
            let x = Su3Algebra(coeffs);

            // exp then log
            let g = SU3::exp(&x);
            let x_recovered = g.log().expect("log should succeed for exp output");

            // Inverse scaling-squaring algorithm achieves machine precision (~1e-14)
            let diff = x.add(&x_recovered.scale(-1.0));
            assert!(
                diff.norm() < 1e-10,
                "log(exp(X)) should equal X: ||diff|| = {:.2e}",
                diff.norm()
            );
        }
    }

    /// **Log-Exp Round-Trip Test for SU(3)**
    ///
    /// For any group element g, we should have:
    /// ```text
    /// exp(log(g)) = g
    /// ```
    #[test]
    fn test_log_exp_roundtrip() {
        use crate::traits::LieGroup;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rngs::StdRng::seed_from_u64(22222);
        let dist = Uniform::new(-0.5, 0.5);

        for _ in 0..50 {
            // Generate random SU(3) element via exp of small algebra element
            let mut coeffs = [0.0; 8];
            for coeff in &mut coeffs {
                *coeff = dist.sample(&mut rng);
            }
            let g = SU3::exp(&Su3Algebra(coeffs));

            // log then exp
            let x = g.log().expect("log should succeed for valid SU(3) element");
            let g_recovered = SU3::exp(&x);

            // Inverse scaling-squaring algorithm achieves machine precision (~1e-14)
            let diff = g.compose(&g_recovered.inverse()).distance_to_identity();
            assert!(
                diff < 1e-10,
                "exp(log(g)) should equal g: diff = {:.2e}",
                diff
            );
        }
    }

    // ========================================================================
    // Casimir Operator Tests
    // ========================================================================

    #[test]
    fn test_su3_casimir_trivial() {
        // (0,0) trivial: c₂ = 0
        use crate::Casimir;
        use crate::Su3Irrep;

        let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&Su3Irrep::TRIVIAL);
        assert_eq!(c2, 0.0, "Casimir of trivial representation should be 0");
    }

    #[test]
    fn test_su3_casimir_fundamental() {
        // (1,0) fundamental (quark): c₂ = 4/3
        use crate::Casimir;
        use crate::Su3Irrep;

        let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&Su3Irrep::FUNDAMENTAL);
        let expected = 4.0 / 3.0;
        assert!(
            (c2 - expected).abs() < 1e-10,
            "Casimir of fundamental should be 4/3, got {}",
            c2
        );
    }

    #[test]
    fn test_su3_casimir_antifundamental() {
        // (0,1) antifundamental: c₂ = 4/3
        use crate::Casimir;
        use crate::Su3Irrep;

        let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&Su3Irrep::ANTIFUNDAMENTAL);
        let expected = 4.0 / 3.0;
        assert!(
            (c2 - expected).abs() < 1e-10,
            "Casimir of antifundamental should be 4/3, got {}",
            c2
        );
    }

    #[test]
    fn test_su3_casimir_adjoint() {
        // (1,1) adjoint (gluon): c₂ = 3
        use crate::Casimir;
        use crate::Su3Irrep;

        let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&Su3Irrep::ADJOINT);
        assert_eq!(c2, 3.0, "Casimir of adjoint representation should be 3");
    }

    #[test]
    fn test_su3_casimir_symmetric() {
        // (2,0) symmetric (diquark): c₂ = 10/3
        use crate::Casimir;
        use crate::Su3Irrep;

        let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&Su3Irrep::SYMMETRIC);
        let expected = 10.0 / 3.0;
        assert!(
            (c2 - expected).abs() < 1e-10,
            "Casimir of symmetric should be 10/3, got {}",
            c2
        );
    }

    #[test]
    fn test_su3_casimir_formula() {
        // Test the formula c₂(p,q) = (1/3)(p² + q² + pq + 3p + 3q)
        use crate::Casimir;
        use crate::Su3Irrep;

        for p in 0..5 {
            for q in 0..5 {
                let irrep = Su3Irrep::new(p, q);
                let c2 = Su3Algebra::quadratic_casimir_eigenvalue(&irrep);

                let pf = p as f64;
                let qf = q as f64;
                let expected = (pf * pf + qf * qf + pf * qf + 3.0 * pf + 3.0 * qf) / 3.0;

                assert!(
                    (c2 - expected).abs() < 1e-10,
                    "Casimir for ({},{}) should be {}, got {}",
                    p,
                    q,
                    expected,
                    c2
                );
            }
        }
    }

    #[test]
    fn test_su3_rank() {
        // SU(3) has rank 2
        use crate::Casimir;

        assert_eq!(Su3Algebra::rank(), 2, "SU(3) should have rank 2");
        assert_eq!(
            Su3Algebra::num_casimirs(),
            2,
            "SU(3) should have 2 Casimir operators"
        );
    }

    // ==========================================================================
    // Cubic Casimir Tests
    // ==========================================================================

    #[test]
    fn test_su3_cubic_casimir_trivial() {
        // (0,0) trivial: c₃ = 0
        use crate::Casimir;
        use crate::Su3Irrep;

        let c3_vec = Su3Algebra::higher_casimir_eigenvalues(&Su3Irrep::TRIVIAL);
        assert_eq!(c3_vec.len(), 1, "Should return exactly one higher Casimir");
        assert_eq!(c3_vec[0], 0.0, "Cubic Casimir of trivial should be 0");
    }

    #[test]
    fn test_su3_cubic_casimir_fundamental() {
        // (1,0) fundamental: c₃ = (1/18)(1-0)(2+0+3)(1+0+3) = (1/18)(1)(5)(4) = 20/18 = 10/9
        use crate::Casimir;
        use crate::Su3Irrep;

        let c3_vec = Su3Algebra::higher_casimir_eigenvalues(&Su3Irrep::FUNDAMENTAL);
        let c3 = c3_vec[0];
        let expected = 10.0 / 9.0;
        assert!(
            (c3 - expected).abs() < 1e-10,
            "Cubic Casimir of fundamental should be 10/9, got {}",
            c3
        );
    }

    #[test]
    fn test_su3_cubic_casimir_antifundamental() {
        // (0,1) antifundamental: c₃ = (1/18)(0-1)(0+1+3)(0+2+3) = (1/18)(-1)(4)(5) = -20/18 = -10/9
        use crate::Casimir;
        use crate::Su3Irrep;

        let c3_vec = Su3Algebra::higher_casimir_eigenvalues(&Su3Irrep::ANTIFUNDAMENTAL);
        let c3 = c3_vec[0];
        let expected = -10.0 / 9.0;
        assert!(
            (c3 - expected).abs() < 1e-10,
            "Cubic Casimir of antifundamental should be -10/9, got {}",
            c3
        );
    }

    #[test]
    fn test_su3_cubic_casimir_adjoint() {
        // (1,1) adjoint: c₃ = (1/18)(1-1)(2+1+3)(1+2+3) = 0 (self-conjugate)
        use crate::Casimir;
        use crate::Su3Irrep;

        let c3_vec = Su3Algebra::higher_casimir_eigenvalues(&Su3Irrep::ADJOINT);
        let c3 = c3_vec[0];
        assert!(
            c3.abs() < 1e-10,
            "Cubic Casimir of adjoint (self-conjugate) should be 0, got {}",
            c3
        );
    }

    #[test]
    fn test_su3_cubic_casimir_symmetric() {
        // (2,0) symmetric: c₃ = (1/18)(2-0)(4+0+3)(2+0+3) = (1/18)(2)(7)(5) = 70/18 = 35/9
        use crate::Casimir;
        use crate::Su3Irrep;

        let c3_vec = Su3Algebra::higher_casimir_eigenvalues(&Su3Irrep::SYMMETRIC);
        let c3 = c3_vec[0];
        let expected = 70.0 / 18.0;
        assert!(
            (c3 - expected).abs() < 1e-10,
            "Cubic Casimir of symmetric should be 70/18, got {}",
            c3
        );
    }

    #[test]
    fn test_su3_cubic_casimir_conjugation_symmetry() {
        // c₃(p,q) = -c₃(q,p) for conjugate representations
        use crate::Casimir;
        use crate::Su3Irrep;

        for p in 0..5 {
            for q in 0..5 {
                let irrep_pq = Su3Irrep::new(p, q);
                let irrep_qp = Su3Irrep::new(q, p);

                let c3_pq = Su3Algebra::higher_casimir_eigenvalues(&irrep_pq)[0];
                let c3_qp = Su3Algebra::higher_casimir_eigenvalues(&irrep_qp)[0];

                assert!(
                    (c3_pq + c3_qp).abs() < 1e-10,
                    "c₃({},{}) = {} should equal -c₃({},{}) = {}",
                    p,
                    q,
                    c3_pq,
                    q,
                    p,
                    -c3_qp
                );
            }
        }
    }

    #[test]
    fn test_su3_cubic_casimir_formula() {
        // Test the formula c₃(p,q) = (1/18)(p-q)(2p+q+3)(p+2q+3)
        use crate::Casimir;
        use crate::Su3Irrep;

        for p in 0..5 {
            for q in 0..5 {
                let irrep = Su3Irrep::new(p, q);
                let c3 = Su3Algebra::higher_casimir_eigenvalues(&irrep)[0];

                let pf = p as f64;
                let qf = q as f64;
                let expected = (pf - qf) * (2.0 * pf + qf + 3.0) * (pf + 2.0 * qf + 3.0) / 18.0;

                assert!(
                    (c3 - expected).abs() < 1e-10,
                    "Cubic Casimir for ({},{}) should be {}, got {}",
                    p,
                    q,
                    expected,
                    c3
                );
            }
        }
    }

    // ==========================================================================
    // Edge Case Tests: Near-Singular Scenarios
    // ==========================================================================
    //
    // These tests verify numerical stability at boundary conditions, particularly
    // for the scale-relative threshold improvements.

    /// Test Gram-Schmidt with very small matrices
    ///
    /// Ensures the scale-relative threshold handles matrices with small norms
    /// without false positives for numerical breakdown.
    #[test]
    fn test_gram_schmidt_small_matrix() {
        // Create a matrix with small norm (scale ~ 1e-6)
        let small_algebra = Su3Algebra([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let small_element = SU3::exp(&small_algebra);

        // Should still be unitary
        assert!(
            small_element.verify_unitarity(1e-10),
            "Small element should be unitary"
        );

        // exp followed by log should round-trip
        let recovered = small_element
            .log()
            .expect("log should succeed for near-identity");
        let diff = (recovered.0[0] - small_algebra.0[0]).abs();
        assert!(
            diff < 1e-10,
            "exp/log round-trip failed for small element: diff = {}",
            diff
        );
    }

    /// Test Gram-Schmidt with very large matrices
    ///
    /// Ensures the scale-relative threshold handles matrices with large norms
    /// correctly, where absolute thresholds would be too tight.
    #[test]
    fn test_gram_schmidt_large_matrix() {
        use crate::traits::LieGroup;

        // Create a matrix with large rotation angle (near π)
        let large_algebra = Su3Algebra([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let large_element = SU3::exp(&large_algebra);

        // Should still be unitary after exp (reorthogonalization should help)
        assert!(
            large_element.verify_unitarity(1e-8),
            "Large rotation element should be unitary, distance = {}",
            large_element.distance_to_identity()
        );
    }

    /// Test exp with repeated squaring stability
    ///
    /// Verifies that reorthogonalization every squaring prevents drift.
    #[test]
    fn test_exp_repeated_squaring_stability() {
        use crate::traits::LieGroup;

        // Angle that requires multiple squaring steps (2^k scaling)
        // θ = 3.0 requires about 3-4 squaring steps
        let algebra = Su3Algebra([1.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0]);
        let element = SU3::exp(&algebra);

        // Check unitarity is preserved
        assert!(
            element.verify_unitarity(1e-10),
            "Repeated squaring should preserve unitarity"
        );

        // Check special unitarity: U†U = I implies det(U) has magnitude 1
        // For SU(3), we also require det = 1 (not just |det| = 1)
        // We verify this indirectly through verify_special_unitarity if available,
        // or by checking trace(U†U) = 3 (trace of identity)
        let u_dag_u = element.conjugate_transpose().compose(&element);
        let trace = u_dag_u.trace();
        assert!(
            (trace.re - 3.0).abs() < 1e-10 && trace.im.abs() < 1e-10,
            "U†U should have trace 3, got {:.6}+{:.6}i",
            trace.re,
            trace.im
        );
    }

    /// Test exp/log round-trip near group boundary
    ///
    /// For elements near θ = π, log can have numerical difficulties.
    /// This test verifies graceful handling.
    #[test]
    fn test_exp_log_near_boundary() {
        use crate::traits::LieGroup;

        // Angle close to but not exactly at π
        let theta = std::f64::consts::PI - 0.1;
        let algebra = Su3Algebra([theta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let element = SU3::exp(&algebra);

        // Element should be valid
        assert!(
            element.verify_unitarity(1e-10),
            "Near-boundary element should be unitary"
        );

        // Log might fail or give approximate result - just verify it doesn't panic
        if let Ok(recovered) = element.log() {
            // If log succeeds, verify exp(log(g)) = g
            let round_trip = SU3::exp(&recovered);
            let distance = element.distance(&round_trip);
            assert!(
                distance < 1e-6,
                "Round trip should be close, distance = {}",
                distance
            );
        }
        // Log failure near θ=π is acceptable
        // The element is still valid, just log has a singularity
    }

    /// Test composition preserves unitarity under repeated operations
    ///
    /// Accumulated errors from many compositions could drift from SU(3).
    #[test]
    fn test_composition_stability() {
        use crate::traits::LieGroup;

        // Create several distinct SU(3) elements
        let u1 = SU3::exp(&Su3Algebra([0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let u2 = SU3::exp(&Su3Algebra([0.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let u3 = SU3::exp(&Su3Algebra([0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0]));

        // Compose many times
        let mut result = SU3::identity();
        for _ in 0..100 {
            result = result.compose(&u1);
            result = result.compose(&u2);
            result = result.compose(&u3);
        }

        // Should still be unitary (SU(3) is closed under composition)
        assert!(
            result.verify_unitarity(1e-8),
            "100 compositions should still be unitary"
        );
    }
}
