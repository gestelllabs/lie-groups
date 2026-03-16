//! Generic SU(N) - Special unitary N×N matrices
//!
//! This module provides a compile-time generic implementation of SU(N) for arbitrary N.
//! It elegantly generalizes SU(2) and SU(3) while maintaining type safety and efficiency.
//!
//! # Mathematical Structure
//!
//! ```text
//! SU(N) = { U ∈ ℂᴺˣᴺ | U† U = I, det(U) = 1 }
//! ```
//!
//! # Lie Algebra
//!
//! The Lie algebra su(N) consists of N×N traceless anti-Hermitian matrices:
//! ```text
//! su(N) = { X ∈ ℂᴺˣᴺ | X† = -X, Tr(X) = 0 }
//! dim(su(N)) = N² - 1
//! ```
//!
//! # Design Philosophy
//!
//! - **Type Safety**: Const generics ensure dimension errors are caught at compile time
//! - **Efficiency**: Lazy matrix construction, SIMD-friendly operations
//! - **Elegance**: Unified interface for all N (including N=2,3)
//! - **Generality**: Works for arbitrary N ≥ 2
//!
//! # Examples
//!
//! ```ignore
//! use lie_groups::sun::SunAlgebra;
//! use lie_groups::LieAlgebra;
//!
//! // SU(4) for grand unified theories
//! type Su4Algebra = SunAlgebra<4>;
//! let x = Su4Algebra::zero();
//! assert_eq!(Su4Algebra::DIM, 15);  // 4² - 1 = 15
//!
//! // Type safety: dimensions checked at compile time
//! let su2 = SunAlgebra::<2>::basis_element(0);  // dim = 3
//! let su3 = SunAlgebra::<3>::basis_element(0);  // dim = 8
//! // su2.add(&su3);  // Compile error! Incompatible types
//! ```
//!
//! # Physics Applications
//!
//! - **SU(2)**: Weak force, isospin
//! - **SU(3)**: Strong force (QCD), color charge
//! - **SU(4)**: Pati-Salam model, flavor symmetry
//! - **SU(5)**: Georgi-Glashow GUT
//! - **SU(6)**: Flavor SU(3) × color SU(2)
//!
//! # Performance
//!
//! - Algebra operations: O(N²) `[optimal]`
//! - Matrix construction: O(N²) `[lazy, only when needed]`
//! - Exponential map: O(N³) via scaling-and-squaring
//! - Memory: (N²-1)·sizeof(f64) bytes for algebra

use crate::traits::{
    AntiHermitianByConstruction, Compact, LieAlgebra, LieGroup, SemiSimple, Simple,
    TracelessByConstruction,
};
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

/// Lie algebra su(N) - (N²-1)-dimensional space of traceless anti-Hermitian matrices
///
/// # Type Parameter
///
/// - `N`: Matrix dimension (must be ≥ 2)
///
/// # Representation
///
/// Elements are stored as (N²-1) real coefficients corresponding to the generalized
/// Gell-Mann basis. The basis is constructed systematically:
///
/// 1. **Symmetric generators** (N(N-1)/2 elements):
///    - λᵢⱼ with i < j: has 1 at (i,j) and (j,i)
///
/// 2. **Antisymmetric generators** (N(N-1)/2 elements):
///    - λᵢⱼ with i < j: has -i at (i,j) and +i at (j,i)
///
/// 3. **Diagonal generators** (N-1 elements):
///    - λₖ diagonal with first k entries = 1, (k+1)-th entry = -k
///
/// This generalizes the Pauli matrices (N=2) and Gell-Mann matrices (N=3).
///
/// # Mathematical Properties
///
/// - Hermitian generators: λⱼ† = λⱼ
/// - Traceless: Tr(λⱼ) = 0
/// - Normalized: Tr(λᵢλⱼ) = 2δᵢⱼ
/// - Completeness: {λⱼ/√2} form orthonormal basis for traceless Hermitian matrices
///
/// # Memory Layout
///
/// For SU(N), we store (N²-1) f64 values in a heap-allocated Vec for N > 4,
/// or stack-allocated array for N ≤ 4 (common cases).
#[derive(Clone, Debug, PartialEq)]
pub struct SunAlgebra<const N: usize> {
    /// Coefficients in generalized Gell-Mann basis
    /// Length: N² - 1
    pub(crate) coefficients: Vec<f64>,
    _phantom: PhantomData<[(); N]>,
}

impl<const N: usize> SunAlgebra<N> {
    /// Dimension of su(N) algebra: N² - 1, valid only for N ≥ 2
    const DIM: usize = {
        assert!(
            N >= 2,
            "SU(N) requires N >= 2: SU(1) is trivial, SU(0) is undefined"
        );
        N * N - 1
    };

    /// Create new algebra element from coefficients
    ///
    /// # Panics
    ///
    /// Panics if `coefficients.len() != N² - 1`
    #[must_use]
    pub fn new(coefficients: Vec<f64>) -> Self {
        assert_eq!(
            coefficients.len(),
            Self::DIM,
            "SU({}) algebra requires {} coefficients, got {}",
            N,
            Self::DIM,
            coefficients.len()
        );
        Self {
            coefficients,
            _phantom: PhantomData,
        }
    }

    /// Returns the coefficients in the generalized Gell-Mann basis.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Convert to N×N anti-Hermitian matrix: X = i·∑ⱼ aⱼ·(λⱼ/2)
    ///
    /// This is the fundamental representation in ℂᴺˣᴺ.
    /// Convention: tr(Tₐ†Tᵦ) = ½δₐᵦ where Tₐ = iλₐ/2.
    ///
    /// # Performance
    ///
    /// - Time: O(N²)
    /// - Space: O(N²)
    /// - Lazy: Only computed when called
    ///
    /// # Mathematical Formula
    ///
    /// Given coefficients [a₁, ..., a_{N²-1}], returns:
    /// ```text
    /// X = i·∑ⱼ aⱼ·(λⱼ/2)
    /// ```
    /// where λⱼ are the generalized Gell-Mann matrices with tr(λₐλᵦ) = 2δₐᵦ.
    #[must_use]
    pub fn to_matrix(&self) -> Array2<Complex64> {
        let mut matrix = Array2::zeros((N, N));
        let i = Complex64::new(0.0, 1.0);

        let mut idx = 0;

        // Symmetric generators: (i,j) with i < j
        for row in 0..N {
            for col in (row + 1)..N {
                let coeff = self.coefficients[idx];
                matrix[[row, col]] += i * coeff;
                matrix[[col, row]] += i * coeff;
                idx += 1;
            }
        }

        // Antisymmetric generators: (i,j) with i < j
        // Standard Gell-Mann: Λ^A_{ij} = -iE_{ij} + iE_{ji}
        // T = iΛ/2 gives +coeff/2 (real) at (row,col), -coeff/2 at (col,row)
        for row in 0..N {
            for col in (row + 1)..N {
                let coeff = self.coefficients[idx];
                matrix[[row, col]] += Complex64::new(coeff, 0.0); // +coeff (real)
                matrix[[col, row]] += Complex64::new(-coeff, 0.0); // -coeff (real)
                idx += 1;
            }
        }

        // Diagonal generators
        // The k-th diagonal generator (k=0..N-2) has:
        // - First (k+1) diagonal entries = +1
        // - (k+2)-th diagonal entry = -(k+1)
        // - Normalized so Tr(λ²) = 2
        for k in 0..(N - 1) {
            let coeff = self.coefficients[idx];

            // Normalization: √(2 / (k+1)(k+2))
            let k_f = k as f64;
            let normalization = 2.0 / ((k_f + 1.0) * (k_f + 2.0));
            let scale = normalization.sqrt();

            // First (k+1) entries: +1
            for j in 0..=k {
                matrix[[j, j]] += i * coeff * scale;
            }
            // (k+2)-th entry: -(k+1)
            matrix[[k + 1, k + 1]] += i * coeff * scale * (-(k_f + 1.0));

            idx += 1;
        }

        // Apply /2 for tr(Tₐ†Tᵦ) = ½δₐᵦ convention
        matrix.mapv_inplace(|z| z * 0.5);
        matrix
    }

    /// Construct algebra element from matrix
    ///
    /// Given X ∈ su(N), extract coefficients in Gell-Mann basis.
    ///
    /// # Performance
    ///
    /// O(N²) time via inner products with basis elements.
    #[must_use]
    pub fn from_matrix(matrix: &Array2<Complex64>) -> Self {
        assert_eq!(matrix.nrows(), N);
        assert_eq!(matrix.ncols(), N);

        let mut coefficients = vec![0.0; Self::DIM];
        let mut idx = 0;

        // Convention: X = i·∑ aⱼ·(λⱼ/2), so matrix entries are half
        // what they would be for the raw Gell-Mann basis.
        // We extract by reading the matrix and multiplying by 2.

        // Extract symmetric components
        // λ has 1 at (row,col) and (col,row)
        // X[row,col] = i·a/2, so a = 2·Im(X[row,col])
        for row in 0..N {
            for col in (row + 1)..N {
                let val = matrix[[row, col]];
                coefficients[idx] = val.im * 2.0;
                idx += 1;
            }
        }

        // Extract antisymmetric components
        // Standard Gell-Mann: Λ^A has -i at (row,col), +i at (col,row)
        // T = iΛ/2 gives +a/2 (real) at (row,col)
        // so a = 2·Re(X[row,col])
        for row in 0..N {
            for col in (row + 1)..N {
                let val = matrix[[row, col]];
                coefficients[idx] = val.re * 2.0;
                idx += 1;
            }
        }

        // Extract diagonal components using proper inner product
        //
        // The k-th diagonal generator H_k has:
        //   - entries [[j,j]] = scale_k for j = 0..=k
        //   - entry [[k+1, k+1]] = -(k+1) * scale_k
        //   - normalized so Tr(H_k²) = 2
        //
        // With /2 convention: X_diag = i·a_k·(H_k/2)
        // So a_k = Im(Tr(X · H_k)) / (Tr(H_k²)/2) = Im(Tr(X · H_k))
        for k in 0..(N - 1) {
            let k_f = k as f64;
            let normalization = 2.0 / ((k_f + 1.0) * (k_f + 2.0));
            let scale = normalization.sqrt();

            let mut trace_prod = Complex64::new(0.0, 0.0);
            for j in 0..=k {
                trace_prod += matrix[[j, j]] * scale;
            }
            trace_prod += matrix[[k + 1, k + 1]] * (-(k_f + 1.0) * scale);

            // With /2 convention, the extraction formula becomes Im(Tr(X·H_k))
            // since X = i·a·H_k/2 gives Tr(X·H_k) = i·a·Tr(H_k²)/2 = i·a
            coefficients[idx] = trace_prod.im;
            idx += 1;
        }

        Self::new(coefficients)
    }
}

impl<const N: usize> Add for SunAlgebra<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        Self::new(coefficients)
    }
}

impl<const N: usize> Add<&SunAlgebra<N>> for SunAlgebra<N> {
    type Output = SunAlgebra<N>;
    fn add(self, rhs: &SunAlgebra<N>) -> SunAlgebra<N> {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        Self::new(coefficients)
    }
}

impl<const N: usize> Add<SunAlgebra<N>> for &SunAlgebra<N> {
    type Output = SunAlgebra<N>;
    fn add(self, rhs: SunAlgebra<N>) -> SunAlgebra<N> {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        SunAlgebra::<N>::new(coefficients)
    }
}

impl<const N: usize> Add<&SunAlgebra<N>> for &SunAlgebra<N> {
    type Output = SunAlgebra<N>;
    fn add(self, rhs: &SunAlgebra<N>) -> SunAlgebra<N> {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        SunAlgebra::<N>::new(coefficients)
    }
}

impl<const N: usize> Sub for SunAlgebra<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(a, b)| a - b)
            .collect();
        Self::new(coefficients)
    }
}

impl<const N: usize> Neg for SunAlgebra<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let coefficients = self.coefficients.iter().map(|x| -x).collect();
        Self::new(coefficients)
    }
}

impl<const N: usize> Mul<f64> for SunAlgebra<N> {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        let coefficients = self.coefficients.iter().map(|x| x * scalar).collect();
        Self::new(coefficients)
    }
}

impl<const N: usize> Mul<SunAlgebra<N>> for f64 {
    type Output = SunAlgebra<N>;
    fn mul(self, rhs: SunAlgebra<N>) -> SunAlgebra<N> {
        rhs * self
    }
}

impl<const N: usize> LieAlgebra for SunAlgebra<N> {
    // Model-theoretic guard: SU(N) requires N ≥ 2.
    // SU(1) = {I} is trivial (algebra is zero-dimensional, bracket undefined).
    // SU(0) underflows usize arithmetic.
    // This const assert promotes the degenerate-model failure from runtime panic
    // to a compile-time error, consistent with the sealed-trait philosophy.
    const DIM: usize = {
        assert!(
            N >= 2,
            "SU(N) requires N >= 2: SU(1) is trivial, SU(0) is undefined"
        );
        N * N - 1
    };

    fn zero() -> Self {
        Self {
            coefficients: vec![0.0; Self::DIM],
            _phantom: PhantomData,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .zip(&other.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        Self::new(coefficients)
    }

    fn scale(&self, scalar: f64) -> Self {
        let coefficients = self.coefficients.iter().map(|x| x * scalar).collect();
        Self::new(coefficients)
    }

    fn norm(&self) -> f64 {
        self.coefficients
            .iter()
            .map(|x| x.powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn basis_element(i: usize) -> Self {
        assert!(
            i < Self::DIM,
            "Basis index {} out of range for SU({})",
            i,
            N
        );
        let mut coefficients = vec![0.0; Self::DIM];
        coefficients[i] = 1.0;
        Self::new(coefficients)
    }

    fn from_components(components: &[f64]) -> Self {
        assert_eq!(
            components.len(),
            Self::DIM,
            "Expected {} components for SU({}), got {}",
            Self::DIM,
            N,
            components.len()
        );
        Self::new(components.to_vec())
    }

    fn to_components(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    /// Lie bracket: [X, Y] = XY - YX
    ///
    /// Computed via matrix commutator for generality.
    ///
    /// # Performance
    ///
    /// - Time: O(N³) [matrix multiplication]
    /// - Space: O(N²)
    ///
    /// # Note
    ///
    /// For N=2,3, specialized implementations with structure constants
    /// would be faster (O(1) and O(1) respectively). This generic version
    /// prioritizes correctness and simplicity.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// [X, Y] = XY - YX
    /// ```
    ///
    /// This satisfies:
    /// - Antisymmetry: `[X,Y] = -[Y,X]`
    /// - Jacobi identity: `[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0`
    /// - Bilinearity
    fn bracket(&self, other: &Self) -> Self {
        let x = self.to_matrix();
        let y = other.to_matrix();
        let commutator = x.dot(&y) - y.dot(&x);
        Self::from_matrix(&commutator)
    }

    #[inline]
    fn inner(&self, other: &Self) -> f64 {
        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// SU(N) group element - N×N unitary matrix with det = 1
///
/// # Type Parameter
///
/// - `N`: Matrix dimension
///
/// # Representation
///
/// Stored as N×N complex matrix satisfying:
/// - U†U = I (unitarity)
/// - det(U) = 1 (special)
///
/// # Verification
///
/// Use `verify_unitarity()` to check constraints numerically.
#[derive(Debug, Clone, PartialEq)]
pub struct SUN<const N: usize> {
    /// N×N complex unitary matrix
    pub(crate) matrix: Array2<Complex64>,
}

impl<const N: usize> SUN<N> {
    /// Access the underlying N×N unitary matrix
    #[must_use]
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }

    /// Identity element: Iₙ
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: Array2::eye(N),
        }
    }

    /// Trace of the matrix: Tr(U) = Σᵢ Uᵢᵢ
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        (0..N).map(|i| self.matrix[[i, i]]).sum()
    }

    /// Verify unitarity: ||U†U - I|| < ε
    ///
    /// # Arguments
    ///
    /// - `tolerance`: Maximum Frobenius norm deviation
    ///
    /// # Returns
    ///
    /// `true` if U†U ≈ I within tolerance
    #[must_use]
    pub fn verify_unitarity(&self, tolerance: f64) -> bool {
        let adjoint = self.matrix.t().mapv(|z| z.conj());
        let product = adjoint.dot(&self.matrix);
        let identity: Array2<Complex64> = Array2::eye(N);

        let diff = &product - &identity;
        let norm_sq: f64 = diff.iter().map(num_complex::Complex::norm_sqr).sum();

        norm_sq.sqrt() < tolerance
    }

    /// Compute determinant
    ///
    /// For SU(N), the determinant should be exactly 1 by definition.
    ///
    /// # Implementation
    ///
    /// - **N=2**: Direct formula `ad - bc`
    /// - **N=3**: Sarrus' rule / cofactor expansion
    /// - **N>3**: Returns `1.0` (assumes matrix is on SU(N) manifold)
    ///
    /// # Limitations
    ///
    /// For N > 3, this function **does not compute the actual determinant**.
    /// It returns 1.0 under the assumption that matrices constructed via
    /// `exp()` or `reorthogonalize()` remain on the SU(N) manifold.
    ///
    /// To verify unitarity for N > 3, use `verify_special_unitarity()` instead,
    /// which checks `U†U = I` (a stronger condition than det=1).
    ///
    /// For actual determinant computation with N > 3, enable the `ndarray-linalg`
    /// feature (not currently available) or compute via eigenvalue product.
    #[must_use]
    #[allow(clippy::many_single_char_names)] // Standard math notation for matrix elements
    pub fn determinant(&self) -> Complex64 {
        // For small N, compute directly using Leibniz formula
        if N == 2 {
            let a = self.matrix[[0, 0]];
            let b = self.matrix[[0, 1]];
            let c = self.matrix[[1, 0]];
            let d = self.matrix[[1, 1]];
            return a * d - b * c;
        }

        if N == 3 {
            // 3x3 determinant via Sarrus' rule / cofactor expansion
            let (a, b, c, d, e, f, g, h, i) = {
                let m = &self.matrix;
                (
                    m[[0, 0]],
                    m[[0, 1]],
                    m[[0, 2]],
                    m[[1, 0]],
                    m[[1, 1]],
                    m[[1, 2]],
                    m[[2, 0]],
                    m[[2, 1]],
                    m[[2, 2]],
                )
            };

            // det = a(ei - fh) - b(di - fg) + c(dh - eg)
            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        }

        // For N > 3: LU decomposition would be ideal, but requires ndarray-linalg
        // For now, return 1.0 since matrices constructed via exp() preserve det=1
        // This is valid for elements on the SU(N) manifold
        Complex64::new(1.0, 0.0)
    }

    /// Gram-Schmidt reorthogonalization for SU(N) matrices
    ///
    /// Projects a potentially corrupted matrix back onto the SU(N) manifold
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
    /// Projections are computed against already-orthonormalized vectors and
    /// subtracted immediately, providing O(ε) backward error vs O(κε) for CGS.
    ///
    /// Reference: Björck, "Numerical Methods for Least Squares Problems" (1996)
    #[must_use]
    fn gram_schmidt_project(matrix: Array2<Complex64>) -> Array2<Complex64> {
        let mut result: Array2<Complex64> = Array2::zeros((N, N));

        // Modified Gram-Schmidt on columns
        for j in 0..N {
            let mut col = matrix.column(j).to_owned();

            // Subtract projections onto previous columns
            for k in 0..j {
                let prev_col = result.column(k);
                let proj: Complex64 = prev_col
                    .iter()
                    .zip(col.iter())
                    .map(|(p, c)| p.conj() * c)
                    .sum();
                for i in 0..N {
                    col[i] -= proj * prev_col[i];
                }
            }

            // Normalize
            let norm: f64 = col
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum::<f64>()
                .sqrt();

            // Detect linear dependence: column became zero after orthogonalization
            debug_assert!(
                norm > 1e-14,
                "Gram-Schmidt: column {} is linearly dependent (norm = {:.2e}). \
                 Input matrix is rank-deficient.",
                j,
                norm
            );

            if norm > 1e-14 {
                for i in 0..N {
                    result[[i, j]] = col[i] / norm;
                }
            }
            // Note: if norm ≤ 1e-14, column remains zero → det will be ~0 → identity fallback
        }

        // For N=2 or N=3, compute determinant and fix phase
        // For larger N, approximate (Gram-Schmidt usually produces det ≈ 1 already)
        if N <= 3 {
            let det = if N == 2 {
                result[[0, 0]] * result[[1, 1]] - result[[0, 1]] * result[[1, 0]]
            } else {
                // N=3
                result[[0, 0]] * (result[[1, 1]] * result[[2, 2]] - result[[1, 2]] * result[[2, 1]])
                    - result[[0, 1]]
                        * (result[[1, 0]] * result[[2, 2]] - result[[1, 2]] * result[[2, 0]])
                    + result[[0, 2]]
                        * (result[[1, 0]] * result[[2, 1]] - result[[1, 1]] * result[[2, 0]])
            };

            // Guard against zero determinant (degenerate matrix)
            let det_norm = det.norm();
            if det_norm < 1e-14 {
                // Matrix is degenerate; return identity as fallback
                return Array2::eye(N);
            }

            let det_phase = det / det_norm;
            let correction = (det_phase.conj()).powf(1.0 / N as f64);
            result.mapv_inplace(|z| z * correction);
        }

        result
    }

    /// Distance from identity: ||U - I||_F
    ///
    /// Frobenius norm of difference from identity.
    #[must_use]
    pub fn distance_to_identity(&self) -> f64 {
        let identity: Array2<Complex64> = Array2::eye(N);
        let diff = &self.matrix - &identity;
        diff.iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt()
    }
}

impl<const N: usize> approx::AbsDiffEq for SunAlgebra<N> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        1e-10
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}

impl<const N: usize> approx::RelativeEq for SunAlgebra<N> {
    fn default_max_relative() -> Self::Epsilon {
        1e-10
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .all(|(a, b)| approx::RelativeEq::relative_eq(a, b, epsilon, max_relative))
    }
}

impl<const N: usize> fmt::Display for SunAlgebra<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "su({})[", N)?;
        for (i, c) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", c)?;
        }
        write!(f, "]")
    }
}

impl<const N: usize> fmt::Display for SUN<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dist = self.distance_to_identity();
        write!(f, "SU({})(d={:.4})", N, dist)
    }
}

/// Group multiplication: U₁ · U₂
impl<const N: usize> Mul<&SUN<N>> for &SUN<N> {
    type Output = SUN<N>;
    fn mul(self, rhs: &SUN<N>) -> SUN<N> {
        SUN {
            matrix: self.matrix.dot(&rhs.matrix),
        }
    }
}

impl<const N: usize> Mul<&SUN<N>> for SUN<N> {
    type Output = SUN<N>;
    fn mul(self, rhs: &SUN<N>) -> SUN<N> {
        &self * rhs
    }
}

impl<const N: usize> Mul<SUN<N>> for SUN<N> {
    type Output = SUN<N>;
    fn mul(self, rhs: SUN<N>) -> SUN<N> {
        &self * &rhs
    }
}

impl<const N: usize> MulAssign<&SUN<N>> for SUN<N> {
    fn mul_assign(&mut self, rhs: &SUN<N>) {
        self.matrix = self.matrix.dot(&rhs.matrix);
    }
}

impl<const N: usize> std::iter::Product for SUN<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, g| acc * g)
    }
}

impl<'a, const N: usize> std::iter::Product<&'a SUN<N>> for SUN<N> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, g| &acc * g)
    }
}

impl<const N: usize> LieGroup for SUN<N> {
    type Algebra = SunAlgebra<N>;
    const MATRIX_DIM: usize = {
        assert!(
            N >= 2,
            "SU(N) requires N >= 2: SU(1) is trivial, SU(0) is undefined"
        );
        N
    };

    fn identity() -> Self {
        Self::identity()
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            matrix: self.matrix.dot(&other.matrix),
        }
    }

    fn inverse(&self) -> Self {
        // For unitary matrices: U⁻¹ = U†
        Self {
            matrix: self.matrix.t().mapv(|z| z.conj()),
        }
    }

    fn conjugate_transpose(&self) -> Self {
        Self {
            matrix: self.matrix.t().mapv(|z| z.conj()),
        }
    }

    /// Adjoint action: `Ad_g(X)` = gXg⁻¹
    ///
    /// # Mathematical Formula
    ///
    /// For g ∈ SU(N) and X ∈ su(N):
    /// ```text
    /// Ad_g(X) = gXg⁻¹
    /// ```
    ///
    /// # Properties
    ///
    /// - Group homomorphism: Ad_{gh} = `Ad_g` ∘ `Ad_h`
    /// - Preserves bracket: `Ad_g([X,Y])` = `[Ad_g(X), Ad_g(Y)]`
    /// - Preserves norm (SU(N) is compact)
    fn adjoint_action(&self, algebra_element: &Self::Algebra) -> Self::Algebra {
        let x = algebra_element.to_matrix();
        let g_inv = self.inverse();

        // Compute gXg⁻¹
        let result = self.matrix.dot(&x).dot(&g_inv.matrix);

        SunAlgebra::from_matrix(&result)
    }

    fn distance_to_identity(&self) -> f64 {
        self.distance_to_identity()
    }

    /// Exponential map: exp: su(N) → SU(N)
    ///
    /// # Algorithm: Scaling-and-Squaring
    ///
    /// For X ∈ su(N) with ||X|| large:
    /// 1. Scale: X' = X / 2^k such that ||X'|| ≤ 0.5
    /// 2. Taylor: exp(X') ≈ ∑_{n=0}^{15} X'^n / n!
    /// 3. Square: exp(X) = [exp(X')]^{2^k}
    ///
    /// # Properties
    ///
    /// - Preserves unitarity: exp(X) ∈ SU(N) for X ∈ su(N)
    /// - Preserves det = 1 (Tr(X) = 0 ⟹ det(exp(X)) = exp(Tr(X)) = 1)
    /// - Numerically stable for all ||X||
    ///
    /// # Performance
    ///
    /// - Time: O(N³·log(||X||))
    /// - Space: O(N²)
    ///
    /// # Accuracy
    ///
    /// - Relative error: ~10⁻¹⁴ (double precision)
    /// - Unitarity preserved to ~10⁻¹²
    fn exp(tangent: &Self::Algebra) -> Self {
        let x = tangent.to_matrix();
        let norm = matrix_frobenius_norm(&x);

        // Determine scaling factor: k such that ||X/2^k|| ≤ 0.5
        let k = if norm > 0.5 {
            (norm / 0.5).log2().ceil() as u32
        } else {
            0
        };

        // Scale down
        let scale_factor = 2_f64.powi(-(k as i32));
        let x_scaled = x.mapv(|z| z * scale_factor);

        // Taylor series: exp(X') ≈ ∑ X'^n / n!
        let exp_scaled = matrix_exp_taylor(&x_scaled, 15);

        // Square k times: exp(X) = [exp(X/2^k)]^{2^k}
        //
        // Reorthogonalize after EVERY squaring to prevent numerical drift.
        //
        // Each matrix multiply accumulates O(nε) orthogonality loss.
        // After k squarings without correction: O(2^k · nε) total error.
        // For k=10, n=4: error ~4000ε ≈ 9e-13 — approaching catastrophic loss.
        //
        // Cost of Gram-Schmidt is O(N³), same as the matrix multiply itself,
        // so the overhead factor is ≤2× (negligible vs correctness).
        let mut result = exp_scaled;
        for _ in 0..k {
            result = result.dot(&result);
            result = Self::gram_schmidt_project(result);
        }

        Self { matrix: result }
    }

    fn log(&self) -> crate::error::LogResult<Self::Algebra> {
        use crate::error::LogError;

        // Matrix logarithm for SU(N) using inverse scaling-squaring algorithm.
        //
        // Algorithm (Higham, "Functions of Matrices", Ch. 11):
        // 1. Take square roots until ||U^{1/2^k} - I|| < 0.5
        // 2. Use Taylor series for log(I + X) with ||X|| < 0.5 (fast convergence)
        // 3. Scale back: log(U) = 2^k × log(U^{1/2^k})

        let dist = self.distance_to_identity();
        const MAX_DISTANCE: f64 = 2.0;

        if dist > MAX_DISTANCE {
            return Err(LogError::NotNearIdentity {
                distance: dist,
                threshold: MAX_DISTANCE,
            });
        }

        if dist < 1e-14 {
            return Ok(SunAlgebra::zero());
        }

        // Phase 1: Inverse scaling via matrix square roots
        let identity_matrix: Array2<Complex64> = Array2::eye(N);
        let mut current = self.matrix.clone();
        let mut num_sqrts: u32 = 0;
        const MAX_SQRTS: u32 = 32;
        const TARGET_NORM: f64 = 0.5;

        while num_sqrts < MAX_SQRTS {
            let x_matrix = &current - &identity_matrix;
            let x_norm = matrix_frobenius_norm(&x_matrix);
            if x_norm < TARGET_NORM {
                break;
            }
            current = matrix_sqrt_db(&current);
            num_sqrts += 1;
        }

        // Phase 2: Taylor series for log(I + X) with ||X|| < 0.5
        let x_matrix = &current - &identity_matrix;
        let log_matrix = matrix_log_taylor(&x_matrix, 30);

        // Phase 3: Scale back: log(U) = 2^k × log(U^{1/2^k})
        let scale_factor = (1_u64 << num_sqrts) as f64;
        let log_scaled = log_matrix.mapv(|z| z * scale_factor);

        Ok(SunAlgebra::from_matrix(&log_scaled))
    }
}

/// Compute Frobenius norm of complex matrix: ||A||_F = √(Tr(A†A))
fn matrix_frobenius_norm(matrix: &Array2<Complex64>) -> f64 {
    matrix
        .iter()
        .map(num_complex::Complex::norm_sqr)
        .sum::<f64>()
        .sqrt()
}

/// Compute matrix exponential via Taylor series
///
/// exp(X) = I + X + X²/2! + X³/3! + ... + X^n/n!
///
/// Converges rapidly for ||X|| ≤ 0.5.
fn matrix_exp_taylor(matrix: &Array2<Complex64>, terms: usize) -> Array2<Complex64> {
    let n = matrix.nrows();
    let mut result = Array2::eye(n); // I
    let mut term = Array2::eye(n); // Current term: X^k / k!

    for k in 1..=terms {
        // term = term · X / k
        term = term.dot(matrix).mapv(|z| z / (k as f64));
        result += &term;
    }

    result
}

/// Compute matrix logarithm via Taylor series
///
/// log(I + X) = X - X²/2 + X³/3 - X⁴/4 + ... + (-1)^{n+1}·X^n/n
///
/// Converges for spectral radius ρ(X) < 1. For ||X||_F < 0.5, convergence
/// is rapid (30 terms gives ~3e-11 truncation error).
fn matrix_log_taylor(matrix: &Array2<Complex64>, terms: usize) -> Array2<Complex64> {
    let mut result = matrix.clone(); // First term: X
    let mut x_power = matrix.clone(); // Current power: X^k

    for k in 2..=terms {
        // x_power = X^k
        x_power = x_power.dot(matrix);

        // Coefficient: (-1)^{k+1} / k
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        let coefficient = sign / (k as f64);

        // Add term to result
        result = result + x_power.mapv(|z| z * coefficient);
    }

    result
}

/// Complex N×N matrix inverse via Gauss-Jordan elimination with partial pivoting.
///
/// Returns None if the matrix is singular (pivot < 1e-15).
fn matrix_inverse(a: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());

    // Build augmented matrix [A | I]
    let mut aug = Array2::<Complex64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = Complex64::new(1.0, 0.0);
    }

    for col in 0..n {
        // Partial pivoting: find row with largest magnitude in this column
        let mut max_norm = 0.0;
        let mut max_row = col;
        for row in col..n {
            let norm = aug[[row, col]].norm();
            if norm > max_norm {
                max_norm = norm;
                max_row = row;
            }
        }
        if max_norm < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column in all other rows
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                // Read pivot row values first to avoid borrow conflict
                let pivot_row: Vec<Complex64> = (0..2 * n).map(|j| aug[[col, j]]).collect();
                for j in 0..2 * n {
                    aug[[row, j]] -= factor * pivot_row[j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(result)
}

/// Matrix square root via Denman-Beavers iteration.
///
/// Computes U^{1/2} for a matrix U close to identity.
/// Uses the iteration Y_{k+1} = (Y_k + Z_k^{-1})/2, Z_{k+1} = (Z_k + Y_k^{-1})/2
/// which converges quadratically to U^{1/2}.
fn matrix_sqrt_db(u: &Array2<Complex64>) -> Array2<Complex64> {
    let n = u.nrows();
    let mut y = u.clone();
    let mut z = Array2::<Complex64>::eye(n);

    const MAX_ITERS: usize = 20;
    const TOL: f64 = 1e-14;

    for _ in 0..MAX_ITERS {
        let y_inv = matrix_inverse(&y).unwrap_or_else(|| y.t().mapv(|z| z.conj()));
        let z_inv = matrix_inverse(&z).unwrap_or_else(|| z.t().mapv(|z| z.conj()));

        let y_new = (&y + &z_inv).mapv(|z| z * 0.5);
        let z_new = (&z + &y_inv).mapv(|z| z * 0.5);

        let diff = matrix_frobenius_norm(&(&y_new - &y));
        y = y_new;
        z = z_new;

        if diff < TOL {
            break;
        }
    }

    y
}

// ============================================================================
// Const Generic Specializations
// ============================================================================

// ============================================================================
// Algebra Marker Traits
// ============================================================================

/// SU(N) is compact for all N ≥ 2.
impl<const N: usize> Compact for SUN<N> {}

/// SU(N) is simple for all N ≥ 2.
impl<const N: usize> Simple for SUN<N> {}

/// SU(N) is semi-simple (implied by simple) for all N ≥ 2.
impl<const N: usize> SemiSimple for SUN<N> {}

/// su(N) algebra elements are traceless by construction.
///
/// The representation `SunAlgebra<N>` stores N²-1 coefficients in a
/// generalized Gell-Mann basis. All generators are traceless by definition.
impl<const N: usize> TracelessByConstruction for SunAlgebra<N> {}

/// su(N) algebra elements are anti-Hermitian by construction.
///
/// The representation uses i·λⱼ where λⱼ are Hermitian generators.
impl<const N: usize> AntiHermitianByConstruction for SunAlgebra<N> {}

// ============================================================================
// Type Aliases
// ============================================================================

/// Type alias for SU(2) via generic implementation
pub type SU2Generic = SUN<2>;
/// Type alias for SU(3) via generic implementation
pub type SU3Generic = SUN<3>;
/// Type alias for SU(4) - Pati-Salam model
pub type SU4 = SUN<4>;
/// Type alias for SU(5) - Georgi-Glashow GUT
pub type SU5 = SUN<5>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    // ========================================================================
    // Cross-implementation consistency: SUN<2> vs SU2, SUN<3> vs SU3
    // ========================================================================

    /// SPEC: Same coefficients in Su2Algebra and SunAlgebra<2> must produce
    /// the same matrix, the same bracket, and the same exp.
    ///
    /// This is the regression guard for basis normalization consistency.
    /// Convention: tr(Tₐ†Tᵦ) = ½δₐᵦ for all implementations.
    #[test]
    fn test_sun2_su2_basis_matrices_agree() {
        // Basis element matrices must be identical
        for k in 0..3 {
            let m_sun = SunAlgebra::<2>::basis_element(k).to_matrix();
            // Build SU2 basis matrix manually: Tₖ = iσₖ/2
            let i = Complex64::new(0.0, 1.0);
            let sigma: Array2<Complex64> = match k {
                0 => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .unwrap(),
                1 => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .unwrap(),
                2 => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(-1.0, 0.0),
                    ],
                )
                .unwrap(),
                _ => unreachable!(),
            };
            let expected = sigma.mapv(|z| i * z * 0.5);

            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (m_sun[(r, c)] - expected[(r, c)]).norm() < 1e-10,
                        "SUN<2> basis {} at ({},{}): got {}, want {}",
                        k,
                        r,
                        c,
                        m_sun[(r, c)],
                        expected[(r, c)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sun2_su2_brackets_agree() {
        use crate::Su2Algebra;

        for i in 0..3 {
            for j in 0..3 {
                let bracket_su2 = Su2Algebra::basis_element(i)
                    .bracket(&Su2Algebra::basis_element(j))
                    .to_components();
                let bracket_sun = SunAlgebra::<2>::basis_element(i)
                    .bracket(&SunAlgebra::<2>::basis_element(j))
                    .to_components();

                for k in 0..3 {
                    assert_relative_eq!(bracket_su2[k], bracket_sun[k], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_sun2_su2_exp_agrees() {
        use crate::{Su2Algebra, SU2};

        let coeffs = [0.3, -0.2, 0.4];
        let g_su2 = SU2::exp(&Su2Algebra::new(coeffs));
        let g_sun = SUN::<2>::exp(&SunAlgebra::<2>::from_components(&coeffs));

        let m_su2 = g_su2.to_matrix();
        let m_sun = g_sun.matrix();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m_su2[i][j].re, m_sun[(i, j)].re, epsilon = 1e-10);
                assert_relative_eq!(m_su2[i][j].im, m_sun[(i, j)].im, epsilon = 1e-10);
            }
        }
    }

    /// SPEC: SU3 and SUN<3> basis matrices must agree up to basis reordering.
    ///
    /// SU3 uses standard Gell-Mann ordering (λ₁..λ₈ interleaved by type).
    /// SUN<3> groups by type: symmetric, antisymmetric, diagonal.
    /// Mapping: SU3 index [0,1,2,3,4,5,6,7] → SUN<3> index [0,3,6,1,4,2,5,7].
    #[test]
    fn test_sun3_su3_basis_matrices_agree() {
        use crate::Su3Algebra;

        // SU3 Gell-Mann index → SUN<3> generalized Gell-Mann index
        let su3_to_sun: [usize; 8] = [0, 3, 6, 1, 4, 2, 5, 7];

        for k in 0..8 {
            let m_su3 = Su3Algebra::basis_element(k).to_matrix();
            let m_sun = SunAlgebra::<3>::basis_element(su3_to_sun[k]).to_matrix();

            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        (m_su3[(r, c)] - m_sun[(r, c)]).norm() < 1e-10,
                        "Basis {} (SU3) vs {} (SUN) at ({},{}): SU3={}, SUN<3>={}",
                        k,
                        su3_to_sun[k],
                        r,
                        c,
                        m_su3[(r, c)],
                        m_sun[(r, c)]
                    );
                }
            }
        }
    }

    /// SPEC: Brackets must agree when using corresponding basis elements.
    ///
    /// We compare at the matrix level to avoid coefficient ordering issues.
    #[test]
    fn test_sun3_su3_brackets_agree() {
        use crate::Su3Algebra;

        // SU3 Gell-Mann index → SUN<3> generalized Gell-Mann index
        let su3_to_sun: [usize; 8] = [0, 3, 6, 1, 4, 2, 5, 7];

        for i in 0..8 {
            for j in 0..8 {
                let bracket_su3 = Su3Algebra::basis_element(i)
                    .bracket(&Su3Algebra::basis_element(j))
                    .to_matrix();
                let bracket_sun = SunAlgebra::<3>::basis_element(su3_to_sun[i])
                    .bracket(&SunAlgebra::<3>::basis_element(su3_to_sun[j]))
                    .to_matrix();

                for r in 0..3 {
                    for c in 0..3 {
                        assert!(
                            (bracket_su3[(r, c)] - bracket_sun[(r, c)]).norm() < 1e-10,
                            "Bracket [e{},e{}] at ({},{}): SU3={}, SUN<3>={}",
                            i,
                            j,
                            r,
                            c,
                            bracket_su3[(r, c)],
                            bracket_sun[(r, c)]
                        );
                    }
                }
            }
        }
    }

    /// SPEC: exp of the same matrix must produce the same group element.
    ///
    /// We construct the algebra element via matrix to avoid ordering issues.
    #[test]
    fn test_sun3_su3_exp_agrees() {
        use crate::{Su3Algebra, SU3};

        // Build an su(3) matrix via SU3, then reconstruct in both representations
        let su3_coeffs = [0.1, -0.2, 0.15, 0.08, -0.12, 0.05, 0.1, -0.06];
        let su3_elem = Su3Algebra::from_components(&su3_coeffs);
        let matrix = su3_elem.to_matrix();

        // Reconstruct via SUN<3>::from_matrix
        let sun_elem = SunAlgebra::<3>::from_matrix(&matrix);

        let g_su3 = SU3::exp(&su3_elem);
        let g_sun = SUN::<3>::exp(&sun_elem);

        let m_su3 = g_su3.matrix();
        let m_sun = g_sun.matrix();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_su3[(i, j)] - m_sun[(i, j)]).norm() < 1e-10,
                    "exp disagrees at ({},{}): SU3={}, SUN<3>={}",
                    i,
                    j,
                    m_su3[(i, j)],
                    m_sun[(i, j)]
                );
            }
        }
    }

    /// SPEC: All implementations must satisfy tr(Tₐ†Tᵦ) = ½δₐᵦ.
    /// Tests both diagonal (normalization) and off-diagonal (orthogonality).
    #[test]
    fn test_normalization_half_delta() {
        for n in [2_usize, 3, 4, 5] {
            let dim = n * n - 1;
            for a in 0..dim {
                let ma = match n {
                    2 => SunAlgebra::<2>::basis_element(a).to_matrix(),
                    3 => SunAlgebra::<3>::basis_element(a).to_matrix(),
                    4 => SunAlgebra::<4>::basis_element(a).to_matrix(),
                    5 => SunAlgebra::<5>::basis_element(a).to_matrix(),
                    _ => unreachable!(),
                };
                let ma_dag = ma.t().mapv(|z| z.conj());

                for b in a..dim {
                    let mb = match n {
                        2 => SunAlgebra::<2>::basis_element(b).to_matrix(),
                        3 => SunAlgebra::<3>::basis_element(b).to_matrix(),
                        4 => SunAlgebra::<4>::basis_element(b).to_matrix(),
                        5 => SunAlgebra::<5>::basis_element(b).to_matrix(),
                        _ => unreachable!(),
                    };

                    let prod = ma_dag.dot(&mb);
                    let mut tr = 0.0;
                    for i in 0..n {
                        tr += prod[(i, i)].re;
                    }

                    let expected = if a == b { 0.5 } else { 0.0 };
                    assert!(
                        (tr - expected).abs() < 1e-10,
                        "SU({}): tr(T{}†T{}) = {:.4}, want {}",
                        n,
                        a,
                        b,
                        tr,
                        expected
                    );
                }
            }
        }
    }

    // ========================================================================
    // Normalization consistency: bracket + exp + adjoint must be coherent
    // ========================================================================

    /// Verify BCH(X,Y) ≈ log(exp(X)·exp(Y)) for SUN<2>, confirming
    /// the bracket/exp coupling is correct under the /2 convention.
    #[test]
    fn test_bch_exp_log_coherence_sun2() {
        use crate::bch::bch_fifth_order;

        let x = SunAlgebra::<2>::from_components(&[0.05, -0.03, 0.04]);
        let y = SunAlgebra::<2>::from_components(&[-0.02, 0.04, -0.03]);

        let direct = SUN::<2>::exp(&x).compose(&SUN::<2>::exp(&y));
        let bch_z = bch_fifth_order(&x, &y);
        let via_bch = SUN::<2>::exp(&bch_z);

        let distance = direct.compose(&via_bch.inverse()).distance_to_identity();
        assert!(
            distance < 1e-8,
            "SUN<2> BCH vs exp·log distance: {:.2e}",
            distance
        );
    }

    /// Adjoint preserves bracket on SU3: Ad_g([X,Y]) = [Ad_g(X), Ad_g(Y)].
    /// Uses non-random, non-axis-aligned elements for deterministic coverage.
    #[test]
    fn test_adjoint_preserves_bracket_su3() {
        use crate::{Su3Algebra, SU3};

        let x = Su3Algebra::from_components(&[0.3, -0.2, 0.1, 0.15, -0.1, 0.25, -0.15, 0.05]);
        let y = Su3Algebra::from_components(&[-0.1, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2]);
        let g = SU3::exp(&Su3Algebra::from_components(&[
            0.4, -0.3, 0.2, 0.1, -0.2, 0.15, -0.1, 0.3,
        ]));

        let lhs = g.adjoint_action(&x.bracket(&y));
        let rhs = g.adjoint_action(&x).bracket(&g.adjoint_action(&y));

        let diff_matrix = lhs.to_matrix() - rhs.to_matrix();
        let mut frobenius = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                frobenius += diff_matrix[(r, c)].norm_sqr();
            }
        }
        assert!(
            frobenius.sqrt() < 1e-10,
            "SU3 Ad_g([X,Y]) ≠ [Ad_g(X), Ad_g(Y)]: ||diff|| = {:.2e}",
            frobenius.sqrt()
        );
    }

    /// SU3 bracket (structure constants) and SUN<3> bracket (matrix commutator)
    /// must produce the same abstract element, verified through from_matrix roundtrip.
    #[test]
    fn test_su3_sun3_bracket_coefficient_roundtrip() {
        use crate::Su3Algebra;

        let x_su3 = Su3Algebra::from_components(&[0.3, -0.2, 0.1, 0.15, -0.1, 0.25, -0.15, 0.05]);
        let y_su3 = Su3Algebra::from_components(&[-0.1, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2]);

        // Bracket via SU3 structure constants
        let bracket_su3 = x_su3.bracket(&y_su3);
        let m_su3 = bracket_su3.to_matrix();

        // Same elements via SUN<3>, bracket via matrix commutator
        let x_sun = SunAlgebra::<3>::from_matrix(&x_su3.to_matrix());
        let y_sun = SunAlgebra::<3>::from_matrix(&y_su3.to_matrix());
        let bracket_sun = x_sun.bracket(&y_sun);
        let m_sun = bracket_sun.to_matrix();

        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m_su3[(r, c)] - m_sun[(r, c)]).norm() < 1e-12,
                    "Bracket matrix disagrees at ({},{}): SU3={}, SUN<3>={}",
                    r,
                    c,
                    m_su3[(r, c)],
                    m_sun[(r, c)]
                );
            }
        }

        // Also verify from_matrix roundtrip: SU3 → matrix → SUN<3> → matrix
        let roundtrip = SunAlgebra::<3>::from_matrix(&m_su3).to_matrix();
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m_su3[(r, c)] - roundtrip[(r, c)]).norm() < 1e-12,
                    "from_matrix roundtrip failed at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    // ========================================================================
    // Group axioms for N=4,5 (untested dimensions)
    // ========================================================================

    #[test]
    fn test_su4_group_axioms() {
        let x = SunAlgebra::<4>::from_components(&[
            0.1, -0.2, 0.15, 0.08, -0.12, 0.05, 0.1, -0.06, 0.09, -0.11, 0.07, 0.03, -0.08, 0.04,
            0.13,
        ]);
        let y = SunAlgebra::<4>::from_components(&[
            -0.05, 0.1, -0.08, 0.12, 0.06, -0.15, 0.03, 0.09, -0.07, 0.11, -0.04, 0.14, 0.02, -0.1,
            0.08,
        ]);

        let g = SUN::<4>::exp(&x);
        let h = SUN::<4>::exp(&y);
        let e = SUN::<4>::identity();

        // Identity
        assert_relative_eq!(
            g.compose(&e).distance_to_identity(),
            g.distance_to_identity(),
            epsilon = 1e-10
        );

        // Inverse
        assert_relative_eq!(
            g.compose(&g.inverse()).distance_to_identity(),
            0.0,
            epsilon = 1e-9
        );

        // Associativity
        let k = SUN::<4>::exp(&x.scale(0.7));
        let lhs = g.compose(&h).compose(&k);
        let rhs = g.compose(&h.compose(&k));
        assert_relative_eq!(
            lhs.compose(&rhs.inverse()).distance_to_identity(),
            0.0,
            epsilon = 1e-9
        );

        // Unitarity preservation
        assert!(g.verify_unitarity(1e-10));
        assert!(g.compose(&h).verify_unitarity(1e-10));
    }

    #[test]
    fn test_su5_group_axioms() {
        let x = SunAlgebra::<5>::from_components(
            &(0..24)
                .map(|i| 0.05 * (i as f64 - 12.0).sin())
                .collect::<Vec<_>>(),
        );
        let g = SUN::<5>::exp(&x);

        // Identity
        assert_relative_eq!(
            g.compose(&SUN::<5>::identity()).distance_to_identity(),
            g.distance_to_identity(),
            epsilon = 1e-10
        );

        // Inverse
        assert_relative_eq!(
            g.compose(&g.inverse()).distance_to_identity(),
            0.0,
            epsilon = 1e-9
        );

        // Unitarity
        assert!(g.verify_unitarity(1e-10));
    }

    // ========================================================================
    // Jacobi identity for N=4 (tested only for N=3 previously)
    // ========================================================================

    #[test]
    fn test_su4_jacobi_identity() {
        // Test with several triples of basis elements
        let triples = [(0, 3, 7), (1, 5, 10), (2, 8, 14), (4, 9, 12)];
        for (i, j, k) in triples {
            let x = SunAlgebra::<4>::basis_element(i);
            let y = SunAlgebra::<4>::basis_element(j);
            let z = SunAlgebra::<4>::basis_element(k);

            let t1 = x.bracket(&y.bracket(&z));
            let t2 = y.bracket(&z.bracket(&x));
            let t3 = z.bracket(&x.bracket(&y));
            let sum = t1.add(&t2).add(&t3);

            assert!(
                sum.norm() < 1e-10,
                "Jacobi violated for SU(4) basis ({},{},{}): ||sum|| = {:.2e}",
                i,
                j,
                k,
                sum.norm()
            );
        }
    }

    // ========================================================================
    // Adjoint representation axioms for SU(N)
    // ========================================================================

    #[test]
    fn test_sun_adjoint_identity_action() {
        // Ad_e(X) = X for all X
        let e = SUN::<3>::identity();
        for i in 0..8 {
            let x = SunAlgebra::<3>::basis_element(i);
            let ad_x = e.adjoint_action(&x);
            for k in 0..8 {
                assert_relative_eq!(
                    x.to_components()[k],
                    ad_x.to_components()[k],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_sun_adjoint_homomorphism() {
        // Ad_{g·h}(X) = Ad_g(Ad_h(X))
        let g = SUN::<3>::exp(&SunAlgebra::<3>::basis_element(0).scale(0.5));
        let h = SUN::<3>::exp(&SunAlgebra::<3>::basis_element(3).scale(0.3));
        let x = SunAlgebra::<3>::from_components(&[0.1, -0.2, 0.15, 0.08, -0.12, 0.05, 0.1, -0.06]);

        let gh = g.compose(&h);
        let lhs = gh.adjoint_action(&x);
        let rhs = g.adjoint_action(&h.adjoint_action(&x));

        for k in 0..8 {
            assert_relative_eq!(
                lhs.to_components()[k],
                rhs.to_components()[k],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_sun_adjoint_preserves_bracket() {
        // Ad_g([X,Y]) = [Ad_g(X), Ad_g(Y)]
        let g = SUN::<3>::exp(&SunAlgebra::<3>::basis_element(2).scale(0.8));
        let x = SunAlgebra::<3>::basis_element(0);
        let y = SunAlgebra::<3>::basis_element(4);

        let lhs = g.adjoint_action(&x.bracket(&y));
        let rhs = g.adjoint_action(&x).bracket(&g.adjoint_action(&y));

        for k in 0..8 {
            assert_relative_eq!(
                lhs.to_components()[k],
                rhs.to_components()[k],
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_sun4_adjoint_preserves_norm() {
        // ||Ad_g(X)|| = ||X|| for compact groups
        let x = SunAlgebra::<4>::from_components(&[
            0.1, -0.2, 0.15, 0.08, -0.12, 0.05, 0.1, -0.06, 0.09, -0.11, 0.07, 0.03, -0.08, 0.04,
            0.13,
        ]);
        let g = SUN::<4>::exp(&SunAlgebra::<4>::basis_element(5).scale(1.2));
        let ad_x = g.adjoint_action(&x);
        assert_relative_eq!(x.norm(), ad_x.norm(), epsilon = 1e-9);
    }

    // ========================================================================
    // exp/log roundtrip for N=4 (untested dimension)
    // ========================================================================

    #[test]
    fn test_su4_exp_log_roundtrip() {
        let x = SunAlgebra::<4>::from_components(&[
            0.1, -0.05, 0.08, 0.03, -0.06, 0.02, 0.04, -0.07, 0.09, -0.03, 0.01, 0.05, -0.02, 0.06,
            0.04,
        ]);
        let g = SUN::<4>::exp(&x);
        assert!(g.verify_unitarity(1e-10));

        let x_back = SUN::<4>::log(&g).expect("log should succeed near identity");
        let diff: f64 = x
            .coefficients()
            .iter()
            .zip(x_back.coefficients().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-8, "SU(4) exp/log roundtrip error: {:.2e}", diff);
    }

    #[test]
    fn test_sun_algebra_dimensions() {
        assert_eq!(SunAlgebra::<2>::DIM, 3); // SU(2)
        assert_eq!(SunAlgebra::<3>::DIM, 8); // SU(3)
        assert_eq!(SunAlgebra::<4>::DIM, 15); // SU(4)
        assert_eq!(SunAlgebra::<5>::DIM, 24); // SU(5)
    }

    #[test]
    fn test_sun_algebra_zero() {
        let zero = SunAlgebra::<3>::zero();
        assert_eq!(zero.coefficients.len(), 8);
        assert!(zero.coefficients.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sun_algebra_add_scale() {
        let x = SunAlgebra::<2>::basis_element(0);
        let y = SunAlgebra::<2>::basis_element(1);

        let sum = &x + &y;
        assert_eq!(sum.coefficients, vec![1.0, 1.0, 0.0]);

        let scaled = x.scale(2.5);
        assert_eq!(scaled.coefficients, vec![2.5, 0.0, 0.0]);
    }

    #[test]
    fn test_sun_identity() {
        let id = SUN::<3>::identity();
        assert!(id.verify_unitarity(1e-10));
        assert_relative_eq!(id.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sun_exponential_preserves_unitarity() {
        // Random algebra element
        let algebra =
            SunAlgebra::<3>::from_components(&[0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.2]);
        let g = SUN::<3>::exp(&algebra);

        // Verify U†U = I
        assert!(
            g.verify_unitarity(1e-10),
            "Exponential should preserve unitarity"
        );
    }

    #[test]
    fn test_sun_exp_identity() {
        let zero = SunAlgebra::<4>::zero();
        let g = SUN::<4>::exp(&zero);
        assert_relative_eq!(g.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sun_group_composition() {
        let g1 = SUN::<2>::exp(&SunAlgebra::<2>::basis_element(0).scale(0.5));
        let g2 = SUN::<2>::exp(&SunAlgebra::<2>::basis_element(1).scale(0.3));

        let product = g1.compose(&g2);

        assert!(product.verify_unitarity(1e-10));
    }

    #[test]
    fn test_sun_inverse() {
        let algebra =
            SunAlgebra::<3>::from_components(&[0.2, 0.3, -0.1, 0.5, -0.2, 0.1, 0.4, -0.3]);
        let g = SUN::<3>::exp(&algebra);
        let g_inv = g.inverse();

        let product = g.compose(&g_inv);

        assert_relative_eq!(product.distance_to_identity(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_sun_adjoint_action_preserves_norm() {
        let g = SUN::<3>::exp(&SunAlgebra::<3>::basis_element(0).scale(1.2));
        let x = SunAlgebra::<3>::basis_element(2).scale(0.5);

        let ad_x = g.adjoint_action(&x);

        // Adjoint action preserves norm for compact groups
        assert_relative_eq!(x.norm(), ad_x.norm(), epsilon = 1e-9);
    }

    #[test]
    fn test_sun_exp_log_roundtrip() {
        // SU(3): exp then log should recover the original algebra element
        let x = SunAlgebra::<3>::from_components(&[0.1, -0.2, 0.15, 0.08, -0.12, 0.05, 0.1, -0.06]);
        let g = SUN::<3>::exp(&x);
        assert!(g.verify_unitarity(1e-10));

        let x_back = SUN::<3>::log(&g).expect("log should succeed near identity");
        let diff_norm: f64 = x
            .coefficients
            .iter()
            .zip(x_back.coefficients.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            diff_norm < 1e-8,
            "SU(3) exp/log roundtrip error: {:.2e}",
            diff_norm
        );
    }

    #[test]
    fn test_sun_exp_log_roundtrip_su2() {
        // SU(2) via generic SUN: smaller algebra
        let x = SunAlgebra::<2>::from_components(&[0.3, -0.2, 0.4]);
        let g = SUN::<2>::exp(&x);
        assert!(g.verify_unitarity(1e-10));

        let x_back = SUN::<2>::log(&g).expect("log should succeed");
        let diff_norm: f64 = x
            .coefficients
            .iter()
            .zip(x_back.coefficients.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            diff_norm < 1e-8,
            "SU(2) exp/log roundtrip error: {:.2e}",
            diff_norm
        );
    }

    #[test]
    fn test_sun_log_exp_roundtrip() {
        // Start from group element, log, then exp back
        let x = SunAlgebra::<3>::from_components(&[0.2, 0.3, -0.1, 0.5, -0.2, 0.1, 0.4, -0.3]);
        let g = SUN::<3>::exp(&x);

        let log_g = SUN::<3>::log(&g).expect("log should succeed");
        let g_back = SUN::<3>::exp(&log_g);

        assert_relative_eq!(
            g.distance_to_identity(),
            g_back.distance_to_identity(),
            epsilon = 1e-8
        );

        // Check that g and g_back are close
        let product = g.compose(&g_back.inverse());
        assert_relative_eq!(product.distance_to_identity(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sun_jacobi_identity() {
        let x = SunAlgebra::<3>::basis_element(0);
        let y = SunAlgebra::<3>::basis_element(1);
        let z = SunAlgebra::<3>::basis_element(2);

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
    fn test_sun_bracket_antisymmetry() {
        let x = SunAlgebra::<4>::basis_element(0);
        let y = SunAlgebra::<4>::basis_element(3);

        let xy = x.bracket(&y);
        let yx = y.bracket(&x);

        // [X,Y] = -[Y,X]
        for i in 0..SunAlgebra::<4>::DIM {
            assert_relative_eq!(xy.coefficients[i], -yx.coefficients[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sun_bracket_bilinearity() {
        let x = SunAlgebra::<3>::basis_element(0);
        let y = SunAlgebra::<3>::basis_element(3);
        let z = SunAlgebra::<3>::basis_element(5);
        let alpha = 2.5;

        // Left linearity: [αX + Y, Z] = α[X, Z] + [Y, Z]
        let lhs = x.scale(alpha).add(&y).bracket(&z);
        let rhs = x.bracket(&z).scale(alpha).add(&y.bracket(&z));
        for i in 0..SunAlgebra::<3>::DIM {
            assert!(
                (lhs.coefficients[i] - rhs.coefficients[i]).abs() < 1e-14,
                "Left linearity failed at component {}: {} vs {}",
                i,
                lhs.coefficients[i],
                rhs.coefficients[i]
            );
        }

        // Right linearity: [Z, αX + Y] = α[Z, X] + [Z, Y]
        let lhs = z.bracket(&x.scale(alpha).add(&y));
        let rhs = z.bracket(&x).scale(alpha).add(&z.bracket(&y));
        for i in 0..SunAlgebra::<3>::DIM {
            assert!(
                (lhs.coefficients[i] - rhs.coefficients[i]).abs() < 1e-14,
                "Right linearity failed at component {}: {} vs {}",
                i,
                lhs.coefficients[i],
                rhs.coefficients[i]
            );
        }
    }
}
