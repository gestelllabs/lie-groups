//! Lie group SO(3) - 3D rotation group
//!
//! SO(3) is the group of 3×3 real orthogonal matrices with determinant 1.
//! It represents rotations in 3-dimensional Euclidean space.
//!
//! # Mathematical Structure
//!
//! ```text
//! SO(3) = { R ∈ ℝ³ˣ³ | R^T R = I, det(R) = 1 }
//! ```
//!
//! # Lie Algebra
//!
//! The Lie algebra so(3) consists of 3×3 real antisymmetric matrices:
//! ```text
//! so(3) = { X ∈ ℝ³ˣ³ | X^T = -X }
//! ```
//!
//! This is 3-dimensional, naturally isomorphic to ℝ³ via the cross product:
//! ```text
//! X ↔ v  where  X·w = v × w  for all w ∈ ℝ³
//! ```
//!
//! # Relationship to SU(2)
//!
//! SO(3) ≅ SU(2)/ℤ₂ — SU(2) is the double cover of SO(3).
//! Both have 3-dimensional Lie algebras: so(3) ≅ su(2) ≅ ℝ³
//!
//! # Sign Convention
//!
//! The so(3) bracket is the cross product: `[v, w] = v × w`, giving
//! structure constants `fᵢⱼₖ = +εᵢⱼₖ`. The su(2) bracket (in the `{iσ/2}`
//! basis) uses `[X, Y] = -(X × Y)`, giving `fᵢⱼₖ = -εᵢⱼₖ`.
//!
//! The bracket-preserving isomorphism φ: su(2) → so(3) maps `eₐ ↦ -Lₐ`,
//! so that `φ([eₐ, eᵦ]) = [φ(eₐ), φ(eᵦ)]`. This sign arises from the
//! standard `{iσ/2}` basis choice for su(2).

use crate::traits::{AntiHermitianByConstruction, LieAlgebra, LieGroup, TracelessByConstruction};
use nalgebra::Matrix3;
use std::fmt;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

/// Lie algebra so(3) ≅ ℝ³
///
/// Elements of so(3) are 3×3 real antisymmetric matrices, which we represent
/// as 3-vectors via the natural isomorphism with ℝ³.
///
/// # Isomorphism with ℝ³
///
/// An element v = (x, y, z) ∈ ℝ³ corresponds to the antisymmetric matrix:
/// ```text
/// [v]_× = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
/// ```
///
/// This satisfies: `[v]_×` · w = v × w (cross product)
///
/// # Basis Elements
///
/// The standard basis corresponds to angular momentum operators:
/// ```text
/// L_x = (1, 0, 0) ↔ [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
/// L_y = (0, 1, 0) ↔ [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
/// L_z = (0, 0, 1) ↔ [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct So3Algebra(pub(crate) [f64; 3]);

impl So3Algebra {
    /// Create a new so(3) algebra element from components.
    ///
    /// The components `[x, y, z]` correspond to the antisymmetric matrix
    /// `[[0, -z, y], [z, 0, -x], [-y, x, 0]]`.
    #[must_use]
    pub fn new(components: [f64; 3]) -> Self {
        Self(components)
    }

    /// Returns the components as a fixed-size array reference.
    #[must_use]
    pub fn components(&self) -> &[f64; 3] {
        &self.0
    }
}

impl Add for So3Algebra {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Add<&So3Algebra> for So3Algebra {
    type Output = So3Algebra;
    fn add(self, rhs: &So3Algebra) -> So3Algebra {
        self + *rhs
    }
}

impl Add<So3Algebra> for &So3Algebra {
    type Output = So3Algebra;
    fn add(self, rhs: So3Algebra) -> So3Algebra {
        *self + rhs
    }
}

impl Add<&So3Algebra> for &So3Algebra {
    type Output = So3Algebra;
    fn add(self, rhs: &So3Algebra) -> So3Algebra {
        *self + *rhs
    }
}

impl Sub for So3Algebra {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl Neg for So3Algebra {
    type Output = Self;
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Mul<f64> for So3Algebra {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self([self.0[0] * scalar, self.0[1] * scalar, self.0[2] * scalar])
    }
}

impl Mul<So3Algebra> for f64 {
    type Output = So3Algebra;
    fn mul(self, rhs: So3Algebra) -> So3Algebra {
        rhs * self
    }
}

impl LieAlgebra for So3Algebra {
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
        assert!(i < 3, "SO(3) algebra is 3-dimensional");
        let mut v = [0.0; 3];
        v[i] = 1.0;
        Self(v)
    }

    #[inline]
    fn from_components(components: &[f64]) -> Self {
        assert_eq!(components.len(), 3, "so(3) has dimension 3");
        Self([components[0], components[1], components[2]])
    }

    #[inline]
    fn to_components(&self) -> Vec<f64> {
        self.0.to_vec()
    }

    /// Lie bracket for so(3): [v, w] = v × w (cross product)
    ///
    /// For so(3) ≅ ℝ³, the Lie bracket is exactly the vector cross product.
    ///
    /// # Properties
    ///
    /// - Antisymmetric: [v, w] = -[w, v]
    /// - Jacobi identity: [u, [v, w]] + [v, [w, u]] + [w, [u, v]] = 0
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::so3::So3Algebra;
    /// use lie_groups::traits::LieAlgebra;
    ///
    /// let lx = So3Algebra::basis_element(0);  // (1, 0, 0)
    /// let ly = So3Algebra::basis_element(1);  // (0, 1, 0)
    /// let bracket = lx.bracket(&ly);           // (1,0,0) × (0,1,0) = (0,0,1)
    ///
    /// // Should give L_z = (0, 0, 1)
    /// assert!((bracket.components()[0]).abs() < 1e-10);
    /// assert!((bracket.components()[1]).abs() < 1e-10);
    /// assert!((bracket.components()[2] - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    fn bracket(&self, other: &Self) -> Self {
        // Cross product: v × w
        let v = self.0;
        let w = other.0;

        Self([
            v[1] * w[2] - v[2] * w[1], // x component
            v[2] * w[0] - v[0] * w[2], // y component
            v[0] * w[1] - v[1] * w[0], // z component
        ])
    }

    #[inline]
    fn inner(&self, other: &Self) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }
}

/// SO(3) group element - 3×3 real orthogonal matrix with determinant 1
///
/// Represents a rotation in 3D space.
///
/// # Representation
///
/// We use `Matrix3<f64>` from nalgebra to represent the 3×3 rotation matrix.
///
/// # Constraints
///
/// - Orthogonality: R^T R = I
/// - Determinant: det(R) = 1
///
/// # Examples
///
/// ```
/// use lie_groups::so3::SO3;
/// use lie_groups::traits::LieGroup;
///
/// // Rotation around Z-axis by π/2
/// let rot = SO3::rotation_z(std::f64::consts::FRAC_PI_2);
///
/// // Verify it's orthogonal
/// assert!(rot.verify_orthogonality(1e-10));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SO3 {
    /// 3×3 real orthogonal matrix
    pub(crate) matrix: Matrix3<f64>,
}

impl SO3 {
    /// Access the underlying 3×3 orthogonal matrix
    #[must_use]
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    /// Identity element (no rotation)
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Rotation around X-axis by angle θ (in radians)
    ///
    /// ```text
    /// R_x(θ) = [[1, 0, 0], [0, cos(θ), -sin(θ)], [0, sin(θ), cos(θ)]]
    /// ```
    #[must_use]
    pub fn rotation_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();

        Self {
            matrix: Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c),
        }
    }

    /// Rotation around Y-axis by angle θ (in radians)
    ///
    /// ```text
    /// R_y(θ) = [[cos(θ), 0, sin(θ)], [0, 1, 0], [-sin(θ), 0, cos(θ)]]
    /// ```
    #[must_use]
    pub fn rotation_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();

        Self {
            matrix: Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c),
        }
    }

    /// Rotation around Z-axis by angle θ (in radians)
    ///
    /// ```text
    /// R_z(θ) = [[cos(θ), -sin(θ), 0], [sin(θ), cos(θ), 0], [0, 0, 1]]
    /// ```
    #[must_use]
    pub fn rotation_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();

        Self {
            matrix: Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Rotation around arbitrary axis by angle
    ///
    /// Uses Rodrigues' rotation formula:
    /// ```text
    /// R(θ, n̂) = I + sin(θ)[n̂]_× + (1-cos(θ))[n̂]_×²
    /// ```
    ///
    /// where [n̂]_× is the skew-symmetric matrix for axis n̂.
    #[must_use]
    pub fn rotation(axis: [f64; 3], angle: f64) -> Self {
        let norm = (axis[0].powi(2) + axis[1].powi(2) + axis[2].powi(2)).sqrt();
        if norm < 1e-10 {
            return Self::identity();
        }

        // Normalize axis
        let n = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

        // Rodrigues formula
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;

        let matrix = Matrix3::new(
            t * n[0] * n[0] + c,
            t * n[0] * n[1] - s * n[2],
            t * n[0] * n[2] + s * n[1],
            t * n[0] * n[1] + s * n[2],
            t * n[1] * n[1] + c,
            t * n[1] * n[2] - s * n[0],
            t * n[0] * n[2] - s * n[1],
            t * n[1] * n[2] + s * n[0],
            t * n[2] * n[2] + c,
        );

        Self { matrix }
    }

    /// Trace of the rotation matrix: Tr(R) = 1 + 2cos(θ)
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.matrix.trace()
    }

    /// Convert to 3×3 array format
    #[must_use]
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        let m = &self.matrix;
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ]
    }

    /// Create from 3×3 array format
    ///
    /// # Panics
    ///
    /// Does not verify orthogonality. Use `verify_orthogonality()` after construction.
    #[must_use]
    pub fn from_matrix(arr: [[f64; 3]; 3]) -> Self {
        Self {
            matrix: Matrix3::new(
                arr[0][0], arr[0][1], arr[0][2], arr[1][0], arr[1][1], arr[1][2], arr[2][0],
                arr[2][1], arr[2][2],
            ),
        }
    }

    /// Verify orthogonality: R^T R = I
    #[must_use]
    pub fn verify_orthogonality(&self, tolerance: f64) -> bool {
        let product = self.matrix.transpose() * self.matrix;
        let identity = Matrix3::identity();

        (product - identity).norm() < tolerance
    }

    /// Matrix inverse (equals transpose for orthogonal matrices)
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.transpose(),
        }
    }

    /// Distance from identity (rotation angle)
    ///
    /// For a rotation matrix R, the angle θ satisfies:
    /// ```text
    /// trace(R) = 1 + 2cos(θ)
    /// ```
    #[must_use]
    pub fn distance_to_identity(&self) -> f64 {
        let trace = self.matrix.trace();
        let cos_theta = (trace - 1.0) / 2.0;

        // Clamp to [-1, 1] to handle numerical errors
        let cos_theta = cos_theta.clamp(-1.0, 1.0);

        cos_theta.acos()
    }

    /// Interpolate between two SO(3) elements with proper orthogonalization
    ///
    /// Uses linear interpolation of matrix elements followed by Gram-Schmidt
    /// orthogonalization to ensure the result stays on SO(3).
    ///
    /// # Arguments
    /// * `other` - The target rotation
    /// * `t` - Interpolation parameter in [0, 1]
    ///
    /// # Returns
    /// An SO(3) element: self at t=0, other at t=1
    ///
    /// # Note
    /// This is NOT geodesic interpolation (SLERP). For true geodesic paths,
    /// convert to quaternions and use quaternion SLERP. However, this method
    /// guarantees the result is always a valid rotation matrix.
    #[must_use]
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }

        // Linear interpolation of matrix elements
        let interpolated = self.matrix * (1.0 - t) + other.matrix * t;

        // Gram-Schmidt orthogonalization to project back onto SO(3)
        Self::gram_schmidt_orthogonalize(interpolated)
    }

    /// Orthogonalize a matrix using Gram-Schmidt process
    ///
    /// Takes a near-orthogonal matrix and projects it back onto SO(3).
    /// This is essential for numerical stability when matrices drift
    /// from orthogonality due to floating-point accumulation.
    ///
    /// # Algorithm
    /// 1. Normalize first column
    /// 2. Orthogonalize and normalize second column
    /// 3. Compute third column as cross product (ensures determinant = 1)
    #[must_use]
    pub fn gram_schmidt_orthogonalize(matrix: Matrix3<f64>) -> Self {
        use nalgebra::Vector3;

        // Extract columns
        let c0 = Vector3::new(matrix[(0, 0)], matrix[(1, 0)], matrix[(2, 0)]);
        let c1 = Vector3::new(matrix[(0, 1)], matrix[(1, 1)], matrix[(2, 1)]);

        // Gram-Schmidt orthogonalization
        let e0 = c0.normalize();
        let e1 = (c1 - e0 * e0.dot(&c1)).normalize();
        // Third column is cross product to ensure det = +1 (proper rotation)
        let e2 = e0.cross(&e1);

        let orthogonal = Matrix3::new(
            e0[0], e1[0], e2[0], e0[1], e1[1], e2[1], e0[2], e1[2], e2[2],
        );

        Self { matrix: orthogonal }
    }

    /// Re-normalize a rotation matrix that may have drifted
    ///
    /// Use this periodically after many matrix multiplications to
    /// prevent numerical drift from accumulating.
    #[must_use]
    pub fn renormalize(&self) -> Self {
        Self::gram_schmidt_orthogonalize(self.matrix)
    }

    /// Geodesic distance between two SO(3) elements
    ///
    /// d(R₁, R₂) = ||log(R₁ᵀ R₂)||
    ///
    /// This is the rotation angle of the relative rotation R₁ᵀ R₂.
    #[must_use]
    pub fn geodesic_distance(&self, other: &Self) -> f64 {
        // Compute relative rotation: R_rel = R₁ᵀ R₂
        let relative = Self {
            matrix: self.matrix.transpose() * other.matrix,
        };
        relative.distance_to_identity()
    }
}

impl approx::AbsDiffEq for So3Algebra {
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

impl approx::RelativeEq for So3Algebra {
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

impl fmt::Display for So3Algebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "so(3)[{:.4}, {:.4}, {:.4}]",
            self.0[0], self.0[1], self.0[2]
        )
    }
}

impl fmt::Display for SO3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dist = self.distance_to_identity();
        write!(f, "SO(3)(θ={:.4})", dist)
    }
}

/// Group multiplication: R₁ · R₂
impl Mul<&SO3> for &SO3 {
    type Output = SO3;
    fn mul(self, rhs: &SO3) -> SO3 {
        SO3 {
            matrix: self.matrix * rhs.matrix,
        }
    }
}

impl Mul<&SO3> for SO3 {
    type Output = SO3;
    fn mul(self, rhs: &SO3) -> SO3 {
        &self * rhs
    }
}

impl MulAssign<&SO3> for SO3 {
    fn mul_assign(&mut self, rhs: &SO3) {
        self.matrix *= rhs.matrix;
    }
}

impl LieGroup for SO3 {
    const MATRIX_DIM: usize = 3;

    type Algebra = So3Algebra;

    fn identity() -> Self {
        Self::identity()
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            matrix: self.matrix * other.matrix,
        }
    }

    fn inverse(&self) -> Self {
        Self::inverse(self)
    }

    fn conjugate_transpose(&self) -> Self {
        // For orthogonal matrices, conjugate transpose = transpose = inverse
        self.inverse()
    }

    fn adjoint_action(&self, algebra_element: &So3Algebra) -> So3Algebra {
        // For SO(3): Ad_R(v) = R v (matrix-vector multiplication)
        //
        // This corresponds to rotating the axis v by rotation R
        let v = algebra_element.0;
        let rotated = self.matrix * nalgebra::Vector3::new(v[0], v[1], v[2]);

        So3Algebra([rotated[0], rotated[1], rotated[2]])
    }

    fn distance_to_identity(&self) -> f64 {
        Self::distance_to_identity(self)
    }

    fn exp(tangent: &So3Algebra) -> Self {
        // Exponential map using Rodrigues formula
        let angle = tangent.norm();

        if angle < 1e-10 {
            return Self::identity();
        }

        let axis = [
            tangent.0[0] / angle,
            tangent.0[1] / angle,
            tangent.0[2] / angle,
        ];

        Self::rotation(axis, angle)
    }

    fn log(&self) -> crate::error::LogResult<So3Algebra> {
        use crate::error::LogError;

        // For SO(3), extract rotation angle and axis from the rotation matrix
        //
        // The angle θ satisfies:
        // Tr(R) = 1 + 2·cos(θ)
        //
        // The axis n̂ can be extracted from the skew-symmetric part:
        // R - R^T = 2·sin(θ)·[n̂]×
        // where [n̂]× is the cross-product matrix

        let trace = self.matrix.trace();

        // Extract angle: θ = arccos((Tr(R) - 1) / 2)
        let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();

        const SMALL_ANGLE_THRESHOLD: f64 = 1e-10;
        const SINGULARITY_THRESHOLD: f64 = 1e-6;

        if theta.abs() < SMALL_ANGLE_THRESHOLD {
            // Near identity: log(I) = 0
            return Ok(So3Algebra::zero());
        }

        if (theta - std::f64::consts::PI).abs() < SINGULARITY_THRESHOLD {
            // Near singularity at θ = π (rotation by 180°)
            // At exactly θ = π, the axis direction is ambiguous
            return Err(LogError::Singularity {
                reason: format!(
                    "SO(3) element at rotation angle θ ≈ π (θ = {:.6}), axis is ambiguous",
                    theta
                ),
            });
        }

        // Extract rotation axis from skew-symmetric part
        // R - R^T = 2·sin(θ)·[n̂]×
        //
        // [n̂]× = [[ 0,  -nz,  ny],
        //         [ nz,  0,  -nx],
        //         [-ny,  nx,  0 ]]
        //
        // So:
        // R[2,1] - R[1,2] = 2·sin(θ)·nx
        // R[0,2] - R[2,0] = 2·sin(θ)·ny
        // R[1,0] - R[0,1] = 2·sin(θ)·nz

        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-10 {
            // This shouldn't happen given our checks above, but guard against it
            return Err(LogError::NumericalInstability {
                reason: "sin(θ) too small for reliable axis extraction".to_string(),
            });
        }

        let two_sin_theta = 2.0 * sin_theta;
        let nx = (self.matrix[(2, 1)] - self.matrix[(1, 2)]) / two_sin_theta;
        let ny = (self.matrix[(0, 2)] - self.matrix[(2, 0)]) / two_sin_theta;
        let nz = (self.matrix[(1, 0)] - self.matrix[(0, 1)]) / two_sin_theta;

        // The logarithm is log(R) = θ·(nx, ny, nz) ∈ so(3)
        Ok(So3Algebra([theta * nx, theta * ny, theta * nz]))
    }
}

// ============================================================================
// Mathematical Property Implementations
// ============================================================================

use crate::traits::{Compact, SemiSimple, Simple};

/// SO(3) is compact
///
/// The rotation group is diffeomorphic to ℝP³ (real projective 3-space).
/// All rotations are bounded: ||R|| = 1.
impl Compact for SO3 {}

/// SO(3) is simple
///
/// It has no non-trivial normal subgroups (for dimension > 2).
impl Simple for SO3 {}

/// SO(3) is semi-simple
impl SemiSimple for SO3 {}

// ============================================================================
// Algebra Marker Traits
// ============================================================================

/// so(3) algebra elements are traceless by construction.
///
/// The representation `So3Algebra::new([f64; 3])` stores coefficients for
/// 3×3 antisymmetric matrices. All antisymmetric matrices are traceless.
impl TracelessByConstruction for So3Algebra {}

/// so(3) algebra elements are anti-Hermitian by construction.
///
/// Real antisymmetric matrices satisfy A^T = -A, which over ℝ is
/// equivalent to A† = -A (anti-Hermitian).
impl AntiHermitianByConstruction for So3Algebra {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let id = SO3::identity();
        assert!(id.verify_orthogonality(1e-10));
        assert_relative_eq!(id.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotations_orthogonal() {
        let rx = SO3::rotation_x(0.5);
        let ry = SO3::rotation_y(1.2);
        let rz = SO3::rotation_z(2.1);

        assert!(rx.verify_orthogonality(1e-10));
        assert!(ry.verify_orthogonality(1e-10));
        assert!(rz.verify_orthogonality(1e-10));
    }

    #[test]
    fn test_inverse() {
        let r = SO3::rotation_x(0.7);
        let r_inv = r.inverse();
        let product = r.compose(&r_inv);

        assert_relative_eq!(product.distance_to_identity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bracket_antisymmetry() {
        use crate::traits::LieAlgebra;

        let v = So3Algebra([0.7, -0.3, 1.2]);
        let w = So3Algebra([-0.5, 0.9, 0.4]);

        let vw = v.bracket(&w);
        let wv = w.bracket(&v);

        // [v, w] = -[w, v]
        for i in 0..3 {
            assert_relative_eq!(vw.0[i], -wv.0[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_bracket_bilinearity() {
        use crate::traits::LieAlgebra;

        let x = So3Algebra([1.0, 0.0, 0.0]);
        let y = So3Algebra([0.0, 1.0, 0.0]);
        let z = So3Algebra([0.0, 0.0, 1.0]);
        let alpha = 2.5;

        // Left linearity: [αX + Y, Z] = α[X, Z] + [Y, Z]
        let lhs = x.scale(alpha).add(&y).bracket(&z);
        let rhs = x.bracket(&z).scale(alpha).add(&y.bracket(&z));
        for i in 0..3 {
            assert_relative_eq!(lhs.0[i], rhs.0[i], epsilon = 1e-14);
        }

        // Right linearity: [Z, αX + Y] = α[Z, X] + [Z, Y]
        let lhs = z.bracket(&x.scale(alpha).add(&y));
        let rhs = z.bracket(&x).scale(alpha).add(&z.bracket(&y));
        for i in 0..3 {
            assert_relative_eq!(lhs.0[i], rhs.0[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_algebra_bracket() {
        use crate::traits::LieAlgebra;

        let lx = So3Algebra::basis_element(0);
        let ly = So3Algebra::basis_element(1);
        let bracket = lx.bracket(&ly);

        // [L_x, L_y] = L_z
        assert_relative_eq!(bracket.0[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(bracket.0[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(bracket.0[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_action() {
        use crate::traits::LieGroup;

        // Rotate L_x by 90° around Z
        let rz = SO3::rotation_z(std::f64::consts::FRAC_PI_2);
        let lx = So3Algebra([1.0, 0.0, 0.0]);
        let rotated = rz.adjoint_action(&lx);

        // Should give L_y = (0, 1, 0)
        assert_relative_eq!(rotated.0[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated.0[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated.0[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_orthogonality() {
        // Interpolation should always produce orthogonal matrices
        let r1 = SO3::rotation_x(0.3);
        let r2 = SO3::rotation_y(1.2);

        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let interp = r1.interpolate(&r2, t);
            assert!(
                interp.verify_orthogonality(1e-10),
                "Interpolated matrix not orthogonal at t={}",
                t
            );
        }
    }

    #[test]
    fn test_interpolate_endpoints() {
        let r1 = SO3::rotation_z(0.5);
        let r2 = SO3::rotation_z(1.5);

        let at_0 = r1.interpolate(&r2, 0.0);
        let at_1 = r1.interpolate(&r2, 1.0);

        assert_relative_eq!(
            at_0.distance_to_identity(),
            r1.distance_to_identity(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            at_1.distance_to_identity(),
            r2.distance_to_identity(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gram_schmidt_orthogonalize() {
        // Create a slightly perturbed (non-orthogonal) matrix
        let perturbed = Matrix3::new(1.001, 0.01, 0.0, 0.0, 0.999, 0.01, 0.0, 0.0, 1.002);

        let fixed = SO3::gram_schmidt_orthogonalize(perturbed);
        assert!(fixed.verify_orthogonality(1e-10));

        // Check determinant is +1 (proper rotation)
        let det = fixed.matrix.determinant();
        assert_relative_eq!(det, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_renormalize() {
        // Simulate drift from many multiplications
        let r = SO3::rotation_z(0.001);
        let mut accumulated = SO3::identity();

        for _ in 0..1000 {
            accumulated = accumulated.compose(&r);
        }

        // May have drifted slightly
        let renormalized = accumulated.renormalize();
        assert!(renormalized.verify_orthogonality(1e-12));
    }

    #[test]
    fn test_geodesic_distance_symmetric() {
        let r1 = SO3::rotation_x(0.5);
        let r2 = SO3::rotation_y(1.0);

        let d12 = r1.geodesic_distance(&r2);
        let d21 = r2.geodesic_distance(&r1);

        assert_relative_eq!(d12, d21, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        use crate::traits::{LieAlgebra, LieGroup};

        // Small angles (well within convergence)
        for &angle in &[0.1, 0.5, 1.0, 2.0] {
            let v = So3Algebra([angle * 0.6, angle * 0.8, 0.0]);
            let r = SO3::exp(&v);
            assert!(r.verify_orthogonality(1e-10));

            let v_back = SO3::log(&r).expect("log should succeed for small angles");
            let diff = v.add(&v_back.scale(-1.0));
            assert!(
                diff.norm() < 1e-9,
                "exp/log roundtrip failed at angle {}: error = {:.2e}",
                angle,
                diff.norm()
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        use crate::traits::LieGroup;

        // Start from group elements, take log, then exp back
        let rotations = [
            SO3::rotation_x(0.7),
            SO3::rotation_y(1.3),
            SO3::rotation_z(0.4),
            SO3::rotation_x(0.3).compose(&SO3::rotation_y(0.5)),
        ];

        for (i, r) in rotations.iter().enumerate() {
            let v = SO3::log(r).expect("log should succeed");
            let r_back = SO3::exp(&v);
            let dist = r.geodesic_distance(&r_back);
            assert!(
                dist < 1e-9,
                "log/exp roundtrip failed for rotation {}: distance = {:.2e}",
                i,
                dist
            );
        }
    }

    #[test]
    fn test_jacobi_identity() {
        use crate::traits::LieAlgebra;

        let x = So3Algebra([1.0, 0.0, 0.0]);
        let y = So3Algebra([0.0, 1.0, 0.0]);
        let z = So3Algebra([0.0, 0.0, 1.0]);

        // [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
        let t1 = x.bracket(&y.bracket(&z));
        let t2 = y.bracket(&z.bracket(&x));
        let t3 = z.bracket(&x.bracket(&y));
        let sum = t1.add(&t2).add(&t3);

        assert!(
            sum.norm() < 1e-14,
            "Jacobi identity violated: ||sum|| = {:.2e}",
            sum.norm()
        );
    }

    #[test]
    fn test_geodesic_distance_to_self() {
        let r = SO3::rotation([1.0, 2.0, 3.0], 0.8);
        let d = r.geodesic_distance(&r);
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_distance_to_identity() {
        let angle = 0.7;
        let r = SO3::rotation_z(angle);

        let d = SO3::identity().geodesic_distance(&r);
        assert_relative_eq!(d, angle, epsilon = 1e-10);
    }

    // ========================================================================
    // Property-Based Tests for Group Axioms
    // ========================================================================
    //
    // These tests use proptest to verify that SO(3) satisfies the
    // mathematical axioms of a Lie group for randomly generated elements.
    //
    // This is a form of **specification-based testing**: the group axioms
    // are the specification, and we verify they hold for all inputs.
    //
    // Run with: cargo test --features nightly

    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    /// Strategy for generating arbitrary SO(3) elements.
    ///
    /// We generate SO(3) elements by composing three Euler rotations:
    /// `R = R_z(α) · R_y(β) · R_x(γ)`
    ///
    /// This gives good coverage of SO(3) ≅ ℝP³.
    #[cfg(feature = "proptest")]
    fn arb_so3() -> impl Strategy<Value = SO3> {
        use std::f64::consts::TAU;

        // Generate three Euler angles
        let alpha = 0.0..TAU;
        let beta = 0.0..TAU;
        let gamma = 0.0..TAU;

        (alpha, beta, gamma).prop_map(|(a, b, c)| {
            SO3::rotation_z(a)
                .compose(&SO3::rotation_y(b))
                .compose(&SO3::rotation_x(c))
        })
    }

    /// Strategy for generating arbitrary `So3Algebra` elements.
    ///
    /// We generate algebra elements by picking random coefficients in [-π, π]
    /// for each of the three basis directions (`L_x`, `L_y`, `L_z`).
    #[cfg(feature = "proptest")]
    fn arb_so3_algebra() -> impl Strategy<Value = So3Algebra> {
        use std::f64::consts::PI;

        ((-PI..PI), (-PI..PI), (-PI..PI)).prop_map(|(a, b, c)| So3Algebra([a, b, c]))
    }

    #[cfg(feature = "proptest")]
    proptest! {
        /// **Group Axiom 1: Identity Element**
        ///
        /// For all R ∈ SO(3):
        /// - I · R = R (left identity)
        /// - R · I = R (right identity)
        ///
        /// where I = identity rotation
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_identity_axiom(r in arb_so3()) {
            let e = SO3::identity();

            // Left identity: I · R = R
            let left = e.compose(&r);
            prop_assert!(
                left.distance(&r) < 1e-7,
                "Left identity failed: I·R != R, distance = {}",
                left.distance(&r)
            );

            // Right identity: R · I = R
            let right = r.compose(&e);
            prop_assert!(
                right.distance(&r) < 1e-7,
                "Right identity failed: R·I != R, distance = {}",
                right.distance(&r)
            );
        }

        /// **Group Axiom 2: Inverse Element**
        ///
        /// For all R ∈ SO(3):
        /// - R · R⁻¹ = I (right inverse)
        /// - R⁻¹ · R = I (left inverse)
        ///
        /// where R⁻¹ = inverse of R (equals R^T for orthogonal matrices)
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_inverse_axiom(r in arb_so3()) {
            let r_inv = r.inverse();

            // Right inverse: R · R⁻¹ = I
            let right_product = r.compose(&r_inv);
            prop_assert!(
                right_product.is_near_identity(1e-7),
                "Right inverse failed: R·R⁻¹ != I, distance = {}",
                right_product.distance_to_identity()
            );

            // Left inverse: R⁻¹ · R = I
            let left_product = r_inv.compose(&r);
            prop_assert!(
                left_product.is_near_identity(1e-7),
                "Left inverse failed: R⁻¹·R != I, distance = {}",
                left_product.distance_to_identity()
            );
        }

        /// **Group Axiom 3: Associativity**
        ///
        /// For all R₁, R₂, R₃ ∈ SO(3):
        /// - (R₁ · R₂) · R₃ = R₁ · (R₂ · R₃)
        ///
        /// Group multiplication is associative.
        ///
        /// Note: We use tolerance 1e-7 to account for floating-point
        /// rounding errors in matrix operations.
        #[test]
        fn prop_associativity(r1 in arb_so3(), r2 in arb_so3(), r3 in arb_so3()) {
            // Left association: (R₁ · R₂) · R₃
            let left_assoc = r1.compose(&r2).compose(&r3);

            // Right association: R₁ · (R₂ · R₃)
            let right_assoc = r1.compose(&r2.compose(&r3));

            prop_assert!(
                left_assoc.distance(&right_assoc) < 1e-7,
                "Associativity failed: (R₁·R₂)·R₃ != R₁·(R₂·R₃), distance = {}",
                left_assoc.distance(&right_assoc)
            );
        }

        /// **Lie Group Property: Inverse is Smooth**
        ///
        /// For SO(3), the inverse operation is smooth (continuously differentiable).
        /// We verify this by checking that nearby elements have nearby inverses.
        #[test]
        fn prop_inverse_continuity(r in arb_so3()) {
            // Create a small perturbation
            let epsilon = 0.01;
            let perturbation = SO3::rotation_x(epsilon);
            let r_perturbed = r.compose(&perturbation);

            // Check that inverses are close
            let inv_distance = r.inverse().distance(&r_perturbed.inverse());

            prop_assert!(
                inv_distance < 0.1,
                "Inverse not continuous: small perturbation caused large inverse change, distance = {}",
                inv_distance
            );
        }

        /// **Orthogonality Preservation**
        ///
        /// All SO(3) operations should preserve orthogonality.
        /// This is not strictly a group axiom, but it's essential for SO(3).
        #[test]
        fn prop_orthogonality_preserved(r1 in arb_so3(), r2 in arb_so3()) {
            // Composition preserves orthogonality
            let product = r1.compose(&r2);
            prop_assert!(
                product.verify_orthogonality(1e-10),
                "Composition violated orthogonality"
            );

            // Inverse preserves orthogonality
            let inv = r1.inverse();
            prop_assert!(
                inv.verify_orthogonality(1e-10),
                "Inverse violated orthogonality"
            );
        }

        /// **Adjoint Representation: Group Homomorphism**
        ///
        /// The adjoint representation Ad: G → Aut(𝔤) is a group homomorphism:
        /// - Ad_{R₁∘R₂}(v) = Ad_{R₁}(Ad_{R₂}(v))
        ///
        /// This is a fundamental property that must hold for the adjoint action
        /// to be a valid representation of the group.
        #[test]
        fn prop_adjoint_homomorphism(
            r1 in arb_so3(),
            r2 in arb_so3(),
            v in arb_so3_algebra()
        ) {
            // Compute Ad_{R₁∘R₂}(v)
            let r_composed = r1.compose(&r2);
            let left = r_composed.adjoint_action(&v);

            // Compute Ad_{R₁}(Ad_{R₂}(v))
            let ad_r2_v = r2.adjoint_action(&v);
            let right = r1.adjoint_action(&ad_r2_v);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-7,
                "Adjoint homomorphism failed: Ad_{{R₁∘R₂}}(v) != Ad_{{R₁}}(Ad_{{R₂}}(v)), diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Identity Action**
        ///
        /// The identity element acts trivially on the Lie algebra:
        /// - Ad_I(v) = v for all v ∈ so(3)
        #[test]
        fn prop_adjoint_identity(v in arb_so3_algebra()) {
            let e = SO3::identity();
            let result = e.adjoint_action(&v);

            let diff = result.add(&v.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-10,
                "Identity action failed: Ad_I(v) != v, diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Lie Bracket Preservation**
        ///
        /// The adjoint representation preserves the Lie bracket:
        /// - Ad_R([v,w]) = [Ad_R(v), Ad_R(w)]
        ///
        /// This is a critical property that ensures the adjoint action
        /// is a Lie algebra automorphism.
        #[test]
        fn prop_adjoint_bracket_preservation(
            r in arb_so3(),
            v in arb_so3_algebra(),
            w in arb_so3_algebra()
        ) {
            use crate::traits::LieAlgebra;

            // Compute Ad_R([v,w])
            let bracket_vw = v.bracket(&w);
            let left = r.adjoint_action(&bracket_vw);

            // Compute [Ad_R(v), Ad_R(w)]
            let ad_v = r.adjoint_action(&v);
            let ad_w = r.adjoint_action(&w);
            let right = ad_v.bracket(&ad_w);

            // They should be equal
            let diff = left.add(&right.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-6,
                "Bracket preservation failed: Ad_R([v,w]) != [Ad_R(v), Ad_R(w)], diff norm = {}",
                diff.norm()
            );
        }

        /// **Adjoint Representation: Inverse Property**
        ///
        /// The inverse of an element acts as the inverse transformation:
        /// - Ad_{R⁻¹}(Ad_R(v)) = v
        #[test]
        fn prop_adjoint_inverse(r in arb_so3(), v in arb_so3_algebra()) {
            // Apply Ad_R then Ad_{R⁻¹}
            let ad_r_v = r.adjoint_action(&v);
            let r_inv = r.inverse();
            let result = r_inv.adjoint_action(&ad_r_v);

            // Should recover v
            let diff = result.add(&v.scale(-1.0));
            prop_assert!(
                diff.norm() < 1e-7,
                "Inverse property failed: Ad_{{R⁻¹}}(Ad_R(v)) != v, diff norm = {}",
                diff.norm()
            );
        }
    }
}
