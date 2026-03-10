//! Quaternionic Formulation of SU(2)
//!
//! **Addressing Penrose's Recommendation:**
//! > "SU(2) ≅ unit quaternions. Why not use this beautiful structure?"
//!
//! This module implements SU(2) using quaternions instead of 2×2 matrices.
//!
//! # Mathematical Background
//!
//! **Quaternions:** ℍ = {a + bi + cj + dk | a,b,c,d ∈ ℝ}
//! with multiplication rules: i² = j² = k² = ijk = -1
//!
//! **Unit Quaternions:** S³ = {q ∈ ℍ | |q| = 1} ⊂ ℍ
//!
//! **Fundamental Isomorphism:** SU(2) ≅ S³
//!
//! The map is:
//! ```text
//! SU(2) matrix [[α, -β*], [β, α*]]  ↔  Quaternion q = α + βj
//!                                        where α = a + bi, β = c + di
//!                                        ⟺ q = a + bi + cj + dk
//! ```
//!
//! # Advantages Over Matrix Representation
//!
//! 1. **Compact**: 4 reals vs 8 reals (complex 2×2 matrix)
//! 2. **Efficient**: Quaternion multiplication is faster than matrix multiplication
//! 3. **Geometric**: Direct axis-angle interpretation for rotations
//! 4. **Numerical**: Better numerical stability (no matrix decomposition needed)
//! 5. **Elegant**: Exponential map is cleaner
//!
//! # The SU(2) Double Cover of SO(3) ⭐
//!
//! **Critical Physics Insight:**
//!
//! SU(2) is a **double cover** of SO(3) (3D rotations):
//! ```text
//! SU(2) → SO(3)  is a 2-to-1 surjective homomorphism
//! ```
//!
//! **Meaning:** Quaternions **q** and **-q** represent the **same rotation** in 3D space,
//! but **different quantum states** in SU(2).
//!
//! ## Mathematical Statement
//!
//! For any unit quaternion q:
//! - **q** rotates a vector v by angle θ around axis n̂
//! - **-q** also rotates v by angle θ around axis n̂  (same rotation!)
//! - But q ≠ -q as group elements in SU(2)
//!
//! Path from q → -q → q (going 4π around) is **topologically non-trivial**.
//!
//! ## Physical Consequences
//!
//! 1. **Fermions (spin-½ particles)**:
//!    - Under 2π rotation: ψ → -ψ  (sign flip!)
//!    - Under 4π rotation: ψ → ψ  (back to original)
//!    - This is why fermions have **half-integer spin**
//!
//! 2. **Bosons (integer spin)**:
//!    - Under 2π rotation: ψ → ψ  (no sign change)
//!    - Described by SO(3), not SU(2)
//!
//! 3. **Berry Phase**:
//!    - Geometric phase accumulated when traversing closed path in SU(2)
//!    - q → -q → q acquires phase π (Pancharatnam-Berry phase)
//!
//! ## Example: Rotation by 2π
//!
//! ```text
//! Rotation by angle θ = 2π around z-axis:
//!   q = cos(π) + sin(π)·k = -1 + 0·k = -1  (minus identity!)
//!
//! But -1 acts on vector v as:
//!   v → -1·v·(-1)⁻¹ = -1·v·(-1) = v  (identity rotation in SO(3))
//! ```
//!
//! ## Gauge Theory Interpretation
//!
//! In gauge theory on networks:
//! - **Connections** A_{ij} ∈ SU(2) carry full SU(2) structure (not just rotations)
//! - **Wilson loops** W(C) can wind around non-trivially in SU(2)
//! - Path C → 2π rotation can give W(C) = -I ≠ I (topological effect!)
//! - This is invisible in classical physics but crucial for quantum systems
//!
//! ## References
//!
//! - Dirac, P.A.M.: "The Principles of Quantum Mechanics" (1930) - Original treatment
//! - Feynman & Hibbs: "Quantum Mechanics and Path Integrals" - Spin-statistics connection
//! - Penrose: "The Road to Reality" § 11.3 - Geometric interpretation
//! - Aharonov & Susskind: "Observability of the sign change of spinors" (1967)

use std::ops::{Mul, MulAssign};

/// Unit quaternion representing an element of SU(2)
///
/// Quaternion q = w + xi + yj + zk where w² + x² + y² + z² = 1
///
/// # Geometric Interpretation
///
/// Every unit quaternion represents a rotation in 3D space:
/// - Rotation by angle θ around axis n̂ = (nx, ny, nz):
///   q = cos(θ/2) + sin(θ/2)(nx·i + ny·j + nz·k)
///
/// # ⚠️ Important: Double Cover Property
///
/// **q and -q represent the SAME rotation but DIFFERENT quantum states:**
/// - Both rotate 3D vectors identically (v → q·v·q⁻¹ = (-q)·v·(-q)⁻¹)
/// - But q ≠ -q as SU(2) group elements
/// - Consequence: Fermions change sign under 2π rotation (ψ → -ψ)
///
/// This is NOT a numerical issue or degeneracy - it's fundamental topology!
///
/// # Relation to SU(2) Matrix
///
/// q = w + xi + yj + zk corresponds to SU(2) matrix:
/// ```text
/// U = [[ w + ix,  -y + iz],
///      [ y + iz,   w - ix]]
/// ```
///
/// Note: -q gives matrix -U, which acts identically on vectors but represents
/// a different element of SU(2) (relevant for spinors/fermions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitQuaternion {
    // Private fields enforce unit norm invariant through constructors
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl UnitQuaternion {
    /// Get the real (scalar) component.
    #[inline]
    #[must_use]
    pub fn w(&self) -> f64 {
        self.w
    }

    /// Get the imaginary i component.
    #[inline]
    #[must_use]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the imaginary j component.
    #[inline]
    #[must_use]
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Get the imaginary k component.
    #[inline]
    #[must_use]
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Get all components as a tuple (w, x, y, z).
    #[inline]
    #[must_use]
    pub fn components(&self) -> (f64, f64, f64, f64) {
        (self.w, self.x, self.y, self.z)
    }

    /// Get all components as an array [w, x, y, z].
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

// ============================================================================
// Numerical Tolerance Constants
// ============================================================================
//
// These thresholds are chosen based on IEEE 754 f64 precision (ε ≈ 2.2e-16)
// and practical considerations for accumulated floating-point error.
//
// Rationale: 1e-10 = ~10^6 × machine_epsilon
// This allows for approximately 6 orders of magnitude of accumulated error
// from typical operations (multiplication, addition, normalization) before
// triggering degeneracy handling.
//
// For quaternion operations on unit quaternions, typical error accumulation:
// - Single operation: O(ε) ≈ 2e-16
// - After ~1000 operations: O(1000ε) ≈ 2e-13
// - Threshold 1e-10 provides ~1000× safety margin
//
// Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2002)

/// Threshold below which a norm is considered zero (degenerate case).
/// Used for normalization guards and axis-angle singularity detection.
const NORM_EPSILON: f64 = 1e-10;

/// Threshold for detecting near-identity quaternions.
/// Used in logarithm to avoid division by near-zero sin(θ/2).
const IDENTITY_EPSILON: f64 = 1e-10;

/// Threshold for detecting rotation angle near π (axis ambiguity).
/// At θ = π, the rotation axis becomes undefined (any axis works).
/// Reserved for future implementation of θ≈π handling.
#[allow(dead_code)]
const SINGULARITY_EPSILON: f64 = 1e-6;

impl UnitQuaternion {
    /// Identity element: q = 1
    #[must_use]
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create from components (automatically normalizes to unit quaternion)
    ///
    /// Returns identity if input norm < `NORM_EPSILON` (degenerate case).
    #[must_use]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        if norm < NORM_EPSILON {
            return Self::identity();
        }
        Self {
            w: w / norm,
            x: x / norm,
            y: y / norm,
            z: z / norm,
        }
    }

    /// Create from axis-angle representation
    ///
    /// # Arguments
    /// * `axis` - Rotation axis (nx, ny, nz), will be normalized
    /// * `angle` - Rotation angle in radians
    ///
    /// # Returns
    /// Unit quaternion q = cos(θ/2) + sin(θ/2)(nx·i + ny·j + nz·k)
    #[must_use]
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let [nx, ny, nz] = axis;
        let axis_norm = (nx * nx + ny * ny + nz * nz).sqrt();

        // Degenerate axis: return identity (no rotation)
        if axis_norm < NORM_EPSILON {
            return Self::identity();
        }

        let half_angle = angle / 2.0;
        let (sin_half, cos_half) = half_angle.sin_cos();

        Self {
            w: cos_half,
            x: sin_half * nx / axis_norm,
            y: sin_half * ny / axis_norm,
            z: sin_half * nz / axis_norm,
        }
    }

    /// Extract axis and angle from quaternion
    ///
    /// # Returns
    /// (axis, angle) where axis is normalized and angle ∈ [0, 2π]
    #[must_use]
    pub fn to_axis_angle(&self) -> ([f64; 3], f64) {
        // q = cos(θ/2) + sin(θ/2)·n̂
        let angle = 2.0 * self.w.clamp(-1.0, 1.0).acos();
        let sin_half = (1.0 - self.w * self.w).max(0.0).sqrt();

        if sin_half < IDENTITY_EPSILON {
            // Near identity, axis is arbitrary
            return ([1.0, 0.0, 0.0], 0.0);
        }

        let axis = [self.x / sin_half, self.y / sin_half, self.z / sin_half];

        (axis, angle)
    }

    /// Rotation around X-axis by angle θ
    #[must_use]
    pub fn rotation_x(theta: f64) -> Self {
        Self::from_axis_angle([1.0, 0.0, 0.0], theta)
    }

    /// Rotation around Y-axis by angle θ
    #[must_use]
    pub fn rotation_y(theta: f64) -> Self {
        Self::from_axis_angle([0.0, 1.0, 0.0], theta)
    }

    /// Rotation around Z-axis by angle θ
    #[must_use]
    pub fn rotation_z(theta: f64) -> Self {
        Self::from_axis_angle([0.0, 0.0, 1.0], theta)
    }

    /// Quaternion conjugate: q* = w - xi - yj - zk
    ///
    /// For unit quaternions: q* = q⁻¹ (inverse)
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion inverse: q⁻¹
    ///
    /// For unit quaternions: q⁻¹ = q*
    #[must_use]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// Norm squared: |q|² = w² + x² + y² + z²
    ///
    /// Should always be 1 for unit quaternions
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm: |q| = sqrt(w² + x² + y² + z²)
    ///
    /// Should always be 1 for unit quaternions
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Renormalize to unit quaternion (fix numerical drift)
    #[must_use]
    pub fn normalize(&self) -> Self {
        Self::new(self.w, self.x, self.y, self.z)
    }

    /// Distance to identity (geodesic distance on S³)
    ///
    /// d(q, 1) = 2·arccos(|w|)
    ///
    /// This is the rotation angle in [0, π]
    #[must_use]
    pub fn distance_to_identity(&self) -> f64 {
        2.0 * self.w.abs().clamp(-1.0, 1.0).acos()
    }

    /// Geodesic distance to another quaternion
    ///
    /// d(q₁, q₂) = arccos(|q₁·q₂|) where · is quaternion dot product
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        dot.abs().clamp(-1.0, 1.0).acos()
    }

    /// Spherical linear interpolation (SLERP)
    ///
    /// Interpolate between self and other with parameter t ∈ `[0,1]`
    /// Returns shortest path on S³
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;

        // If quaternions are close, use linear interpolation
        if dot.abs() > 0.9995 {
            return Self::new(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            );
        }

        // Ensure shortest path
        let (other_w, other_x, other_y, other_z) = if dot < 0.0 {
            (-other.w, -other.x, -other.y, -other.z)
        } else {
            (other.w, other.x, other.y, other.z)
        };

        let theta = dot.abs().acos();
        let sin_theta = theta.sin();

        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Self::new(
            a * self.w + b * other_w,
            a * self.x + b * other_x,
            a * self.y + b * other_y,
            a * self.z + b * other_z,
        )
    }

    /// Exponential map from Lie algebra 𝔰𝔲(2) to group SU(2)
    ///
    /// Given a vector v = (v₁, v₂, v₃) ∈ ℝ³ (Lie algebra element),
    /// compute exp(v) as a unit quaternion.
    ///
    /// Formula: exp(v) = cos(|v|/2) + sin(|v|/2)·(v/|v|)
    #[must_use]
    pub fn exp(v: [f64; 3]) -> Self {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

        if norm < NORM_EPSILON {
            return Self::identity();
        }

        let half_norm = norm / 2.0;
        let (sin_half, cos_half) = half_norm.sin_cos();

        Self {
            w: cos_half,
            x: sin_half * v[0] / norm,
            y: sin_half * v[1] / norm,
            z: sin_half * v[2] / norm,
        }
    }

    /// Logarithm map from group SU(2) to Lie algebra 𝔰𝔲(2)
    ///
    /// Returns vector v ∈ ℝ³ such that exp(v) = q
    ///
    /// Formula: log(q) = (θ/sin(θ/2))·(x, y, z) where θ = 2·arccos(w)
    #[must_use]
    pub fn log(&self) -> [f64; 3] {
        let theta = 2.0 * self.w.clamp(-1.0, 1.0).acos();
        let sin_half = (1.0 - self.w * self.w).max(0.0).sqrt();

        if sin_half < IDENTITY_EPSILON {
            // Near identity
            return [0.0, 0.0, 0.0];
        }

        let scale = theta / sin_half;
        [scale * self.x, scale * self.y, scale * self.z]
    }

    /// Convert to SU(2) matrix representation (for compatibility)
    ///
    /// Returns 2×2 complex matrix:
    /// U = [[ w + ix,  -y + iz],
    ///      [ y + iz,   w - ix]]
    #[must_use]
    pub fn to_matrix(&self) -> [[num_complex::Complex64; 2]; 2] {
        use num_complex::Complex64;

        [
            [
                Complex64::new(self.w, self.x),
                Complex64::new(-self.y, self.z),
            ],
            [
                Complex64::new(self.y, self.z),
                Complex64::new(self.w, -self.x),
            ],
        ]
    }

    /// Create from SU(2) matrix
    ///
    /// Expects matrix [[α, -β*], [β, α*]] with |α|² + |β|² = 1
    #[must_use]
    pub fn from_matrix(matrix: [[num_complex::Complex64; 2]; 2]) -> Self {
        // Extract α = a + ib from matrix[0][0]
        let a = matrix[0][0].re;
        let b = matrix[0][0].im;

        // Extract β = c + id from matrix[1][0]
        let c = matrix[1][0].re;
        let d = matrix[1][0].im;

        // q = a + bi + cj + dk
        Self::new(a, b, c, d)
    }

    /// Act on a 3D vector by conjugation: v' = qvq*
    ///
    /// This is the rotation action of SU(2) on ℝ³
    /// Identifies v = (x,y,z) with quaternion xi + yj + zk
    ///
    /// Uses the direct formula (Rodrigues) for efficiency:
    /// v' = v + 2w(n×v) + 2(n×(n×v)) where n = (x,y,z)
    #[must_use]
    pub fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let (w, qx, qy, qz) = (self.w, self.x, self.y, self.z);
        let (vx, vy, vz) = (v[0], v[1], v[2]);

        // Cross product: n × v
        let cx = qy * vz - qz * vy;
        let cy = qz * vx - qx * vz;
        let cz = qx * vy - qy * vx;

        // Cross product: n × (n × v)
        let ccx = qy * cz - qz * cy;
        let ccy = qz * cx - qx * cz;
        let ccz = qx * cy - qy * cx;

        // v' = v + 2w(n×v) + 2(n×(n×v))
        [
            vx + 2.0 * (w * cx + ccx),
            vy + 2.0 * (w * cy + ccy),
            vz + 2.0 * (w * cz + ccz),
        ]
    }

    /// Verify this is approximately a unit quaternion
    #[must_use]
    pub fn verify_unit(&self, tolerance: f64) -> bool {
        (self.norm_squared() - 1.0).abs() < tolerance
    }

    /// Construct the 3×3 rotation matrix corresponding to this unit quaternion.
    ///
    /// Returns the SO(3) matrix R such that R·v = `rotate_vector(v)` for all v ∈ ℝ³.
    ///
    /// # Lean Correspondence
    ///
    /// This is the Rust counterpart of `toRotationMatrix` in
    /// `OrthogonalGroups.lean`. The Lean proofs establish:
    /// - `toRotationMatrix_orthogonal_axiom`: R(q)ᵀ·R(q) = I
    /// - `toRotationMatrix_det_axiom`: det(R(q)) = 1
    /// - `toRotationMatrix_mul_axiom`: R(q₁·q₂) = R(q₁)·R(q₂)
    /// - `toRotationMatrix_neg`: R(-q) = R(q)
    #[must_use]
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

// Quaternion multiplication: Hamilton product
impl Mul for UnitQuaternion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // (w₁ + x₁i + y₁j + z₁k)(w₂ + x₂i + y₂j + z₂k)
        Self::new(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        )
    }
}

impl MulAssign for UnitQuaternion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_identity() {
        let q = UnitQuaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
        assert!(q.verify_unit(1e-10));
    }

    #[test]
    fn test_normalization() {
        let q = UnitQuaternion::new(1.0, 2.0, 3.0, 4.0);
        assert!((q.norm() - 1.0).abs() < 1e-10);
        assert!(q.verify_unit(1e-10));
    }

    #[test]
    fn test_axis_angle_roundtrip() {
        let axis = [1.0, 2.0, 3.0];
        let angle = PI / 3.0;

        let q = UnitQuaternion::from_axis_angle(axis, angle);
        let (axis_out, angle_out) = q.to_axis_angle();

        // Normalize input axis for comparison
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let axis_normalized = [
            axis[0] / axis_norm,
            axis[1] / axis_norm,
            axis[2] / axis_norm,
        ];

        assert!((angle - angle_out).abs() < 1e-10);
        for i in 0..3 {
            assert!((axis_normalized[i] - axis_out[i]).abs() < 1e-10);
        }

        println!(
            "✓ Axis-angle roundtrip: axis=[{:.3},{:.3},{:.3}], angle={:.3}",
            axis_out[0], axis_out[1], axis_out[2], angle_out
        );
    }

    #[test]
    fn test_rotation_x() {
        let q = UnitQuaternion::rotation_x(PI / 2.0);

        // Should rotate (0,1,0) to (0,0,1)
        let v = [0.0, 1.0, 0.0];
        let rotated = q.rotate_vector(v);

        assert!(rotated[0].abs() < 1e-10);
        assert!(rotated[1].abs() < 1e-10);
        assert!((rotated[2] - 1.0).abs() < 1e-10);

        println!(
            "✓ Rotation X by π/2: (0,1,0) → ({:.3},{:.3},{:.3})",
            rotated[0], rotated[1], rotated[2]
        );
    }

    #[test]
    fn test_quaternion_multiplication() {
        let q1 = UnitQuaternion::rotation_x(PI / 4.0);
        let q2 = UnitQuaternion::rotation_y(PI / 4.0);

        let q3 = q1 * q2;

        // Should still be unit quaternion
        assert!(q3.verify_unit(1e-10));

        // Composition of rotations
        let v = [1.0, 0.0, 0.0];
        let rotated_composed = q3.rotate_vector(v);
        let rotated_separate = q1.rotate_vector(q2.rotate_vector(v));

        for i in 0..3 {
            assert!((rotated_composed[i] - rotated_separate[i]).abs() < 1e-10);
        }

        println!("✓ Quaternion multiplication preserves unitarity and composition");
    }

    #[test]
    fn test_inverse() {
        let q = UnitQuaternion::rotation_z(PI / 3.0);
        let q_inv = q.inverse();

        let product = q * q_inv;

        assert!((product.w - 1.0).abs() < 1e-10);
        assert!(product.x.abs() < 1e-10);
        assert!(product.y.abs() < 1e-10);
        assert!(product.z.abs() < 1e-10);

        println!("✓ Inverse: q · q⁻¹ = identity");
    }

    #[test]
    fn test_distance_to_identity() {
        let q_identity = UnitQuaternion::identity();
        assert!(q_identity.distance_to_identity() < 1e-10);

        let q_pi = UnitQuaternion::rotation_x(PI);
        assert!((q_pi.distance_to_identity() - PI).abs() < 1e-10);

        let q_half_pi = UnitQuaternion::rotation_y(PI / 2.0);
        assert!((q_half_pi.distance_to_identity() - PI / 2.0).abs() < 1e-10);

        println!(
            "✓ Distance to identity: d(I)=0, d(π)={:.3}, d(π/2)={:.3}",
            q_pi.distance_to_identity(),
            q_half_pi.distance_to_identity()
        );
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let v = [0.5, 0.3, 0.2];

        let q = UnitQuaternion::exp(v);
        let v_out = q.log();

        for i in 0..3 {
            assert!((v[i] - v_out[i]).abs() < 1e-10);
        }

        println!(
            "✓ Exp-log roundtrip: v=[{:.3},{:.3},{:.3}]",
            v[0], v[1], v[2]
        );
    }

    #[test]
    fn test_slerp() {
        let q1 = UnitQuaternion::rotation_x(0.0);
        let q2 = UnitQuaternion::rotation_x(PI / 2.0);

        let q_mid = q1.slerp(&q2, 0.5);

        // Should be rotation by π/4
        assert!((q_mid.distance_to_identity() - PI / 4.0).abs() < 1e-10);

        // At t=0, should be q1
        let q_0 = q1.slerp(&q2, 0.0);
        assert!(q_0.distance_to(&q1) < 1e-10);

        // At t=1, should be q2
        let q_1 = q1.slerp(&q2, 1.0);
        assert!(q_1.distance_to(&q2) < 1e-10);

        println!("✓ SLERP: smooth interpolation on S³");
    }

    #[test]
    fn test_near_pi_axis_angle_roundtrip() {
        // Near θ = π, sin(θ/2) ≈ 1 and cos(θ/2) ≈ 0.
        // Axis extraction divides by sin(θ/2), which is well-conditioned here.
        // But w ≈ 0 means acos(w) sensitivity is low — this should work fine.
        let axis = [0.0, 0.0, 1.0];
        let angle = PI - 1e-8;

        let q = UnitQuaternion::from_axis_angle(axis, angle);
        assert!(q.verify_unit(1e-10));

        let (axis_out, angle_out) = q.to_axis_angle();
        assert!(
            (angle - angle_out).abs() < 1e-6,
            "Near-π angle roundtrip: got {}, expected {}",
            angle_out,
            angle
        );
        assert!(
            (axis_out[2] - 1.0).abs() < 1e-6,
            "Near-π axis roundtrip: got {:?}",
            axis_out
        );

        // Also test the rotation acts correctly
        let v = [1.0, 0.0, 0.0];
        let rotated = q.rotate_vector(v);
        // Rotation by π-ε around z should map (1,0,0) ≈ (-1, 0, 0)
        assert!(
            (rotated[0] + 1.0).abs() < 1e-6,
            "Near-π rotation: got {:?}",
            rotated
        );
    }

    #[test]
    fn test_exp_log_near_identity() {
        // Very small algebra element — tests the near-identity branch
        let v = [1e-12, 0.0, 0.0];
        let q = UnitQuaternion::exp(v);
        assert!(q.verify_unit(1e-14));
        assert!(q.distance_to_identity() < 1e-10);

        let log_q = q.log();
        // Near identity, log should return near-zero
        for &c in &log_q {
            assert!(c.abs() < 1e-8);
        }
    }

    #[test]
    fn test_matrix_conversion() {
        let q = UnitQuaternion::rotation_z(PI / 3.0);

        let matrix = q.to_matrix();
        let q_back = UnitQuaternion::from_matrix(matrix);

        assert!((q.w - q_back.w).abs() < 1e-10);
        assert!((q.x - q_back.x).abs() < 1e-10);
        assert!((q.y - q_back.y).abs() < 1e-10);
        assert!((q.z - q_back.z).abs() < 1e-10);

        println!("✓ Matrix conversion roundtrip preserves quaternion");
    }
}
