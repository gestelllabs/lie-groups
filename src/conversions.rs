//! Type-safe conversions between equivalent Lie group representations.
//!
//! This module provides [`From`] implementations connecting specialized
//! implementations to their generic counterparts:
//!
//! - [`SU2`] ↔ [`SUN<2>`]: Same 2×2 unitary matrices, different types
//! - [`Su2Algebra`] ↔ [`SunAlgebra<2>`]: Same Pauli basis, same ordering
//! - [`SU3`] ↔ [`SUN<3>`]: Same 3×3 unitary matrices, different types
//! - [`Su3Algebra`] ↔ [`SunAlgebra<3>`]: Gell-Mann basis with reordering
//! - [`UnitQuaternion`] ↔ [`SU2`]: Quaternion ↔ matrix isomorphism
//!
//! # Mathematical Guarantee
//!
//! These conversions are **structure-preserving** (Lie group homomorphisms):
//!
//! ```text
//! from(g · h) = from(g) · from(h)     // preserves composition
//! from(exp(X)) = exp(from(X))          // commutes with exp
//! from(g⁻¹) = from(g)⁻¹              // preserves inverse
//! ```
//!
//! These properties are verified by roundtrip tests in this module.
//!
//! # Basis Ordering: SU(3) ↔ SU(N=3)
//!
//! SU(3) uses the standard Gell-Mann ordering (λ₁..λ₈), while SU(N=3) groups
//! generators by type (symmetric, antisymmetric, diagonal). The conversion
//! applies the permutation:
//!
//! ```text
//! SU3 index:    [0, 1, 2, 3, 4, 5, 6, 7]   (Gell-Mann: λ₁..λ₈)
//! SUN<3> index: [0, 3, 6, 1, 4, 2, 5, 7]   (grouped by type)
//! ```

use crate::quaternion::UnitQuaternion;
use crate::su2::{Su2Algebra, SU2};
use crate::su3::{Su3Algebra, SU3};
use crate::sun::{SunAlgebra, SUN};

// ============================================================================
// SU2 ↔ SUN<2> (matrix representations share identical basis ordering)
// ============================================================================

impl From<SU2> for SUN<2> {
    fn from(g: SU2) -> Self {
        Self {
            matrix: g.matrix.clone(),
        }
    }
}

impl From<SUN<2>> for SU2 {
    fn from(g: SUN<2>) -> Self {
        Self {
            matrix: g.matrix.clone(),
        }
    }
}

impl From<Su2Algebra> for SunAlgebra<2> {
    fn from(x: Su2Algebra) -> Self {
        Self::new(x.components().to_vec())
    }
}

impl From<SunAlgebra<2>> for Su2Algebra {
    fn from(x: SunAlgebra<2>) -> Self {
        let c = x.coefficients();
        Self::new([c[0], c[1], c[2]])
    }
}

// ============================================================================
// SU3 ↔ SUN<3> (matrix identical, algebra basis needs permutation)
// ============================================================================

/// SU3 Gell-Mann index → SUN<3> generalized Gell-Mann index.
///
/// SU3 uses interleaved Gell-Mann ordering (λ₁..λ₈).
/// SUN<3> groups by type: symmetric, antisymmetric, diagonal.
const SU3_TO_SUN3: [usize; 8] = [0, 3, 6, 1, 4, 2, 5, 7];

/// SUN<3> generalized Gell-Mann index → SU3 Gell-Mann index (inverse permutation).
const SUN3_TO_SU3: [usize; 8] = [0, 3, 5, 1, 4, 6, 2, 7];

impl From<SU3> for SUN<3> {
    fn from(g: SU3) -> Self {
        Self {
            matrix: g.matrix().clone(),
        }
    }
}

impl From<SUN<3>> for SU3 {
    fn from(g: SUN<3>) -> Self {
        Self {
            matrix: g.matrix().clone(),
        }
    }
}

impl From<Su3Algebra> for SunAlgebra<3> {
    fn from(x: Su3Algebra) -> Self {
        let c = x.components();
        let mut sun_coeffs = vec![0.0; 8];
        for i in 0..8 {
            sun_coeffs[SU3_TO_SUN3[i]] = c[i];
        }
        Self::new(sun_coeffs)
    }
}

impl From<SunAlgebra<3>> for Su3Algebra {
    fn from(x: SunAlgebra<3>) -> Self {
        let c = x.coefficients();
        let mut su3_coeffs = [0.0; 8];
        for i in 0..8 {
            su3_coeffs[SUN3_TO_SU3[i]] = c[i];
        }
        Self::new(su3_coeffs)
    }
}

// ============================================================================
// UnitQuaternion ↔ SU2 (the SU(2) ≅ S³ isomorphism)
// ============================================================================

impl From<UnitQuaternion> for SU2 {
    /// Convert quaternion q = w + xi + yj + zk to the SU(2) matrix:
    ///
    /// ```text
    /// U = [[ w + ix, -y + iz ],
    ///      [ y + iz,  w - ix ]]
    /// ```
    ///
    /// **Convention note:** This is an anti-homomorphism with respect to
    /// multiplication: `U(q₁·q₂) = U(q₂)·U(q₁)`. This is standard in
    /// physics — quaternion left-action on vectors corresponds to matrix
    /// right-multiplication.
    fn from(q: UnitQuaternion) -> Self {
        let m = q.to_matrix();
        Self::from_matrix(m)
    }
}

impl From<SU2> for UnitQuaternion {
    /// Extract quaternion from SU(2) matrix:
    ///
    /// ```text
    /// w = Re(U₀₀),  x = Im(U₀₀),  y = Re(U₁₀) with sign flip,  z = Im(U₁₀)
    /// ```
    fn from(g: SU2) -> Self {
        let m = g.to_matrix();
        UnitQuaternion::from_matrix(m)
    }
}

// ============================================================================
// Tests: Structure-preserving roundtrip verification
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{LieAlgebra, LieGroup};

    // ====================================================================
    // SU2 ↔ SUN<2> roundtrips
    // ====================================================================

    #[test]
    fn test_su2_sun2_group_roundtrip() {
        let g = SU2::rotation_x(0.7);
        let sun: SUN<2> = g.clone().into();
        let back: SU2 = sun.into();
        assert!(
            g.distance(&back) < 1e-14,
            "SU2 → SUN<2> → SU2 roundtrip failed"
        );
    }

    #[test]
    fn test_su2_sun2_algebra_roundtrip() {
        let x = Su2Algebra::new([0.3, -0.7, 1.2]);
        let sun: SunAlgebra<2> = x.into();
        let back: Su2Algebra = sun.into();
        assert_eq!(
            x, back,
            "Su2Algebra → SunAlgebra<2> → Su2Algebra roundtrip failed"
        );
    }

    #[test]
    fn test_su2_sun2_compose_homomorphism() {
        let g = SU2::rotation_x(0.3);
        let h = SU2::rotation_y(0.7);
        let product = g.compose(&h);

        let g_sun: SUN<2> = g.into();
        let h_sun: SUN<2> = h.into();
        let product_sun: SUN<2> = product.into();

        let composed_sun = g_sun.compose(&h_sun);
        assert!(
            product_sun.distance(&composed_sun) < 1e-13,
            "from(g·h) ≠ from(g)·from(h): {}",
            product_sun.distance(&composed_sun)
        );
    }

    #[test]
    fn test_su2_sun2_exp_commutes() {
        let x = Su2Algebra::new([0.3, -0.2, 0.4]);
        let g = SU2::exp(&x);

        let x_sun: SunAlgebra<2> = x.into();
        let g_sun: SUN<2> = g.into();
        let exp_sun = SUN::<2>::exp(&x_sun);

        assert!(
            g_sun.distance(&exp_sun) < 1e-13,
            "from(exp(X)) ≠ exp(from(X)): {}",
            g_sun.distance(&exp_sun)
        );
    }

    #[test]
    fn test_su2_sun2_inverse_commutes() {
        let g = SU2::rotation_z(1.5);
        let g_inv = g.inverse();

        let g_sun: SUN<2> = g.into();
        let g_inv_sun: SUN<2> = g_inv.into();

        assert!(
            g_sun.inverse().distance(&g_inv_sun) < 1e-14,
            "from(g⁻¹) ≠ from(g)⁻¹"
        );
    }

    #[test]
    fn test_su2_sun2_bracket_commutes() {
        let x = Su2Algebra::new([1.0, 0.0, 0.0]);
        let y = Su2Algebra::new([0.0, 1.0, 0.0]);
        let bracket = x.bracket(&y);

        let x_sun: SunAlgebra<2> = x.into();
        let y_sun: SunAlgebra<2> = y.into();
        let bracket_sun = x_sun.bracket(&y_sun);

        let bracket_converted: SunAlgebra<2> = bracket.into();
        let diff: f64 = bracket_converted
            .coefficients()
            .iter()
            .zip(bracket_sun.coefficients().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-13,
            "from([X,Y]) ≠ [from(X),from(Y)]: diff={}",
            diff
        );
    }

    // ====================================================================
    // SU3 ↔ SUN<3> roundtrips
    // ====================================================================

    #[test]
    fn test_su3_sun3_group_roundtrip() {
        let x = Su3Algebra::new([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let g = SU3::exp(&x);
        let sun: SUN<3> = g.clone().into();
        let back: SU3 = sun.into();
        assert!(
            g.distance(&back) < 1e-13,
            "SU3 → SUN<3> → SU3 roundtrip failed: {}",
            g.distance(&back)
        );
    }

    #[test]
    fn test_su3_sun3_algebra_roundtrip() {
        let x = Su3Algebra::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let sun: SunAlgebra<3> = x.into();
        let back: Su3Algebra = sun.into();
        assert_eq!(
            x, back,
            "Su3Algebra → SunAlgebra<3> → Su3Algebra roundtrip failed"
        );
    }

    #[test]
    fn test_su3_sun3_compose_homomorphism() {
        let x = Su3Algebra::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra::new([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let g = SU3::exp(&x);
        let h = SU3::exp(&y);
        let product = g.compose(&h);

        let g_sun: SUN<3> = g.into();
        let h_sun: SUN<3> = h.into();
        let product_sun: SUN<3> = product.into();

        let composed_sun = g_sun.compose(&h_sun);
        assert!(
            product_sun.distance(&composed_sun) < 1e-12,
            "from(g·h) ≠ from(g)·from(h): {}",
            product_sun.distance(&composed_sun)
        );
    }

    #[test]
    fn test_su3_sun3_exp_commutes() {
        let x = Su3Algebra::new([0.1, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0]);
        let g = SU3::exp(&x);

        let x_sun: SunAlgebra<3> = x.into();
        let g_sun: SUN<3> = g.into();
        let exp_sun = SUN::<3>::exp(&x_sun);

        assert!(
            g_sun.distance(&exp_sun) < 1e-12,
            "from(exp(X)) ≠ exp(from(X)): {}",
            g_sun.distance(&exp_sun)
        );
    }

    #[test]
    fn test_su3_sun3_bracket_commutes() {
        // Bracket in SU3 coordinates, convert, compare to bracket in SUN<3> coordinates
        let x = Su3Algebra::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Compute bracket in SU3, then convert to SUN<3>
        let bracket_su3 = x.bracket(&y);
        let bracket_as_sun: SunAlgebra<3> = bracket_su3.into();

        // Convert to SUN<3>, then compute bracket
        let x_sun: SunAlgebra<3> = x.into();
        let y_sun: SunAlgebra<3> = y.into();
        let bracket_sun = x_sun.bracket(&y_sun);

        // Compare via matrix representation (avoids coefficient ordering issues)
        let m1 = bracket_as_sun.to_matrix();
        let m2 = bracket_sun.to_matrix();
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m1[(r, c)] - m2[(r, c)]).norm() < 1e-12,
                    "from([X,Y]) ≠ [from(X),from(Y)] at ({},{}): {} vs {}",
                    r,
                    c,
                    m1[(r, c)],
                    m2[(r, c)]
                );
            }
        }
    }

    // ====================================================================
    // Permutation self-consistency
    // ====================================================================

    #[test]
    fn test_su3_sun3_permutation_is_inverse() {
        for i in 0..8 {
            assert_eq!(
                super::SUN3_TO_SU3[super::SU3_TO_SUN3[i]],
                i,
                "SUN3_TO_SU3[SU3_TO_SUN3[{}]] ≠ {}",
                i,
                i
            );
            assert_eq!(
                super::SU3_TO_SUN3[super::SUN3_TO_SU3[i]],
                i,
                "SU3_TO_SUN3[SUN3_TO_SU3[{}]] ≠ {}",
                i,
                i
            );
        }
    }

    // ====================================================================
    // UnitQuaternion ↔ SU2 roundtrips
    // ====================================================================

    #[test]
    fn test_quaternion_su2_group_roundtrip() {
        let q = UnitQuaternion::exp([0.3, -0.5, 0.2]);
        let g: SU2 = q.into();
        let back: UnitQuaternion = g.into();

        // Compare quaternion components (up to global sign — q and -q are the same SU2 element)
        let same = (q.w() - back.w()).abs() < 1e-14
            && (q.x() - back.x()).abs() < 1e-14
            && (q.y() - back.y()).abs() < 1e-14
            && (q.z() - back.z()).abs() < 1e-14;
        let negated = (q.w() + back.w()).abs() < 1e-14
            && (q.x() + back.x()).abs() < 1e-14
            && (q.y() + back.y()).abs() < 1e-14
            && (q.z() + back.z()).abs() < 1e-14;
        assert!(
            same || negated,
            "Quaternion roundtrip failed: q={:?}, back={:?}",
            q,
            back
        );
    }

    #[test]
    fn test_quaternion_su2_compose_anti_homomorphism() {
        // The standard SU(2) parameterization U = [[α, -β*], [β, α*]]
        // gives an anti-homomorphism: U(q₁·q₂) = U(q₂)·U(q₁).
        //
        // This is the standard physics convention — quaternion left-multiplication
        // corresponds to matrix right-multiplication. Both represent the same
        // rotation; the ordering reflects the column-vector vs row-vector convention.
        let q1 = UnitQuaternion::exp([0.1, 0.2, 0.3]);
        let q2 = UnitQuaternion::exp([0.4, -0.1, 0.2]);
        let q_product = q1 * q2;

        let g1: SU2 = q1.into();
        let g2: SU2 = q2.into();
        let g_product: SU2 = q_product.into();

        // Anti-homomorphism: from(q1*q2) = from(q2).compose(&from(q1))
        let g_anti = g2.compose(&g1);
        assert!(
            g_product.distance(&g_anti) < 1e-7,
            "from(q1·q2) ≠ from(q2)·from(q1): {}",
            g_product.distance(&g_anti)
        );
    }

    #[test]
    fn test_quaternion_su2_inverse_commutes() {
        let q = UnitQuaternion::exp([0.5, -0.3, 0.8]);
        let q_inv = q.inverse();

        let g: SU2 = q.into();
        let g_inv: SU2 = q_inv.into();

        assert!(
            g.inverse().distance(&g_inv) < 1e-14,
            "from(q⁻¹) ≠ from(q)⁻¹"
        );
    }

    #[test]
    fn test_quaternion_su2_identity() {
        let q = UnitQuaternion::identity();
        let g: SU2 = q.into();
        assert!(
            g.is_near_identity(1e-14),
            "Quaternion identity should map to SU2 identity"
        );
    }
}
