//! Casimir operators for Lie algebras.
//!
//! The Casimir operators are elements of the center of the universal enveloping
//! algebra U(𝔤) that commute with all group generators. They provide crucial
//! invariants for classifying irreducible representations.
//!
//! # Mathematical Background
//!
//! ## Quadratic Casimir Operator
//!
//! The **quadratic Casimir** C₂ is defined as:
//! ```text
//! C₂ = Σᵢ Tᵢ²  (sum over orthonormal basis of 𝔤)
//! ```
//!
//! **Key Property**: In any irreducible representation ρ:
//! ```text
//! ρ(C₂) = c₂(ρ) · I
//! ```
//! where c₂(ρ) is a scalar eigenvalue that labels the representation.
//!
//! ## Physical Interpretation
//!
//! For different Lie groups, C₂ has different physical meanings:
//! - **SU(2)**: C₂ = J² (total angular momentum squared)
//! - **SU(3)**: C₂ labels quark and gluon color representations
//! - **Lorentz group**: C₂ and C₃ label particle mass and spin
//!
//! ## Higher Casimirs
//!
//! For Lie algebras of rank r (dimension of Cartan subalgebra),
//! there are r independent Casimir operators:
//! - Rank 1 (SU(2), SO(3)): only C₂
//! - Rank 2 (SU(3), SO(5)): C₂ and C₃
//! - Rank n (SU(n+1)): C₂, C₃, ..., C_{n+1}
//!
//! # Examples
//!
//! ## SU(2) Eigenvalues
//!
//! For spin-j representation:
//! ```text
//! c₂(j) = j(j+1)
//!
//! j = 0:     c₂ = 0   (scalar)
//! j = 1/2:   c₂ = 3/4 (spinor)
//! j = 1:     c₂ = 2   (vector/adjoint)
//! j = 3/2:   c₂ = 15/4
//! ```
//!
//! ## SU(3) Eigenvalues
//!
//! For representation (p,q):
//! ```text
//! c₂(p,q) = (1/3)(p² + q² + pq + 3p + 3q)
//!
//! (0,0): c₂ = 0     (trivial/singlet)
//! (1,0): c₂ = 4/3   (fundamental/quark)
//! (0,1): c₂ = 4/3   (antifundamental/antiquark)
//! (1,1): c₂ = 3     (adjoint/gluon)
//! ```
//!
//! # References
//!
//! - **Georgi**: "Lie Algebras in Particle Physics" (1999), Chapter 3
//! - **Cahn**: "Semi-Simple Lie Algebras and Their Representations" (1984), Chapter 7
//! - **Gilmore**: "Lie Groups, Lie Algebras, and Some of Their Applications" (1974)

/// Casimir operators for Lie algebras.
///
/// This trait provides methods for computing eigenvalues of Casimir operators
/// in different irreducible representations.
///
/// # Implementation Notes
///
/// For a Lie algebra 𝔤:
/// 1. Implement `quadratic_casimir_eigenvalue()` using the standard formula
/// 2. For rank > 1 algebras, optionally implement `higher_casimir_eigenvalues()`
/// 3. Use exact rational arithmetic where possible (e.g., 4/3 not 1.333...)
///
/// # Type Safety
///
/// This trait is defined as a standalone trait (not requiring `LieAlgebra`)
/// to allow flexible implementation strategies. Implementors should ensure
/// they also implement `LieAlgebra`.
pub trait Casimir {
    /// Type representing irreducible representations of this algebra.
    ///
    /// # Examples
    /// - SU(2): `Spin` (half-integer j)
    /// - SU(3): `Su3Irrep` (Dynkin labels (p,q))
    /// - SU(N): `YoungTableau` or `DynkinLabels`
    type Representation;

    /// Eigenvalue of the quadratic Casimir operator in a given irrep.
    ///
    /// For an irreducible representation ρ, this computes the scalar c₂(ρ)
    /// such that ρ(C₂) = c₂(ρ) · I.
    ///
    /// # Arguments
    ///
    /// * `irrep` - The irreducible representation
    ///
    /// # Returns
    ///
    /// The eigenvalue c₂(ρ) as a real number.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lie_groups::{Su2Algebra, Casimir, Spin};
    ///
    /// // Spin-1/2 (spinor): c₂ = 3/4
    /// let c2 = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::HALF);
    /// assert_eq!(c2, 0.75);
    /// ```
    fn quadratic_casimir_eigenvalue(irrep: &Self::Representation) -> f64;

    /// Eigenvalues of higher Casimir operators (optional).
    ///
    /// For algebras of rank r > 1, there are r - 1 additional independent
    /// Casimir operators beyond C₂.
    ///
    /// # Returns
    ///
    /// A vector of eigenvalues [c₃(ρ), c₄(ρ), ...] for the given irrep.
    /// Default implementation returns empty vector (no higher Casimirs).
    ///
    /// # Examples
    ///
    /// For SU(3) with cubic Casimir C₃:
    /// ```ignore
    /// let higher = Su3Algebra::higher_casimir_eigenvalues(&irrep);
    /// let c3 = higher[0]; // Cubic Casimir eigenvalue
    /// ```
    fn higher_casimir_eigenvalues(_irrep: &Self::Representation) -> Vec<f64> {
        vec![]
    }

    /// Dimension of the Cartan subalgebra (rank of the algebra).
    ///
    /// This determines the number of independent Casimir operators:
    /// - Rank 1: only C₂ (SU(2), SO(3))
    /// - Rank 2: C₂ and C₃ (SU(3), SO(5))
    /// - Rank n: C₂, ..., C_{n+1} (SU(n+1))
    ///
    /// # Returns
    ///
    /// The rank as a positive integer.
    fn rank() -> usize;

    /// Number of independent Casimir operators.
    ///
    /// This equals the rank of the algebra.
    ///
    /// # Returns
    ///
    /// The number of Casimirs (rank).
    fn num_casimirs() -> usize {
        Self::rank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Placeholder test to ensure module compiles
    #[test]
    fn test_trait_compiles() {
        // This test verifies that the Casimir trait is well-formed
        // Actual tests are in the implementation modules (su2.rs, su3.rs)
    }

    /// Test the Casimir identity: ∑_a T^a · T^a = C(R) · I
    ///
    /// This is the fundamental identity relating Casimir eigenvalues to
    /// the sum of squared generators. For a representation R of a Lie algebra:
    /// ```text
    /// ∑_a T^a_{ij} T^a_{jk} = C(R) δ_{ik}
    /// ```
    ///
    /// where T^a are the generators in representation R and C(R) is the
    /// quadratic Casimir eigenvalue.
    ///
    /// For SU(2) fundamental (j=1/2):
    /// - Generators: T^a = σ^a/2 (Pauli matrices / 2)
    /// - Casimir: C_{1/2} = j(j+1) = 3/4
    /// - Identity: (σ^1/2)² + (σ^2/2)² + (σ^3/2)² = 3/4 · I
    #[test]
    fn test_casimir_identity_su2_fundamental() {
        use crate::representation::Spin;
        use crate::su2::Su2Algebra;
        use ndarray::Array2;
        use num_complex::Complex64;

        // SU(2) generators in fundamental representation: T^a = σ^a/2
        // Pauli matrices:
        // σ^1 = [[0, 1], [1, 0]]
        // σ^2 = [[0, -i], [i, 0]]
        // σ^3 = [[1, 0], [0, -1]]

        let sigma1: Array2<Complex64> = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let sigma2: Array2<Complex64> = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let sigma3: Array2<Complex64> = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

        // T^a = σ^a / 2
        let t1 = &sigma1 / Complex64::new(2.0, 0.0);
        let t2 = &sigma2 / Complex64::new(2.0, 0.0);
        let t3 = &sigma3 / Complex64::new(2.0, 0.0);

        // Compute ∑_a T^a · T^a
        let casimir_sum = t1.dot(&t1) + t2.dot(&t2) + t3.dot(&t3);

        // Expected: C_{1/2} · I = (3/4) · I
        let expected_casimir = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::HALF);
        let expected_matrix: Array2<Complex64> =
            Array2::eye(2) * Complex64::new(expected_casimir, 0.0);

        // Verify the identity
        for i in 0..2 {
            for j in 0..2 {
                let diff = (casimir_sum[[i, j]] - expected_matrix[[i, j]]).norm();
                assert!(
                    diff < 1e-10,
                    "Casimir identity violated at ({},{}): got {}, expected {}",
                    i,
                    j,
                    casimir_sum[[i, j]],
                    expected_matrix[[i, j]]
                );
            }
        }

        // Also verify the Casimir value is 3/4
        assert!(
            (expected_casimir - 0.75).abs() < 1e-10,
            "C_{{1/2}} should be 3/4"
        );
    }

    /// Test Casimir identity for SU(2) adjoint representation (j=1).
    ///
    /// For the adjoint representation of SU(2):
    /// - Dimension: 3 (same as dim(su(2)))
    /// - Casimir: `C_1` = j(j+1) = 2
    ///
    /// The generators in the adjoint are related to structure constants.
    /// Using the physics convention with (T^a)_{bc} = -i f^{abc} (antisymmetric),
    /// we verify that ∑_a (T^a)_{bc} (T^a)_{cd} is proportional to δ_{bd}.
    ///
    /// # Convention Note
    ///
    /// The Levi-Civita contraction gives:
    /// ∑_{a,c} ε_{abc} ε_{acd} = -2 δ_{bd}
    ///
    /// The factor of -2 (rather than +2) comes from the antisymmetric
    /// (real) generator convention. The magnitude |C| = 2 matches the
    /// formula `C_j` = j(j+1) = 2 for j=1.
    #[test]
    fn test_casimir_identity_su2_adjoint() {
        use crate::representation::Spin;
        use crate::su2::Su2Algebra;

        // ε tensor (Levi-Civita symbol)
        let epsilon = |a: usize, b: usize, c: usize| -> f64 {
            if (a, b, c) == (0, 1, 2) || (a, b, c) == (1, 2, 0) || (a, b, c) == (2, 0, 1) {
                1.0
            } else if (a, b, c) == (0, 2, 1) || (a, b, c) == (2, 1, 0) || (a, b, c) == (1, 0, 2) {
                -1.0
            } else {
                0.0
            }
        };

        // Build adjoint generators: (t^a)_{bc} = ε_{abc}
        let mut t_adj: [[[f64; 3]; 3]; 3] = [[[0.0; 3]; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    t_adj[a][b][c] = epsilon(a, b, c);
                }
            }
        }

        // Compute sum: (∑_a T^a T^a)_{bd} = ∑_a ∑_c T^a_{bc} T^a_{cd}
        let mut casimir_sum = [[0.0; 3]; 3];
        for b in 0..3 {
            for d in 0..3 {
                for a in 0..3 {
                    for c in 0..3 {
                        casimir_sum[b][d] += t_adj[a][b][c] * t_adj[a][c][d];
                    }
                }
            }
        }

        // Result: ∑_{a,c} ε_{abc} ε_{acd} = -2 δ_{bd}
        // The magnitude matches C_adjoint = 2, sign is negative due to
        // real antisymmetric generator convention.
        let expected_casimir = Su2Algebra::quadratic_casimir_eigenvalue(&Spin::ONE);

        for b in 0..3 {
            for d in 0..3 {
                // Note: sign is negative for real antisymmetric generators
                let expected = if b == d { -expected_casimir } else { 0.0 };
                let diff = (casimir_sum[b][d] - expected).abs();
                assert!(
                    diff < 1e-10,
                    "Adjoint Casimir identity violated at ({},{}): got {}, expected {}",
                    b,
                    d,
                    casimir_sum[b][d],
                    expected
                );
            }
        }

        // Verify diagonal value is -2 (magnitude matches C_1 = 2)
        assert!(
            (casimir_sum[0][0].abs() - expected_casimir).abs() < 1e-10,
            "|∑_a (T^a T^a)_{{00}}| should equal C_adjoint = 2"
        );
    }
}
