//! Representation theory for Lie groups and Lie algebras.
//!
//! This module provides tools for working with irreducible representations (irreps),
//! tensor products, characters, and Casimir operators.
//!
//! # Overview
//!
//! ## Core Concepts
//!
//! - **Irreducible Representations**: Classified by highest weights
//! - **Casimir Operators**: Invariants that label representations
//! - **Characters**: Traces of representation matrices (Weyl character formula)
//! - **Tensor Products**: Clebsch-Gordan / Littlewood-Richardson decomposition
//!
//! ## Implemented Groups
//!
//! - **SU(2)**: Complete (see `crate::representation` module)
//! - **SU(3)**: In progress (Casimir operators implemented)
//! - **SU(N)**: Planned (generic implementation)
//!
//! # Physical Applications
//!
//! Representation theory is essential for:
//! - Classifying particle states in QCD (quarks, gluons, hadrons)
//! - Computing scattering amplitudes (Clebsch-Gordan coefficients)
//! - Understanding symmetry breaking (Higgs mechanism)
//! - Gauge fixing via Peter-Weyl decomposition
//!
//! # References
//!
//! - **Georgi**: "Lie Algebras in Particle Physics" (1999)
//! - **Slansky**: "Group Theory for Unified Model Building" (1981)
//! - **Weyl**: "The Theory of Groups and Quantum Mechanics" (1928)

pub mod casimir;
pub mod su3_irrep;

pub use casimir::Casimir;
pub use su3_irrep::Su3Irrep;

// ============================================================================
// SU(2) Representation Theory
// ============================================================================

use crate::quaternion::UnitQuaternion;
use crate::su2::SU2;

/// Spin quantum number j (half-integer)
///
/// # Mathematical Definition
///
/// j ∈ {0, 1/2, 1, 3/2, 2, ...}
///
/// Internally represented as 2j (integer) to avoid floating point.
///
/// # Physical Meaning
///
/// - **j = 0**: Scalar (spin-0)
/// - **j = 1/2**: Spinor (spin-1/2, fermions)
/// - **j = 1**: Vector (spin-1, bosons)
/// - **j = n/2**: General half-integer spin
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Spin {
    /// Twice the spin: `two_j` = 2j
    /// This allows half-integer spins while using integers
    pub two_j: u32,
}

impl Spin {
    /// Spin-0 (scalar, trivial representation)
    pub const ZERO: Spin = Spin { two_j: 0 };

    /// Spin-1/2 (spinor, fundamental representation)
    pub const HALF: Spin = Spin { two_j: 1 };

    /// Spin-1 (vector, adjoint representation)
    pub const ONE: Spin = Spin { two_j: 2 };

    /// Create from half-integer: j = n/2
    ///
    /// # Examples
    /// - `Spin::from_half_integer(0)` → j = 0
    /// - `Spin::from_half_integer(1)` → j = 1/2
    /// - `Spin::from_half_integer(2)` → j = 1
    #[must_use]
    pub fn from_half_integer(two_j: u32) -> Self {
        Spin { two_j }
    }

    /// Create from integer spin (convenience)
    #[must_use]
    pub fn from_integer(j: u32) -> Self {
        Spin { two_j: 2 * j }
    }

    /// Get the spin value as f64: j
    #[must_use]
    pub fn value(&self) -> f64 {
        f64::from(self.two_j) / 2.0
    }

    /// Get dimension of this representation: 2j + 1
    #[must_use]
    pub fn dimension(&self) -> usize {
        (self.two_j + 1) as usize
    }

    /// Check if this is integer spin (boson)
    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.two_j % 2 == 0
    }

    /// Check if this is half-integer spin (fermion)
    #[must_use]
    pub fn is_half_integer(&self) -> bool {
        self.two_j % 2 == 1
    }
}

/// Character of a representation: χⱼ(g) = Tr(D^j(g))
///
/// # Mathematical Formula
///
/// For g = exp(iθ n⃗·σ⃗/2), the character is:
/// ```text
/// χⱼ(θ) = sin((2j+1)θ/2) / sin(θ/2)
/// ```
///
/// This is the **Weyl character formula** for SU(2).
///
/// # Properties
///
/// 1. **Class function**: χⱼ(hgh⁻¹) = χⱼ(g)
/// 2. **Orthogonality**: ∫ χⱼ(g) χₖ(g)* dg = δⱼₖ
/// 3. **Completeness**: Σⱼ χⱼ(g) χⱼ(h)* = δ(g, h)
///
/// # Special Values
///
/// - χⱼ(e) = 2j + 1 (dimension at identity)
/// - χⱼ(θ=0) = 2j + 1
/// - χⱼ(θ=π) = sin((2j+1)π/2) / sin(π/2)
/// - χⱼ(θ=2π) = (-1)^{2j} × (2j+1) (double cover!)
#[must_use]
pub fn character(spin: Spin, angle: f64) -> f64 {
    use std::f64::consts::PI;

    let j = spin.value();
    let two_j = spin.two_j;
    let dim = 2.0 * j + 1.0;

    // Find which multiple of 2π we're nearest to
    // θ ≈ 2nπ has a singularity where sin(θ/2) = sin(nπ) ≈ 0
    let n = (angle / (2.0 * PI)).round();
    let near_multiple = (angle - n * 2.0 * PI).abs() < 1e-10;

    if near_multiple {
        // At θ = 2nπ, use the limit formula:
        // χⱼ(2nπ) = (-1)^{2jn} × (2j+1)
        //
        // For SU(2) double cover:
        // - n even: g = I, χ = dim
        // - n odd: g = -I, χ = (-1)^{2j} × dim
        let n_int = n as i64;
        if n_int % 2 == 0 {
            return dim;
        }
        let phase = if two_j % 2 == 0 { 1.0 } else { -1.0 };
        return phase * dim;
    }

    // χⱼ(θ) = sin((2j+1)θ/2) / sin(θ/2)
    let numerator = ((2.0 * j + 1.0) * angle / 2.0).sin();
    let denominator = (angle / 2.0).sin();

    // Handle near-zero denominator (numerical safety)
    if denominator.abs() < 1e-10 {
        // Use L'Hôpital's rule: limit is (2j+1) × cos((2j+1)θ/2) / cos(θ/2)
        let num_deriv = (2.0 * j + 1.0) * ((2.0 * j + 1.0) * angle / 2.0).cos();
        let den_deriv = (angle / 2.0).cos();
        if den_deriv.abs() < 1e-10 {
            return dim; // Fallback
        }
        return num_deriv / den_deriv;
    }

    numerator / denominator
}

/// Character of an SU(2) group element
///
/// # Algorithm
///
/// 1. Convert SU(2) to quaternion
/// 2. Extract rotation angle θ
/// 3. Compute χⱼ(θ) = sin((2j+1)θ/2) / sin(θ/2)
#[must_use]
pub fn character_su2(spin: Spin, g: &SU2) -> f64 {
    // Convert to quaternion and get rotation angle
    let matrix_array = g.to_matrix_array();
    let quat = UnitQuaternion::from_matrix(matrix_array);
    let (_axis, angle) = quat.to_axis_angle();

    character(spin, angle)
}

/// Clebsch-Gordan decomposition: Vⱼ₁ ⊗ Vⱼ₂ = ⨁ₖ Vₖ
///
/// # Mathematical Formula
///
/// The tensor product of representations j₁ and j₂ decomposes as:
/// ```text
/// j₁ ⊗ j₂ = ⨁ₖ k
/// ```
/// where k ranges from |j₁ - j₂| to j₁ + j₂ in integer steps.
///
/// # Examples
///
/// - 1/2 ⊗ 1/2 = 0 ⊕ 1 (spinor × spinor = scalar + vector)
/// - 1/2 ⊗ 1 = 1/2 ⊕ 3/2 (spinor × vector)
/// - 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2 (vector × vector)
///
/// # Physical Meaning
///
/// This is how angular momenta combine in quantum mechanics:
/// - Two spin-1/2 particles combine to give spin-0 (singlet) or spin-1 (triplet)
/// - This explains atomic spectra and chemical bonding
#[must_use]
pub fn clebsch_gordan_decomposition(j1: Spin, j2: Spin) -> Vec<Spin> {
    let min_k = (j1.two_j as i32 - j2.two_j as i32).unsigned_abs();
    let max_k = j1.two_j + j2.two_j;

    let mut result = Vec::new();

    // k ranges from |j₁ - j₂| to j₁ + j₂ in steps of 2
    // (steps of 2 because two_j representation)
    let mut k = min_k;
    while k <= max_k {
        result.push(Spin::from_half_integer(k));
        k += 2;
    }

    result
}

/// Character orthogonality relation (analytical)
///
/// # Mathematical Statement
///
/// Characters of different irreps are orthogonal:
/// ```text
/// ∫_{SU(2)} χⱼ(g) χₖ(g)* dg = δⱼₖ
/// ```
///
/// For SU(2), the Haar measure integral gives:
/// ```text
/// ∫₀^{2π} χⱼ(θ) χₖ(θ) sin²(θ/2) dθ / π = δⱼₖ
/// ```
///
/// # Peter-Weyl Theorem
///
/// This orthogonality is fundamental to the Peter-Weyl theorem,
/// which states that irreps form a complete orthonormal basis for L²(G).
#[must_use]
pub fn character_orthogonality_delta(j1: Spin, j2: Spin) -> bool {
    // Characters are orthogonal: different spins give 0, same spin gives 1
    // This is an analytical fact from representation theory
    j1 == j2
}

/// Dimension formula: dim(Vⱼ) = 2j + 1
///
/// This is the number of basis states in the spin-j representation.
///
/// # Examples
///
/// - j = 0: 1 state (scalar)
/// - j = 1/2: 2 states (|↑⟩, |↓⟩)
/// - j = 1: 3 states (|+1⟩, |0⟩, |-1⟩)
#[must_use]
pub fn representation_dimension(spin: Spin) -> usize {
    spin.dimension()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_spin_values() {
        assert_eq!(Spin::ZERO.value(), 0.0);
        assert_eq!(Spin::HALF.value(), 0.5);
        assert_eq!(Spin::ONE.value(), 1.0);
        assert_eq!(Spin::from_half_integer(3).value(), 1.5);
    }

    #[test]
    fn test_dimension_formula() {
        // dim(j) = 2j + 1
        assert_eq!(Spin::ZERO.dimension(), 1); // Scalar
        assert_eq!(Spin::HALF.dimension(), 2); // Spinor
        assert_eq!(Spin::ONE.dimension(), 3); // Vector
        assert_eq!(Spin::from_half_integer(3).dimension(), 4); // j=3/2
    }

    #[test]
    fn test_integer_vs_half_integer() {
        // Integer spins (bosons): j = 0, 1, 2, ...
        assert!(Spin::ZERO.is_integer());
        assert!(Spin::ONE.is_integer());
        assert!(Spin::from_integer(2).is_integer());

        // Half-integer spins (fermions): j = 1/2, 3/2, 5/2, ...
        assert!(Spin::HALF.is_half_integer());
        assert!(Spin::from_half_integer(3).is_half_integer()); // j = 3/2
        assert!(Spin::from_half_integer(5).is_half_integer()); // j = 5/2
    }

    #[test]
    fn test_character_at_identity() {
        // χⱼ(e) = 2j + 1
        assert_eq!(character(Spin::ZERO, 0.0), 1.0);
        assert_eq!(character(Spin::HALF, 0.0), 2.0);
        assert_eq!(character(Spin::ONE, 0.0), 3.0);
    }

    #[test]
    fn test_character_at_2pi() {
        // χⱼ(2π) = (-1)^{2j}
        let angle = 2.0 * PI;

        // j = 0: (-1)^0 = 1
        assert!((character(Spin::ZERO, angle) - 1.0).abs() < 1e-10);

        // j = 1/2: (-1)^1 = -1
        assert!((character(Spin::HALF, angle) - (-2.0)).abs() < 1e-10);

        // j = 1: (-1)^2 = 1
        assert!((character(Spin::ONE, angle) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clebsch_gordan_spinor_times_spinor() {
        // 1/2 ⊗ 1/2 = 0 ⊕ 1
        let decomp = clebsch_gordan_decomposition(Spin::HALF, Spin::HALF);
        assert_eq!(decomp.len(), 2);
        assert_eq!(decomp[0], Spin::ZERO);
        assert_eq!(decomp[1], Spin::ONE);
    }

    #[test]
    fn test_clebsch_gordan_spinor_times_vector() {
        // 1/2 ⊗ 1 = 1/2 ⊕ 3/2
        let decomp = clebsch_gordan_decomposition(Spin::HALF, Spin::ONE);
        assert_eq!(decomp.len(), 2);
        assert_eq!(decomp[0], Spin::HALF);
        assert_eq!(decomp[1], Spin::from_half_integer(3));
    }

    #[test]
    fn test_clebsch_gordan_vector_times_vector() {
        // 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
        let decomp = clebsch_gordan_decomposition(Spin::ONE, Spin::ONE);
        assert_eq!(decomp.len(), 3);
        assert_eq!(decomp[0], Spin::ZERO);
        assert_eq!(decomp[1], Spin::ONE);
        assert_eq!(decomp[2], Spin::from_integer(2));
    }

    #[test]
    fn test_character_su2_identity() {
        let identity = SU2::identity();
        let chi = character_su2(Spin::ONE, &identity);

        // At identity, χⱼ = 2j + 1 = 3 for j=1
        assert!((chi - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_counts() {
        // Verify dim = 2j + 1 for several spins
        assert_eq!(representation_dimension(Spin::ZERO), 1);
        assert_eq!(representation_dimension(Spin::HALF), 2);
        assert_eq!(representation_dimension(Spin::ONE), 3);
        assert_eq!(representation_dimension(Spin::from_half_integer(3)), 4);
        assert_eq!(representation_dimension(Spin::from_integer(2)), 5);
    }

    #[test]
    fn test_character_formula_j_equals_half() {
        // For j = 1/2: χ(θ) = 2cos(θ/2)
        let angles = [0.0, PI / 4.0, PI / 2.0, PI];

        for &angle in &angles {
            let chi = character(Spin::HALF, angle);
            let expected = 2.0 * (angle / 2.0).cos();
            assert!(
                (chi - expected).abs() < 1e-10,
                "j=1/2: χ({angle}) = {chi}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_character_formula_j_equals_one() {
        // For j = 1: χ(θ) = 1 + 2cos(θ)
        let angles = [0.0, PI / 3.0, PI / 2.0, PI];

        for &angle in &angles {
            let chi = character(Spin::ONE, angle);
            let expected = 1.0 + 2.0 * angle.cos();
            assert!(
                (chi - expected).abs() < 1e-10,
                "j=1: χ({angle}) = {chi}, expected {expected}"
            );
        }
    }
}
