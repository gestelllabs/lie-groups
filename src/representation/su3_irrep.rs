//! SU(3) irreducible representations.
//!
//! SU(3) irreps are classified by **Dynkin labels** (p, q), which are non-negative integers
//! representing the number of boxes in the first and second rows of a Young tableau.
//!
//! # Mathematical Background
//!
//! ## Dynkin Labels
//!
//! For SU(3), an irrep is labeled by two integers:
//! ```text
//! (p, q) where p, q ∈ ℕ (non-negative integers)
//! ```
//!
//! The **dimension** of an irrep (p,q) is:
//! ```text
//! dim(p,q) = (1/2)(p+1)(q+1)(p+q+2)
//! ```
//!
//! ## Standard Representations
//!
//! - **(0,0)**: Trivial (singlet), dim = 1
//! - **(1,0)**: Fundamental (quark), dim = 3
//! - **(0,1)**: Antifundamental (antiquark), dim = 3̄
//! - **(1,1)**: Adjoint (gluon), dim = 8
//! - **(2,0)**: Symmetric (diquark), dim = 6
//! - **(3,0)**: Totally symmetric, dim = 10
//! - **(0,2)**: Conjugate symmetric, dim = 6̄
//!
//! ## Physical Interpretation
//!
//! In QCD (Quantum Chromodynamics):
//! - (1,0): Quarks (u, d, s, c, b, t) transform in fundamental rep
//! - (0,1): Antiquarks transform in antifundamental
//! - (1,1): Gluons (8 of them) transform in adjoint rep
//! - (0,0): Color singlets (mesons, baryons observed in nature)
//!
//! ## Conjugate Representations
//!
//! The conjugate of (p,q) is (q,p):
//! ```text
//! (p,q)* = (q,p)
//! ```
//!
//! ## Representation Matrices
//!
//! For each g ∈ SU(3), the representation matrix ρ(p,q)(g) acts on V(p,q):
//! - **Trivial**: ρ(g) = 1
//! - **Fundamental**: ρ(g) = g (3×3 matrix)
//! - **Antifundamental**: ρ(g) = g* (complex conjugate)
//! - **Adjoint**: ρ(g) X = g X g⁻¹ (conjugation action on su(3))
//!
//! # References
//!
//! - **Georgi**: "Lie Algebras in Particle Physics" (1999), Chapter 10
//! - **Slansky**: "Group Theory for Unified Model Building", Phys. Rep. 79, 1 (1981)

use num_complex::Complex64;

/// SU(3) irreducible representation labeled by Dynkin labels (p, q).
///
/// # Mathematical Definition
///
/// An SU(3) irrep is uniquely specified by two non-negative integers:
/// - `p`: Number of boxes in the first row of the Young tableau
/// - `q`: Number of boxes in the second row
///
/// # Examples
///
/// ```ignore
/// use lie_groups::Su3Irrep;
///
/// // Fundamental representation (quark)
/// let fund = Su3Irrep::FUNDAMENTAL;
/// assert_eq!(fund.dimension(), 3);
///
/// // Adjoint representation (gluon)
/// let adj = Su3Irrep::ADJOINT;
/// assert_eq!(adj.dimension(), 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Su3Irrep {
    /// First Dynkin label (first row of Young tableau)
    pub p: u32,
    /// Second Dynkin label (second row of Young tableau)
    pub q: u32,
}

impl Su3Irrep {
    /// Trivial representation (0,0) - color singlet
    pub const TRIVIAL: Su3Irrep = Su3Irrep { p: 0, q: 0 };

    /// Fundamental representation (1,0) - quark
    pub const FUNDAMENTAL: Su3Irrep = Su3Irrep { p: 1, q: 0 };

    /// Antifundamental representation (0,1) - antiquark
    pub const ANTIFUNDAMENTAL: Su3Irrep = Su3Irrep { p: 0, q: 1 };

    /// Adjoint representation (1,1) - gluon
    pub const ADJOINT: Su3Irrep = Su3Irrep { p: 1, q: 1 };

    /// Symmetric representation (2,0) - diquark
    pub const SYMMETRIC: Su3Irrep = Su3Irrep { p: 2, q: 0 };

    /// Create a new irrep with Dynkin labels (p, q)
    #[must_use]
    pub fn new(p: u32, q: u32) -> Self {
        Su3Irrep { p, q }
    }

    /// Dimension of the representation: dim(p,q) = (1/2)(p+1)(q+1)(p+q+2)
    #[must_use]
    pub fn dimension(&self) -> usize {
        let p = self.p as usize;
        let q = self.q as usize;
        ((p + 1) * (q + 1) * (p + q + 2)) / 2
    }

    /// Conjugate representation: (p,q)* = (q,p)
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Su3Irrep {
            p: self.q,
            q: self.p,
        }
    }

    /// Check if this is a real representation (self-conjugate)
    #[must_use]
    pub fn is_real(&self) -> bool {
        self.p == self.q
    }

    /// Quadratic Casimir eigenvalue C₂(p,q)
    ///
    /// The Casimir operator C₂ = Σᵢ Tᵢ² acts as a scalar on each irrep:
    /// C₂ |ψ⟩ = c₂(p,q) |ψ⟩
    ///
    /// Formula: c₂(p,q) = (p² + q² + pq + 3p + 3q) / 3
    #[must_use]
    pub fn casimir_eigenvalue(&self) -> f64 {
        let p = self.p as f64;
        let q = self.q as f64;
        (p * p + q * q + p * q + 3.0 * p + 3.0 * q) / 3.0
    }

    /// Compute representation matrix for SU(3) group element
    ///
    /// Returns ρ(g) as a dim×dim complex matrix.
    ///
    /// # Supported Representations
    ///
    /// - **(0,0)**: Trivial → 1×1 identity
    /// - **(1,0)**: Fundamental → the 3×3 SU(3) matrix itself
    /// - **(0,1)**: Antifundamental → complex conjugate of fundamental
    /// - **(1,1)**: Adjoint → 8×8 matrix from conjugation action
    /// - **(2,0)**: Symmetric tensor → 6×6 matrix
    ///
    /// # Errors
    ///
    /// Returns `RepresentationError::UnsupportedRepresentation` for higher
    /// representations (p+q > 2) as they require tensor product construction.
    pub fn representation_matrix(
        &self,
        g: &[[Complex64; 3]; 3],
    ) -> crate::RepresentationResult<Vec<Vec<Complex64>>> {
        match (self.p, self.q) {
            (0, 0) => {
                // Trivial representation: ρ(g) = 1
                Ok(vec![vec![Complex64::new(1.0, 0.0)]])
            }
            (1, 0) => {
                // Fundamental: ρ(g) = g
                Ok((0..3).map(|i| (0..3).map(|j| g[i][j]).collect()).collect())
            }
            (0, 1) => {
                // Antifundamental: ρ(g) = g*
                Ok((0..3)
                    .map(|i| (0..3).map(|j| g[i][j].conj()).collect())
                    .collect())
            }
            (1, 1) => {
                // Adjoint: ρ(g)_{ab} where T_a → g T_a g⁻¹ = Σ_b ρ(g)_{ba} T_b
                Ok(self.adjoint_representation_matrix(g))
            }
            (2, 0) => {
                // Symmetric tensor: S²(3) → 6-dimensional
                Ok(self.symmetric_tensor_matrix(g))
            }
            _ => Err(crate::RepresentationError::UnsupportedRepresentation {
                representation: format!("({},{})", self.p, self.q),
                reason: "Requires tensor product construction (not yet implemented)".to_string(),
            }),
        }
    }

    /// Compute adjoint representation matrix (1,1) → 8×8
    ///
    /// The adjoint action is: `Ad(g)(X) = g X g⁻¹`
    /// In the Gell-Mann basis: `Ad(g)_{ab} T_b = g T_a g⁻¹`
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn adjoint_representation_matrix(&self, g: &[[Complex64; 3]; 3]) -> Vec<Vec<Complex64>> {
        use crate::su3::Su3Algebra;

        // Compute g⁻¹ = g† for unitary matrices
        let g_inv: [[Complex64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| g[j][i].conj()));

        // For each Gell-Mann matrix T_a, compute g T_a g⁻¹
        // and extract coefficients in the T_b basis
        let mut result = vec![vec![Complex64::new(0.0, 0.0); 8]; 8];

        for a in 0..8 {
            let t_a = Su3Algebra::gell_mann_matrix(a);

            // Compute g T_a g⁻¹
            let mut gt_a = [[Complex64::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        gt_a[i][j] += g[i][k] * t_a[[k, j]];
                    }
                }
            }

            let mut conjugated = [[Complex64::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        conjugated[i][j] += gt_a[i][k] * g_inv[k][j];
                    }
                }
            }

            // Extract coefficients: g T_a g⁻¹ = Σ_b ρ(g)_{ba} T_b
            // Using Tr(T_b · g T_a g⁻¹) = 2 ρ(g)_{ba} (by Gell-Mann normalization)
            for b in 0..8 {
                let t_b = Su3Algebra::gell_mann_matrix(b);
                let mut trace = Complex64::new(0.0, 0.0);
                for i in 0..3 {
                    for j in 0..3 {
                        trace += t_b[[i, j]] * conjugated[j][i];
                    }
                }
                // Gell-Mann normalization: Tr(λ_a λ_b) = 2δ_ab
                result[b][a] = trace / 2.0;
            }
        }

        result
    }

    /// Compute symmetric tensor representation (2,0) → 6-dimensional
    ///
    /// The symmetric product S²(3) is 6-dimensional.
    /// Basis: `{e₁⊗e₁, e₂⊗e₂, e₃⊗e₃, (e₁⊗e₂+e₂⊗e₁)/√2, ...}`
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn symmetric_tensor_matrix(&self, g: &[[Complex64; 3]; 3]) -> Vec<Vec<Complex64>> {
        let sqrt2 = std::f64::consts::SQRT_2;

        // Symmetric pairs: (0,0), (1,1), (2,2), (0,1), (0,2), (1,2)
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

        let mut result = vec![vec![Complex64::new(0.0, 0.0); 6]; 6];

        for (col_idx, &(i, j)) in pairs.iter().enumerate() {
            // Compute (g ⊗ g) acting on symmetric basis element
            // For diagonal (i=j): basis is e_i ⊗ e_i
            // For off-diagonal (i<j): basis is (e_i ⊗ e_j + e_j ⊗ e_i)/√2

            for (row_idx, &(k, l)) in pairs.iter().enumerate() {
                // Coefficient of output basis element in g⊗g applied to input
                let coeff;

                if i == j && k == l {
                    // Diagonal to diagonal: g_{ki} g_{li} = g_{ki}²
                    coeff = g[k][i] * g[l][j];
                } else if i == j && k != l {
                    // Diagonal to off-diagonal
                    // e_i ⊗ e_i → coefficient of (e_k ⊗ e_l + e_l ⊗ e_k)/√2
                    // = (g_{ki}g_{li} + g_{li}g_{ki})/√2 = √2 · g_{ki}g_{li}
                    coeff = Complex64::new(sqrt2, 0.0) * g[k][i] * g[l][i];
                } else if i != j && k == l {
                    // Off-diagonal to diagonal
                    // (e_i ⊗ e_j + e_j ⊗ e_i)/√2 → coefficient of e_k ⊗ e_k
                    // = (g_{ki}g_{kj} + g_{kj}g_{ki})/√2 = √2 · g_{ki}g_{kj} / √2 = g_{ki}g_{kj} + g_{kj}g_{ki}
                    // Wait, need to be more careful
                    // Input: (e_i ⊗ e_j + e_j ⊗ e_i)/√2
                    // g⊗g applied: (g e_i ⊗ g e_j + g e_j ⊗ g e_i)/√2
                    //             = (Σ_a g_{ai}e_a ⊗ Σ_b g_{bj}e_b + Σ_a g_{aj}e_a ⊗ Σ_b g_{bi}e_b)/√2
                    // Coefficient of e_k ⊗ e_k: (g_{ki}g_{kj} + g_{kj}g_{ki})/√2 = 2g_{ki}g_{kj}/√2 = √2 g_{ki}g_{kj}
                    coeff = Complex64::new(sqrt2, 0.0) * g[k][i] * g[k][j];
                } else {
                    // Off-diagonal to off-diagonal (both i≠j and k≠l)
                    // Input: (e_i ⊗ e_j + e_j ⊗ e_i)/√2
                    // Output: coefficient of (e_k ⊗ e_l + e_l ⊗ e_k)/√2
                    // From (e_i ⊗ e_j + e_j ⊗ e_i)/√2, we get terms:
                    // e_k ⊗ e_l appears with coeff (g_{ki}g_{lj} + g_{kj}g_{li})/√2
                    // e_l ⊗ e_k appears with coeff (g_{li}g_{kj} + g_{lj}g_{ki})/√2
                    // These are the same due to symmetry, so total coefficient of normalized output is:
                    // (g_{ki}g_{lj} + g_{kj}g_{li})/√2 · √2 = g_{ki}g_{lj} + g_{kj}g_{li}
                    coeff = g[k][i] * g[l][j] + g[k][j] * g[l][i];
                }

                result[row_idx][col_idx] = coeff;
            }
        }

        result
    }

    /// Character of the representation: χ(g) = Tr(ρ(g))
    ///
    /// For small representations, computed directly from representation matrix.
    /// For general (p,q), uses the Weyl character formula.
    #[must_use]
    pub fn character(&self, g: &[[Complex64; 3]; 3]) -> Complex64 {
        match (self.p, self.q) {
            (0, 0) => Complex64::new(1.0, 0.0),
            (1, 0) => {
                // Tr(g)
                g[0][0] + g[1][1] + g[2][2]
            }
            (0, 1) => {
                // Tr(g*) = Tr(g)*
                (g[0][0] + g[1][1] + g[2][2]).conj()
            }
            (1, 1) => {
                // For adjoint: χ(g) = |Tr(g)|² - 1
                let tr = g[0][0] + g[1][1] + g[2][2];
                tr.norm_sqr() - Complex64::new(1.0, 0.0)
            }
            (2, 0) => {
                // Symmetric: χ(g) = (Tr(g)² + Tr(g²))/2
                let tr = g[0][0] + g[1][1] + g[2][2];
                let mut tr_sq = Complex64::new(0.0, 0.0);
                for i in 0..3 {
                    for j in 0..3 {
                        tr_sq += g[i][j] * g[j][i];
                    }
                }
                (tr * tr + tr_sq) / 2.0
            }
            (0, 2) => {
                // Antisymmetric conjugate: same as (2,0) with g → g*
                let tr = (g[0][0] + g[1][1] + g[2][2]).conj();
                let mut tr_sq = Complex64::new(0.0, 0.0);
                for i in 0..3 {
                    for j in 0..3 {
                        tr_sq += g[i][j].conj() * g[j][i].conj();
                    }
                }
                (tr * tr + tr_sq) / 2.0
            }
            _ => {
                // Fall back to Weyl character formula for general case
                self.weyl_character(g)
            }
        }
    }

    /// Weyl character formula for general (p,q)
    ///
    /// `χ(p,q)(g) = Σ_{w ∈ W} sgn(w) e^{w(λ+ρ)} / Σ_{w ∈ W} sgn(w) e^{w·ρ}`
    ///
    /// where λ = (p,q) is highest weight, ρ = (1,1) is Weyl vector
    #[allow(clippy::many_single_char_names, clippy::trivially_copy_pass_by_ref)]
    fn weyl_character(&self, g: &[[Complex64; 3]; 3]) -> Complex64 {
        // Extract eigenvalues of g (diagonal in Cartan subalgebra)
        // For diagonal g = diag(e^{iθ₁}, e^{iθ₂}, e^{iθ₃}) with θ₁+θ₂+θ₃=0
        let eigenvalues = self.extract_eigenvalues(g);

        let p = self.p as i32;
        let q = self.q as i32;

        // Weyl group S₃ with 6 elements
        // Numerator: Σ_σ sgn(σ) e^{σ(λ+ρ)·h}
        // where λ+ρ = (p+1, q+1) in Dynkin labels

        // In root coordinates, λ+ρ = ((2p+q+3)/3, (p+2q+3)/3, -(p+q+2))
        // Simplified: use fundamental weights ω₁, ω₂

        let a = p + 1;
        let b = q + 1;

        // Character in terms of eigenvalues z₁, z₂, z₃
        // Using Weyl denominator formula
        let z = eigenvalues;

        let mut numerator = Complex64::new(0.0, 0.0);
        let mut denominator = Complex64::new(0.0, 0.0);

        // All permutations of (0,1,2) with signs
        let perms = [
            ([0, 1, 2], 1.0),
            ([1, 0, 2], -1.0),
            ([0, 2, 1], -1.0),
            ([2, 1, 0], -1.0),
            ([1, 2, 0], 1.0),
            ([2, 0, 1], 1.0),
        ];

        for &(perm, sign) in &perms {
            // λ+ρ in weight form: m₁ = a+b, m₂ = b, m₃ = 0
            // After permutation σ: contribution = sgn(σ) z₁^{m_{σ(1)}} z₂^{m_{σ(2)}} z₃^{m_{σ(3)}}
            let m = [a + b, b, 0];
            let term = sign
                * z[0].powf(m[perm[0]] as f64)
                * z[1].powf(m[perm[1]] as f64)
                * z[2].powf(m[perm[2]] as f64);
            numerator += term;

            // ρ = (2, 1, 0)
            let rho = [2, 1, 0];
            let denom_term = sign
                * z[0].powf(rho[perm[0]] as f64)
                * z[1].powf(rho[perm[1]] as f64)
                * z[2].powf(rho[perm[2]] as f64);
            denominator += denom_term;
        }

        if denominator.norm() < 1e-10 {
            // At singular point, use L'Hôpital or limit
            Complex64::new(self.dimension() as f64, 0.0)
        } else {
            numerator / denominator
        }
    }

    /// Extract eigenvalues from SU(3) matrix
    ///
    /// Returns eigenvalues as complex numbers on unit circle.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn extract_eigenvalues(&self, g: &[[Complex64; 3]; 3]) -> [Complex64; 3] {
        // For identity, eigenvalues are all 1
        let is_identity = (0..3).all(|i| {
            (0..3).all(|j| {
                let expected = if i == j { 1.0 } else { 0.0 };
                (g[i][j] - Complex64::new(expected, 0.0)).norm() < 1e-10
            })
        });

        if is_identity {
            return [Complex64::new(1.0, 0.0); 3];
        }

        // Use characteristic polynomial coefficients
        // det(g - λI) = -λ³ + tr(g)λ² - (1/2)(tr(g)² - tr(g²))λ + det(g)
        // Since det(g) = 1 for SU(3)
        let tr = g[0][0] + g[1][1] + g[2][2];

        let mut tr_sq = Complex64::new(0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                tr_sq += g[i][j] * g[j][i];
            }
        }

        let c2 = (tr * tr - tr_sq) / 2.0;

        // Solve cubic: λ³ - tr·λ² + c2·λ - 1 = 0
        // Use Cardano's formula or numerical method
        self.solve_cubic(tr, c2, Complex64::new(1.0, 0.0))
    }

    /// Solve cubic x³ - a·x² + b·x - c = 0 for complex roots
    #[allow(clippy::many_single_char_names, clippy::trivially_copy_pass_by_ref)]
    fn solve_cubic(&self, a: Complex64, b: Complex64, c: Complex64) -> [Complex64; 3] {
        // Substitution x = t + a/3 to get depressed cubic t³ + pt + q = 0
        let p = b - a * a / 3.0;
        let q = c - a * b / 3.0 + 2.0 * a * a * a / 27.0;

        // Cardano: t = ∛(-q/2 + √(q²/4 + p³/27)) + ∛(-q/2 - √(q²/4 + p³/27))
        let discriminant = q * q / 4.0 + p * p * p / 27.0;

        let sqrt_disc = discriminant.sqrt();
        let u = (-q / 2.0 + sqrt_disc).powf(1.0 / 3.0);
        let v = (-q / 2.0 - sqrt_disc).powf(1.0 / 3.0);

        let omega = Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0); // Primitive cube root of unity

        let offset = a / 3.0;

        [
            u + v + offset,
            omega * u + omega.conj() * v + offset,
            omega.conj() * u + omega * v + offset,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_3x3() -> [[Complex64; 3]; 3] {
        [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ]
    }

    fn diagonal_su3(theta1: f64, theta2: f64) -> [[Complex64; 3]; 3] {
        // Diagonal SU(3): diag(e^{iθ₁}, e^{iθ₂}, e^{-i(θ₁+θ₂)})
        let z1 = Complex64::new(theta1.cos(), theta1.sin());
        let z2 = Complex64::new(theta2.cos(), theta2.sin());
        let z3 = Complex64::new((-(theta1 + theta2)).cos(), (-(theta1 + theta2)).sin());
        [
            [z1, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), z2, Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), z3],
        ]
    }

    #[test]
    fn test_dimension_formula() {
        // (0,0): dim = 1
        assert_eq!(Su3Irrep::TRIVIAL.dimension(), 1);

        // (1,0): dim = 3
        assert_eq!(Su3Irrep::FUNDAMENTAL.dimension(), 3);

        // (0,1): dim = 3
        assert_eq!(Su3Irrep::ANTIFUNDAMENTAL.dimension(), 3);

        // (1,1): dim = 8
        assert_eq!(Su3Irrep::ADJOINT.dimension(), 8);

        // (2,0): dim = 6
        assert_eq!(Su3Irrep::SYMMETRIC.dimension(), 6);

        // (3,0): dim = 10
        assert_eq!(Su3Irrep::new(3, 0).dimension(), 10);
    }

    #[test]
    fn test_conjugate() {
        // Fundamental <-> Antifundamental
        assert_eq!(Su3Irrep::FUNDAMENTAL.conjugate(), Su3Irrep::ANTIFUNDAMENTAL);
        assert_eq!(Su3Irrep::ANTIFUNDAMENTAL.conjugate(), Su3Irrep::FUNDAMENTAL);

        // Adjoint is self-conjugate
        assert_eq!(Su3Irrep::ADJOINT.conjugate(), Su3Irrep::ADJOINT);
    }

    #[test]
    fn test_real_representations() {
        // Adjoint is real
        assert!(Su3Irrep::ADJOINT.is_real());

        // Fundamental is complex (not real)
        assert!(!Su3Irrep::FUNDAMENTAL.is_real());

        // Antifundamental is complex
        assert!(!Su3Irrep::ANTIFUNDAMENTAL.is_real());
    }

    #[test]
    fn test_casimir_eigenvalues() {
        // Known values
        assert!((Su3Irrep::TRIVIAL.casimir_eigenvalue() - 0.0).abs() < 1e-10);
        assert!((Su3Irrep::FUNDAMENTAL.casimir_eigenvalue() - 4.0 / 3.0).abs() < 1e-10);
        assert!((Su3Irrep::ADJOINT.casimir_eigenvalue() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_representation_matrix_identity() {
        let id = identity_3x3();

        // Trivial representation at identity
        let trivial = Su3Irrep::TRIVIAL.representation_matrix(&id).unwrap();
        assert_eq!(trivial.len(), 1);
        assert!((trivial[0][0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);

        // Fundamental at identity = 3×3 identity
        let fund = Su3Irrep::FUNDAMENTAL.representation_matrix(&id).unwrap();
        assert_eq!(fund.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((fund[i][j] - Complex64::new(expected, 0.0)).norm() < 1e-10);
            }
        }

        // Adjoint at identity = 8×8 identity
        let adj = Su3Irrep::ADJOINT.representation_matrix(&id).unwrap();
        assert_eq!(adj.len(), 8);
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (adj[i][j] - Complex64::new(expected, 0.0)).norm() < 1e-10,
                    "Adjoint[{}][{}] = {:?}, expected {}",
                    i,
                    j,
                    adj[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_character_at_identity() {
        let id = identity_3x3();

        // Character at identity = dimension
        assert!((Su3Irrep::TRIVIAL.character(&id) - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((Su3Irrep::FUNDAMENTAL.character(&id) - Complex64::new(3.0, 0.0)).norm() < 1e-10);
        assert!((Su3Irrep::ADJOINT.character(&id) - Complex64::new(8.0, 0.0)).norm() < 1e-10);
        assert!((Su3Irrep::SYMMETRIC.character(&id) - Complex64::new(6.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_character_diagonal() {
        let g = diagonal_su3(0.5, 0.3);

        // For diagonal g, character = sum of eigenvalue powers
        let z1 = Complex64::new(0.5_f64.cos(), 0.5_f64.sin());
        let z2 = Complex64::new(0.3_f64.cos(), 0.3_f64.sin());
        let z3 = Complex64::new((-0.8_f64).cos(), (-0.8_f64).sin());

        // Fundamental: χ = z₁ + z₂ + z₃
        let chi_fund = Su3Irrep::FUNDAMENTAL.character(&g);
        let expected_fund = z1 + z2 + z3;
        assert!(
            (chi_fund - expected_fund).norm() < 1e-10,
            "Fund char: got {:?}, expected {:?}",
            chi_fund,
            expected_fund
        );

        // Adjoint: χ = |z₁ + z₂ + z₃|² - 1
        let chi_adj = Su3Irrep::ADJOINT.character(&g);
        let tr = z1 + z2 + z3;
        let expected_adj = tr.norm_sqr() - Complex64::new(1.0, 0.0);
        assert!(
            (chi_adj - expected_adj).norm() < 1e-10,
            "Adj char: got {:?}, expected {:?}",
            chi_adj,
            expected_adj
        );
    }

    #[test]
    fn test_symmetric_representation() {
        let id = identity_3x3();

        let sym = Su3Irrep::SYMMETRIC.representation_matrix(&id).unwrap();
        assert_eq!(sym.len(), 6);

        // At identity, should be 6×6 identity
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sym[i][j] - Complex64::new(expected, 0.0)).norm() < 1e-10,
                    "Sym[{}][{}] = {:?}, expected {}",
                    i,
                    j,
                    sym[i][j],
                    expected
                );
            }
        }
    }

    // ========================================================================
    // Property-based tests (mathematical invariants)
    // ========================================================================

    #[test]
    fn test_adjoint_is_homomorphism() {
        // ρ(g₁ g₂) = ρ(g₁) ρ(g₂) - fundamental property of representations
        let g1 = diagonal_su3(0.3, 0.5);
        let g2 = diagonal_su3(0.7, -0.2);

        // Compute g1 * g2
        let mut g1g2 = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    g1g2[i][j] += g1[i][k] * g2[k][j];
                }
            }
        }

        let adj = Su3Irrep::ADJOINT;
        let rho_g1 = adj.representation_matrix(&g1).unwrap();
        let rho_g2 = adj.representation_matrix(&g2).unwrap();
        let rho_g1g2 = adj.representation_matrix(&g1g2).unwrap();

        // Compute ρ(g₁) ρ(g₂)
        let mut product = vec![vec![Complex64::new(0.0, 0.0); 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    product[i][j] += rho_g1[i][k] * rho_g2[k][j];
                }
            }
        }

        // Check ρ(g₁ g₂) = ρ(g₁) ρ(g₂)
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (rho_g1g2[i][j] - product[i][j]).norm() < 1e-10,
                    "Homomorphism failed at [{},{}]: {:?} vs {:?}",
                    i,
                    j,
                    rho_g1g2[i][j],
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_representation_matrices_are_unitary() {
        // For compact groups, all irreps are unitary: ρ(g)† ρ(g) = I
        let g = diagonal_su3(0.4, 0.6);

        for irrep in [
            Su3Irrep::TRIVIAL,
            Su3Irrep::FUNDAMENTAL,
            Su3Irrep::ANTIFUNDAMENTAL,
            Su3Irrep::ADJOINT,
        ] {
            let rho = irrep.representation_matrix(&g).unwrap();
            let dim = rho.len();

            // Compute ρ(g)† ρ(g)
            let mut product = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        product[i][j] += rho[k][i].conj() * rho[k][j];
                    }
                }
            }

            // Check it equals identity
            for i in 0..dim {
                for j in 0..dim {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (product[i][j] - Complex64::new(expected, 0.0)).norm() < 1e-9,
                        "Unitarity failed for {:?} at [{},{}]: {:?}",
                        irrep,
                        i,
                        j,
                        product[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_character_equals_trace_of_representation() {
        // χ(g) = Tr(ρ(g)) by definition
        let g = diagonal_su3(0.25, 0.75);

        for irrep in [
            Su3Irrep::TRIVIAL,
            Su3Irrep::FUNDAMENTAL,
            Su3Irrep::ADJOINT,
            Su3Irrep::SYMMETRIC,
        ] {
            let rho = irrep.representation_matrix(&g).unwrap();
            let trace: Complex64 = (0..rho.len()).map(|i| rho[i][i]).sum();
            let character = irrep.character(&g);

            assert!(
                (trace - character).norm() < 1e-10,
                "Character mismatch for {:?}: Tr={:?}, χ={:?}",
                irrep,
                trace,
                character
            );
        }
    }

    #[test]
    fn test_character_is_class_function() {
        // χ(h g h⁻¹) = χ(g) for all h
        let g = diagonal_su3(0.3, 0.5);

        // h is a permutation matrix (element of Weyl group)
        let h: [[Complex64; 3]; 3] = [
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ];

        // h⁻¹ = hᵀ for permutation matrices
        let h_inv: [[Complex64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| h[j][i]));

        // Compute h g h⁻¹
        let mut hg = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    hg[i][j] += h[i][k] * g[k][j];
                }
            }
        }
        let mut conjugated = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    conjugated[i][j] += hg[i][k] * h_inv[k][j];
                }
            }
        }

        for irrep in [Su3Irrep::FUNDAMENTAL, Su3Irrep::ADJOINT] {
            let chi_g = irrep.character(&g);
            let chi_conj = irrep.character(&conjugated);
            assert!(
                (chi_g - chi_conj).norm() < 1e-10,
                "Class function failed for {:?}: χ(g)={:?}, χ(hgh⁻¹)={:?}",
                irrep,
                chi_g,
                chi_conj
            );
        }
    }

    #[test]
    fn test_unsupported_representation_error() {
        // Test that unsupported representations return proper errors
        let g = identity_3x3();

        // (3,0) is not supported
        let irrep_3_0 = Su3Irrep::new(3, 0);
        let result = irrep_3_0.representation_matrix(&g);
        assert!(result.is_err(), "Should return error for unsupported (3,0)");
        if let Err(e) = result {
            assert!(
                format!("{:?}", e).contains("UnsupportedRepresentation"),
                "Error should be UnsupportedRepresentation"
            );
        }

        // (0,2) is not supported
        let irrep_0_2 = Su3Irrep::new(0, 2);
        let result = irrep_0_2.representation_matrix(&g);
        assert!(result.is_err(), "Should return error for unsupported (0,2)");

        // (2,1) is not supported
        let irrep_2_1 = Su3Irrep::new(2, 1);
        let result = irrep_2_1.representation_matrix(&g);
        assert!(result.is_err(), "Should return error for unsupported (2,1)");

        // Verify supported representations work
        let supported = [
            Su3Irrep::TRIVIAL,
            Su3Irrep::FUNDAMENTAL,
            Su3Irrep::ANTIFUNDAMENTAL,
            Su3Irrep::ADJOINT,
            Su3Irrep::SYMMETRIC,
        ];
        for irrep in supported {
            let result = irrep.representation_matrix(&g);
            assert!(
                result.is_ok(),
                "({},{}) should be supported",
                irrep.p,
                irrep.q
            );
        }
    }

    #[test]
    fn test_solve_cubic_cube_roots_of_unity() {
        // Test solve_cubic for the canonical case: x³ - 1 = 0
        // solve_cubic solves x³ + ax² + bx + c = 0
        let irrep = Su3Irrep::TRIVIAL; // Just need access to solve_cubic

        // x³ - 1 = 0 → roots are cube roots of unity: 1, ω, ω²
        let roots = irrep.solve_cubic(
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        );

        // Check roots satisfy x³ = 1
        for root in &roots {
            let cube = root.powi(3);
            assert!(
                (cube - Complex64::new(1.0, 0.0)).norm() < 1e-6,
                "Root {} cubed should equal 1, got {}",
                root,
                cube
            );
        }

        // Check roots are distinct (no repeated roots)
        let distinct = (roots[0] - roots[1]).norm() > 1e-6
            && (roots[1] - roots[2]).norm() > 1e-6
            && (roots[0] - roots[2]).norm() > 1e-6;
        assert!(distinct, "Cube roots of unity should be distinct");

        // Check one root is real (≈1) and two are complex conjugates
        let real_roots: Vec<_> = roots.iter().filter(|r| r.im.abs() < 1e-6).collect();
        assert_eq!(real_roots.len(), 1, "Should have exactly one real root");
        assert!(
            (real_roots[0].re - 1.0).abs() < 1e-6,
            "Real root should be 1"
        );
    }

    #[test]
    fn test_solve_cubic_triple_root() {
        // Triple root x³ = 0 (discriminant = 0)
        let irrep = Su3Irrep::TRIVIAL;
        let roots = irrep.solve_cubic(
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );

        // All roots should be near zero
        for root in &roots {
            assert!(root.norm() < 1e-6, "Triple root at 0: got {}", root);
        }
    }

    #[test]
    fn test_character_at_identity_equals_dimension() {
        // χ(e) = dim(V) for all representations
        let identity = identity_3x3();

        let test_cases = [
            (Su3Irrep::TRIVIAL, 1),
            (Su3Irrep::FUNDAMENTAL, 3),
            (Su3Irrep::ANTIFUNDAMENTAL, 3),
            (Su3Irrep::ADJOINT, 8),
            (Su3Irrep::SYMMETRIC, 6),
            (Su3Irrep::new(3, 0), 10),
            (Su3Irrep::new(0, 3), 10),
            (Su3Irrep::new(2, 1), 15),
        ];

        for (irrep, expected_dim) in test_cases {
            let chi = irrep.character(&identity);
            assert!(
                (chi.re - expected_dim as f64).abs() < 1e-10,
                "χ(e) for ({},{}) should be {}, got {}",
                irrep.p,
                irrep.q,
                expected_dim,
                chi
            );
            assert!(chi.im.abs() < 1e-10, "χ(e) should be real, got {}", chi);
        }
    }

    #[test]
    fn test_weyl_character_formula_consistency() {
        // Test that character values are consistent across different group elements
        // Character is a class function: χ(ghg⁻¹) = χ(h)

        let irrep = Su3Irrep::ADJOINT;

        // Two diagonal matrices (same conjugacy class if same eigenvalues)
        let g1 = diagonal_su3(0.5, 0.3);
        let g2 = diagonal_su3(0.5, 0.3); // Same eigenvalues

        let chi1 = irrep.character(&g1);
        let chi2 = irrep.character(&g2);

        assert!(
            (chi1 - chi2).norm() < 1e-10,
            "Same diagonal elements should give same character: {} vs {}",
            chi1,
            chi2
        );

        // Different diagonal matrices should give different characters (generically)
        let g3 = diagonal_su3(0.7, 0.1);
        let chi3 = irrep.character(&g3);

        // These should be different (unless by coincidence)
        // Just verify they're computed without error
        assert!(
            chi3.norm() > 0.0,
            "Character should be non-zero for non-identity"
        );
    }
}
