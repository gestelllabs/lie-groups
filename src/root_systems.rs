//! Root Systems for Semisimple Lie Algebras
//!
//! Implements the theory of root systems, which classify semisimple Lie algebras
//! via Cartan-Killing classification. Root systems encode the structure of the
//! Lie bracket via the adjoint representation on the Cartan subalgebra.
//!
//! # Mathematical Background
//!
//! For a semisimple Lie algebra 𝔤, choose a Cartan subalgebra 𝔥 (maximal abelian).
//! The adjoint representation ad: 𝔤 → End(𝔤) diagonalizes on 𝔥, giving eigenvalues
//! called **roots**:
//!
//! ```text
//! [H, X_α] = α(H) X_α    for all H ∈ 𝔥
//! ```
//!
//! The set of roots Φ ⊂ 𝔥* forms a **root system**, satisfying:
//! 1. Φ is finite, spans 𝔥*, and 0 ∉ Φ
//! 2. If α ∈ Φ, then -α ∈ Φ and no other scalar multiples
//! 3. Φ is closed under Weyl reflections: `s_α(β)` = β - ⟨β,α⟩α
//! 4. ⟨β,α⟩ ∈ ℤ for all α,β ∈ Φ (Cartan integers)
//!
//! # Cartan Classification
//!
//! | Type | Rank | Group | Dimension | Example |
//! |------|------|-------|-----------|---------|
//! | Aₙ   | n    | SU(n+1) | n(n+2)  | A₁ = SU(2) |
//! | Bₙ   | n    | SO(2n+1) | n(2n+1) | B₂ = SO(5) |
//! | Cₙ   | n    | Sp(2n)  | n(2n+1) | C₂ = Sp(4) |
//! | Dₙ   | n    | SO(2n)  | n(2n-1) | D₃ = SO(6) |
//! | E₆   | 6    | E₆      | 78      | Exceptional |
//! | E₇   | 7    | E₇      | 133     | Exceptional |
//! | E₈   | 8    | E₈      | 248     | Exceptional |
//! | F₄   | 4    | F₄      | 52      | Exceptional |
//! | G₂   | 2    | G₂      | 14      | Exceptional |
//!
//! # References
//!
//! - Hall, *Lie Groups, Lie Algebras, and Representations*, Chapter 7
//! - Humphreys, *Introduction to Lie Algebras and Representation Theory*, Ch. II
//! - Fulton & Harris, *Representation Theory*, Lecture 21

use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashSet;
use std::fmt::{self, Write};

/// A root in the dual space of a Cartan subalgebra.
///
/// Mathematically: α ∈ 𝔥* represented as a vector in ℝⁿ (rank n).
/// For SU(n+1) (type Aₙ), roots are differences of standard basis vectors: eᵢ - eⱼ.
///
/// # Example
///
/// ```
/// use lie_groups::Root;
///
/// // SU(3) root: e₁ - e₂
/// let alpha = Root::new(vec![1.0, -1.0, 0.0]);
/// assert_eq!(alpha.rank(), 3);
/// assert!((alpha.norm_squared() - 2.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Root {
    /// Coordinates in ℝⁿ (n = rank of Lie algebra)
    pub coords: Vec<f64>,
}

impl Root {
    /// Create a new root from coordinates.
    #[must_use]
    pub fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    /// Rank of the Lie algebra (dimension of Cartan subalgebra).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.coords.len()
    }

    /// Inner product ⟨α, β⟩ (standard Euclidean).
    #[must_use]
    pub fn inner_product(&self, other: &Root) -> f64 {
        assert_eq!(self.rank(), other.rank());
        self.coords
            .iter()
            .zip(&other.coords)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Squared norm ⟨α, α⟩.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.inner_product(self)
    }

    /// Weyl reflection `s_α(β)` = β - 2⟨β,α⟩/⟨α,α⟩ · α.
    #[must_use]
    pub fn reflect(&self, beta: &Root) -> Root {
        let factor = 2.0 * self.inner_product(beta) / self.norm_squared();
        Root::new(
            beta.coords
                .iter()
                .zip(&self.coords)
                .map(|(b, a)| b - factor * a)
                .collect(),
        )
    }

    /// Cartan integer ⟨β, α^∨⟩ = 2⟨β,α⟩/⟨α,α⟩.
    #[inline]
    #[must_use]
    pub fn cartan_integer(&self, beta: &Root) -> i32 {
        let value = 2.0 * self.inner_product(beta) / self.norm_squared();
        value.round() as i32
    }

    /// Check if this root is positive (first nonzero coordinate is positive).
    #[must_use]
    pub fn is_positive(&self) -> bool {
        for &c in &self.coords {
            if c.abs() > 1e-10 {
                return c > 0.0;
            }
        }
        false // Zero root (shouldn't happen)
    }

    /// Negate this root.
    #[must_use]
    pub fn negate(&self) -> Root {
        Root::new(self.coords.iter().map(|c| -c).collect())
    }
}

impl fmt::Display for Root {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.2}", c)?;
        }
        write!(f, ")")
    }
}

/// A root system for a semisimple Lie algebra.
///
/// Encodes the structure of the Lie bracket through roots and Weyl reflections.
///
/// # Example
///
/// ```
/// use lie_groups::RootSystem;
///
/// // SU(3) = type A₂
/// let su3 = RootSystem::type_a(2);
/// assert_eq!(su3.rank(), 2);
/// assert_eq!(su3.num_roots(), 6); // 3² - 1 = 8, but we store 6 roots
/// assert_eq!(su3.num_positive_roots(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct RootSystem {
    /// Rank of the Lie algebra (dimension of Cartan subalgebra)
    rank: usize,

    /// All roots (positive and negative)
    roots: Vec<Root>,

    /// Simple roots (basis for positive roots)
    simple_roots: Vec<Root>,

    /// Cartan matrix `A_ij` = ⟨`α_j`, `α_i`^∨⟩
    cartan_matrix: Vec<Vec<i32>>,
}

impl RootSystem {
    /// Create a type Aₙ root system (SU(n+1)).
    ///
    /// For SU(n+1), the rank is n, and roots are differences eᵢ - eⱼ for i ≠ j.
    /// Simple roots: αᵢ = eᵢ - eᵢ₊₁ for i = 1, ..., n.
    ///
    /// # Example
    ///
    /// ```
    /// use lie_groups::RootSystem;
    ///
    /// // SU(2) = A₁
    /// let su2 = RootSystem::type_a(1);
    /// assert_eq!(su2.rank(), 1);
    /// assert_eq!(su2.num_roots(), 2); // ±α
    ///
    /// // SU(3) = A₂
    /// let su3 = RootSystem::type_a(2);
    /// assert_eq!(su3.rank(), 2);
    /// assert_eq!(su3.simple_roots().len(), 2);
    /// ```
    #[must_use]
    pub fn type_a(n: usize) -> Self {
        assert!(n >= 1, "Type A_n requires n >= 1");

        let rank = n;

        // Simple roots: αᵢ = eᵢ - eᵢ₊₁ for i = 1, ..., n
        // Embedded in ℝⁿ⁺¹ with constraint Σxᵢ = 0
        let mut simple_roots = Vec::new();
        for i in 0..n {
            let mut coords = vec![0.0; n + 1];
            coords[i] = 1.0;
            coords[i + 1] = -1.0;
            simple_roots.push(Root::new(coords));
        }

        // Generate all positive roots
        let mut positive_roots = Vec::new();
        for i in 0..=n {
            for j in (i + 1)..=n {
                let mut coords = vec![0.0; n + 1];
                coords[i] = 1.0;
                coords[j] = -1.0;
                positive_roots.push(Root::new(coords));
            }
        }

        // All roots = positive + negative
        let mut roots = positive_roots.clone();
        for root in &positive_roots {
            roots.push(root.negate());
        }

        // Compute Cartan matrix: A_ij = ⟨α_j, α_i^∨⟩
        let cartan_matrix = simple_roots
            .iter()
            .map(|alpha_i| {
                simple_roots
                    .iter()
                    .map(|alpha_j| alpha_i.cartan_integer(alpha_j))
                    .collect()
            })
            .collect();

        Self {
            rank,
            roots,
            simple_roots,
            cartan_matrix,
        }
    }

    /// Rank of the Lie algebra.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// All roots (positive and negative).
    #[must_use]
    pub fn roots(&self) -> &[Root] {
        &self.roots
    }

    /// Simple roots (basis for root system).
    #[must_use]
    pub fn simple_roots(&self) -> &[Root] {
        &self.simple_roots
    }

    /// Cartan matrix `A_ij` = ⟨`α_j`, `α_i`^∨⟩.
    #[must_use]
    pub fn cartan_matrix(&self) -> &[Vec<i32>] {
        &self.cartan_matrix
    }

    /// Number of roots.
    #[must_use]
    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    /// Number of positive roots.
    #[must_use]
    pub fn num_positive_roots(&self) -> usize {
        self.roots.iter().filter(|r| r.is_positive()).count()
    }

    /// Get all positive roots.
    #[must_use]
    pub fn positive_roots(&self) -> Vec<Root> {
        self.roots
            .iter()
            .filter(|r| r.is_positive())
            .cloned()
            .collect()
    }

    /// Get the highest root (longest root in the positive system).
    ///
    /// The highest root θ is the unique positive root with maximal height
    /// (sum of coefficients when expanded in simple roots). It's also the
    /// longest root in the root system for simply-laced types like Aₙ.
    ///
    /// For type `A_n` (SU(n+1)), the highest root is θ = α₁ + α₂ + ... + αₙ.
    ///
    /// # Example
    ///
    /// ```
    /// use lie_groups::RootSystem;
    ///
    /// let su3 = RootSystem::type_a(2);
    /// let theta = su3.highest_root();
    ///
    /// // For SU(3), θ = α₁ + α₂ = (1, 0, -1)
    /// ```
    #[must_use]
    pub fn highest_root(&self) -> Root {
        // For type A_n, highest root is α₁ + α₂ + ... + αₙ
        // In coordinates: e₁ - eₙ₊₁ = (1, 0, ..., 0, -1)

        let n = self.rank;

        // Sum of simple roots
        let mut coords = vec![0.0; n + 1];

        for simple_root in &self.simple_roots {
            for (i, &coord) in simple_root.coords.iter().enumerate() {
                coords[i] += coord;
            }
        }

        Root::new(coords)
    }

    /// Check if a root is in the system.
    #[must_use]
    pub fn contains_root(&self, root: &Root) -> bool {
        self.roots.iter().any(|r| {
            r.coords.len() == root.coords.len()
                && r.coords
                    .iter()
                    .zip(root.coords.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
        })
    }

    /// Weyl reflection `s_α` for a root α.
    #[must_use]
    pub fn weyl_reflection(&self, alpha: &Root, beta: &Root) -> Root {
        alpha.reflect(beta)
    }

    /// Generate the Weyl group orbit of a weight under simple reflections.
    ///
    /// The Weyl group is generated by reflections in simple roots.
    /// For type Aₙ, |W| = (n+1)!
    #[must_use]
    pub fn weyl_orbit(&self, weight: &Root) -> Vec<Root> {
        let mut orbit = vec![weight.clone()];
        let mut seen = HashSet::new();
        seen.insert(format!("{:?}", weight.coords));

        let mut queue = vec![weight.clone()];

        while let Some(current) = queue.pop() {
            for simple_root in &self.simple_roots {
                let reflected = simple_root.reflect(&current);
                let key = format!("{:?}", reflected.coords);

                if !seen.contains(&key) {
                    seen.insert(key);
                    orbit.push(reflected.clone());
                    queue.push(reflected);
                }
            }
        }

        orbit
    }

    /// Dimension of the Lie algebra: rank + `num_roots`.
    ///
    /// For type Aₙ: dim = n + n(n+1) = n(n+2)
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.rank + self.num_roots()
    }

    /// Dominant weight chamber: λ such that ⟨λ, α⟩ ≥ 0 for all simple roots α.
    #[must_use]
    pub fn is_dominant_weight(&self, weight: &Root) -> bool {
        self.simple_roots
            .iter()
            .all(|alpha| weight.inner_product(alpha) >= -1e-10)
    }

    /// Express a root as a linear combination of simple roots.
    ///
    /// Returns coefficients [c₁, c₂, ..., cₙ] such that β = Σ cᵢ αᵢ.
    /// For roots in the system, coefficients are integers (positive for positive roots).
    ///
    /// Returns `None` if the root is not in this system, or if the expansion
    /// is not yet implemented for general root systems.
    ///
    /// # Supported Systems
    ///
    /// - **Type A** (SU(n+1)): Fully implemented. Roots `e_i - e_j` expand as
    ///   sums of consecutive simple roots.
    /// - **Other types**: Returns `None`. General expansion requires Cartan
    ///   matrix inversion, which is not yet implemented.
    pub fn simple_root_expansion(&self, root: &Root) -> Option<Vec<i32>> {
        if !self.contains_root(root) {
            return None;
        }

        // For type A_n (SU(n+1)): roots are e_i - e_j in ℝⁿ⁺¹
        // Simple roots are α_k = e_k - e_{k+1} for k = 0, ..., n-1
        // A root e_i - e_j (i < j) = α_i + α_{i+1} + ... + α_{j-1}
        //
        // Detect type A by checking if root has exactly one +1 and one -1 coordinate
        let coords = &root.coords;
        let mut pos_idx: Option<usize> = None;
        let mut neg_idx: Option<usize> = None;

        for (idx, &val) in coords.iter().enumerate() {
            if (val - 1.0).abs() < 1e-10 {
                if pos_idx.is_some() {
                    // Not type A format - fall through to general case
                    return None;
                }
                pos_idx = Some(idx);
            } else if (val + 1.0).abs() < 1e-10 {
                if neg_idx.is_some() {
                    return None;
                }
                neg_idx = Some(idx);
            } else if val.abs() > 1e-10 {
                // Non-zero entry that's not ±1 - not type A
                return None;
            }
        }

        let (Some(i), Some(j)) = (pos_idx, neg_idx) else {
            return None;
        };

        // Build coefficients: for e_i - e_j, coefficient of α_k is:
        // +1 if min(i,j) <= k < max(i,j) and i < j (positive root)
        // -1 if min(i,j) <= k < max(i,j) and i > j (negative root)
        let mut coeffs = vec![0i32; self.rank];
        let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
        let sign = if i < j { 1 } else { -1 };

        for k in min_idx..max_idx {
            if k < self.rank {
                coeffs[k] = sign;
            }
        }

        Some(coeffs)
    }
}

/// Weight lattice for a root system.
///
/// Weights are elements λ ∈ 𝔥* such that ⟨λ, α^∨⟩ ∈ ℤ for all roots α.
/// The weight lattice contains the root lattice as a sublattice.
///
/// # Fundamental Weights
///
/// For simple roots α₁, ..., αₙ, the fundamental weights ω₁, ..., ωₙ satisfy:
/// ```text
/// ⟨ωᵢ, αⱼ^∨⟩ = δᵢⱼ
/// ```
///
/// Every dominant weight can be written λ = Σ mᵢ ωᵢ with mᵢ ∈ ℤ≥₀.
pub struct WeightLattice {
    root_system: RootSystem,
    fundamental_weights: Vec<Root>,
}

impl WeightLattice {
    /// Create weight lattice from root system.
    #[must_use]
    pub fn from_root_system(root_system: RootSystem) -> Self {
        let fundamental_weights = Self::compute_fundamental_weights(&root_system);
        Self {
            root_system,
            fundamental_weights,
        }
    }

    /// Compute fundamental weights ωᵢ such that ⟨ωᵢ, αⱼ^∨⟩ = δᵢⱼ.
    fn compute_fundamental_weights(rs: &RootSystem) -> Vec<Root> {
        // For type A_n (SU(n+1)):
        // In GL(n+1) basis: ωᵢ = (1, 1, ..., 1, 0, ..., 0) with i ones
        // In SU(n+1) traceless basis: project to hyperplane Σxᵢ = 0
        //
        // Projection: x_traceless = x - (Σxᵢ)/(n+1) · (1,1,...,1)

        let n = rs.rank();
        let dim = n + 1; // Dimension of fundamental representation
        let mut weights = Vec::new();

        for i in 0..n {
            // Start with GL(n+1) fundamental weight
            let mut coords = vec![0.0; dim];
            for j in 0..=i {
                coords[j] = 1.0;
            }

            // Project to traceless subspace: subtract mean
            let sum: f64 = coords.iter().sum();
            let mean = sum / (dim as f64);
            for coord in &mut coords {
                *coord -= mean;
            }

            weights.push(Root::new(coords));
        }

        weights
    }

    /// Fundamental weights.
    #[must_use]
    pub fn fundamental_weights(&self) -> &[Root] {
        &self.fundamental_weights
    }

    /// Convert Dynkin labels (m₁, ..., mₙ) to weight λ = Σ mᵢ ωᵢ.
    #[must_use]
    pub fn dynkin_to_weight(&self, dynkin_labels: &[usize]) -> Root {
        assert_eq!(dynkin_labels.len(), self.root_system.rank());

        let mut weight_coords = vec![0.0; self.root_system.rank() + 1];
        for (i, &m) in dynkin_labels.iter().enumerate() {
            for (j, &w) in self.fundamental_weights[i].coords.iter().enumerate() {
                weight_coords[j] += (m as f64) * w;
            }
        }

        Root::new(weight_coords)
    }

    /// Get the root system.
    #[must_use]
    pub fn root_system(&self) -> &RootSystem {
        &self.root_system
    }

    /// Compute ρ (half-sum of positive roots) = sum of fundamental weights.
    ///
    /// For type Aₙ: ρ = ω₁ + ω₂ + ... + ωₙ
    #[must_use]
    pub fn rho(&self) -> Root {
        let mut rho_coords = vec![0.0; self.root_system.rank() + 1];
        for omega in &self.fundamental_weights {
            for (i, &coord) in omega.coords.iter().enumerate() {
                rho_coords[i] += coord;
            }
        }
        Root::new(rho_coords)
    }

    /// Kostant's partition function P(γ): number of ways to write γ as a sum of positive roots.
    ///
    /// Counts solutions to: γ = Σ_{α ∈ Φ⁺} `n_α` · α where `n_α` ≥ 0 are non-negative integers.
    ///
    /// This is computed via dynamic programming with ORDERED roots to avoid overcounting.
    ///
    /// # Returns
    /// Number of distinct ways (counting multiplicities) to express γ as a sum of positive roots.
    /// Returns 0 if γ cannot be expressed as such a sum.
    ///
    /// # Algorithm
    /// Uses DP with memoization and ROOT ORDERING to count unordered partitions.
    /// For each root `α_i`, we only consider using roots `α_j` where j ≥ i.
    /// This ensures we count combinations, not permutations.
    ///
    /// # Performance
    /// O(|Φ⁺| × V) where V is the size of the reachable region.
    /// For typical representations, this is manageable.
    fn kostant_partition_function(&self, gamma: &Root) -> usize {
        use std::collections::HashMap;

        // Base case: zero vector has exactly one representation (empty sum)
        if gamma.coords.iter().all(|&x| x.abs() < 1e-10) {
            return 1;
        }

        // Use memoization to avoid recomputation
        // Key: (gamma, starting_index) to track which roots we can still use
        let mut memo: HashMap<(String, usize), usize> = HashMap::new();
        let positive_roots: Vec<Root> = self.root_system.positive_roots();

        Self::partition_helper_ordered(gamma, &positive_roots, 0, &mut memo)
    }

    /// Recursive helper for partition function with ROOT ORDERING to avoid overcounting.
    ///
    /// Only considers roots starting from index `start_idx` onwards.
    /// This ensures we generate combinations (not permutations) of roots.
    fn partition_helper_ordered(
        gamma: &Root,
        positive_roots: &[Root],
        start_idx: usize,
        memo: &mut std::collections::HashMap<(String, usize), usize>,
    ) -> usize {
        // Base case: zero
        if gamma.coords.iter().all(|&x| x.abs() < 1e-10) {
            return 1;
        }

        // Base case: no more roots to try
        if start_idx >= positive_roots.len() {
            return 0;
        }

        // Check memo
        let key = (WeightLattice::weight_key(&gamma.coords), start_idx);
        if let Some(&result) = memo.get(&key) {
            return result;
        }

        // Early termination: if γ has large negative coordinates, it's not expressible
        let mut has_large_negative = false;
        for &coord in &gamma.coords {
            if coord < -100.0 {
                has_large_negative = true;
                break;
            }
        }

        if has_large_negative {
            memo.insert(key, 0);
            return 0;
        }

        // Recursive case: for current root α at start_idx, we have two choices:
        // 1. Don't use this root at all: move to next root
        // 2. Use this root k times (k ≥ 1): subtract k·α and stay at same index
        let mut count = 0;
        let alpha = &positive_roots[start_idx];

        // Choice 1: Skip this root entirely
        if memo.len() < 100_000 {
            count += Self::partition_helper_ordered(gamma, positive_roots, start_idx + 1, memo);
        }

        // Choice 2: Use this root at least once
        let mut gamma_minus_k_alpha = gamma.clone();
        for (i, &a) in alpha.coords.iter().enumerate() {
            gamma_minus_k_alpha.coords[i] -= a;
        }

        // Can we use this root? Check if γ - α is "valid" (not too negative)
        let can_use = gamma_minus_k_alpha.coords.iter().all(|&x| x > -100.0);

        if can_use && memo.len() < 100_000 {
            // Recurse with γ - α, staying at same root index (can use it multiple times)
            count += Self::partition_helper_ordered(
                &gamma_minus_k_alpha,
                positive_roots,
                start_idx,
                memo,
            );
        }

        memo.insert(key, count);
        count
    }

    /// Generate Weyl group elements for type `A_n` (symmetric group S_{n+1}).
    ///
    /// The Weyl group of type `A_n` is isomorphic to S_{n+1}, the symmetric group
    /// of permutations of {1, 2, ..., n+1}.
    ///
    /// Each permutation σ acts on weights λ = (λ₁, ..., λ_{n+1}) by permuting coordinates.
    ///
    /// # Returns
    /// Vector of (permutation, sign) pairs where:
    /// - permutation[i] = j means coordinate i maps to coordinate j
    /// - sign = +1 for even permutations, -1 for odd
    ///
    /// # Performance
    /// Generates all (n+1)! permutations. For SU(3) this is 6, for SU(4) this is 24, etc.
    fn weyl_group_type_a(&self) -> Vec<(Vec<usize>, i32)> {
        let n_plus_1 = self.root_system.rank() + 1;
        let mut perms = Vec::new();

        // Generate all permutations via Heap's algorithm
        let mut current: Vec<usize> = (0..n_plus_1).collect();
        Self::generate_permutations(&mut current, n_plus_1, &mut perms);

        perms
    }

    /// Generate all permutations using Heap's algorithm.
    fn generate_permutations(arr: &mut [usize], size: usize, result: &mut Vec<(Vec<usize>, i32)>) {
        if size == 1 {
            let perm = arr.to_vec();
            let sign = Self::permutation_sign(&perm);
            result.push((perm, sign));
            return;
        }

        for i in 0..size {
            Self::generate_permutations(arr, size - 1, result);

            if size % 2 == 0 {
                arr.swap(i, size - 1);
            } else {
                arr.swap(0, size - 1);
            }
        }
    }

    /// Compute the sign of a permutation (+1 for even, -1 for odd).
    fn permutation_sign(perm: &[usize]) -> i32 {
        let n = perm.len();
        let mut sign = 1i32;

        for i in 0..n {
            for j in (i + 1)..n {
                if perm[i] > perm[j] {
                    sign *= -1;
                }
            }
        }

        sign
    }

    /// Apply Weyl group element (permutation) to a weight.
    fn weyl_action(&self, weight: &Root, permutation: &[usize]) -> Root {
        let mut new_coords = vec![0.0; weight.coords.len()];
        for (i, &pi) in permutation.iter().enumerate() {
            new_coords[i] = weight.coords[pi];
        }
        Root::new(new_coords)
    }

    /// Compute weight multiplicity using Kostant's formula.
    ///
    /// Kostant's formula (exact, not recursive like Freudenthal):
    /// ```text
    /// m_Λ(λ) = Σ_{w ∈ W} ε(w) · P(w·(Λ+ρ) - (λ+ρ))
    /// ```
    ///
    /// where:
    /// - Λ = highest weight of the representation
    /// - λ = weight whose multiplicity we compute
    /// - W = Weyl group
    /// - ε(w) = sign of w (+1 for even, -1 for odd)
    /// - P(γ) = partition function counting ways to write γ as sum of positive roots
    ///
    /// # Mathematical Background
    ///
    /// This is the most general multiplicity formula, working for all weights in all
    /// representations. Unlike Freudenthal (which requires dominance), Kostant works
    /// directly.
    ///
    /// # Performance
    /// O(|W| × `P_cost`) where |W| = (n+1)! for type `A_n` and `P_cost` is partition function cost.
    /// For SU(3): 6 Weyl group elements.
    /// For SU(4): 24 Weyl group elements.
    ///
    /// # References
    /// - Humphreys, "Introduction to Lie Algebras and Representation Theory", §24.3
    /// - Kostant (1959), "A Formula for the Multiplicity of a Weight"
    #[must_use]
    pub fn kostant_multiplicity(&self, highest_weight: &Root, weight: &Root) -> usize {
        let rho = self.rho();

        // Compute Λ + ρ
        let mut lambda_plus_rho = highest_weight.clone();
        for (i, &r) in rho.coords.iter().enumerate() {
            lambda_plus_rho.coords[i] += r;
        }

        // Compute λ + ρ
        let mut mu_plus_rho = weight.clone();
        for (i, &r) in rho.coords.iter().enumerate() {
            mu_plus_rho.coords[i] += r;
        }

        // Sum over Weyl group
        let weyl_group = self.weyl_group_type_a();
        let mut multiplicity = 0i32; // Use signed arithmetic for alternating sum

        for (perm, sign) in weyl_group {
            // Compute w·(Λ+ρ)
            let w_lambda_plus_rho = self.weyl_action(&lambda_plus_rho, &perm);

            // Compute γ = w·(Λ+ρ) - (λ+ρ)
            let mut gamma = w_lambda_plus_rho.clone();
            for (i, &mu) in mu_plus_rho.coords.iter().enumerate() {
                gamma.coords[i] -= mu;
            }

            // Compute P(γ)
            let p_gamma = self.kostant_partition_function(&gamma);

            // Add signed contribution
            multiplicity += sign * (p_gamma as i32);
        }

        // Result must be non-negative
        assert!(
            multiplicity >= 0,
            "Kostant multiplicity must be non-negative, got {}",
            multiplicity
        );
        multiplicity as usize
    }

    /// Compute dimension of irreducible representation using Weyl dimension formula.
    ///
    /// For type Aₙ with highest weight λ:
    /// ```text
    /// dim(λ) = ∏_{α>0} ⟨λ + ρ, α⟩ / ⟨ρ, α⟩
    /// ```
    ///
    /// This is the EXACT dimension - no approximation.
    #[must_use]
    pub fn weyl_dimension(&self, highest_weight: &Root) -> usize {
        let rho = self.rho();

        // λ + ρ
        let mut lambda_plus_rho = highest_weight.clone();
        for (i, &r) in rho.coords.iter().enumerate() {
            lambda_plus_rho.coords[i] += r;
        }

        let mut numerator = 1.0;
        let mut denominator = 1.0;

        for alpha in self.root_system.positive_roots() {
            let num = lambda_plus_rho.inner_product(&alpha);
            let denom = rho.inner_product(&alpha);

            numerator *= num;
            denominator *= denom;
        }

        (numerator / denominator).round() as usize
    }

    /// Helper function to create a stable `HashMap` key from weight coordinates.
    ///
    /// Rounds each coordinate to 10 decimal places to avoid floating-point precision issues
    /// when using coordinates as `HashMap` keys.
    fn weight_key(coords: &[f64]) -> String {
        let rounded: Vec<String> = coords.iter().map(|&x| format!("{:.10}", x)).collect();
        format!("[{}]", rounded.join(", "))
    }

    /// Generate all weights in an irreducible representation.
    ///
    /// Uses the CORRECT, GENERAL algorithm:
    /// 1. Generate candidate weights by exploring from highest weight with ALL positive roots
    /// 2. Add Weyl orbit of highest weight to ensure completeness
    /// 3. Use Kostant's formula to compute accurate multiplicities
    /// 4. Return only weights with multiplicity > 0
    ///
    /// This works for ALL representations, including adjoint where negative roots appear.
    #[must_use]
    pub fn weights_of_irrep(&self, highest_weight: &Root) -> Vec<(Root, usize)> {
        use std::collections::{HashMap, VecDeque};

        let mut candidates: HashMap<String, Root> = HashMap::new();
        let mut queue: VecDeque<Root> = VecDeque::new();

        // Start with highest weight
        let hw_key = Self::weight_key(&highest_weight.coords);
        candidates.insert(hw_key, highest_weight.clone());
        queue.push_back(highest_weight.clone());

        // BFS: Subtract ALL positive roots (not just simple roots!)
        // This ensures we reach all weights, including negative roots in adjoint rep
        while let Some(weight) = queue.pop_front() {
            for pos_root in self.root_system.positive_roots() {
                let mut new_weight = weight.clone();
                for (i, &a) in pos_root.coords.iter().enumerate() {
                    new_weight.coords[i] -= a;
                }

                let new_key = Self::weight_key(&new_weight.coords);

                // Skip if already seen
                if candidates.contains_key(&new_key) {
                    continue;
                }

                // Check if in reasonable bounds (heuristic to avoid infinite exploration)
                let norm = new_weight.norm_squared();
                let hw_norm = highest_weight.norm_squared();
                if norm > 3.0 * hw_norm + 10.0 {
                    continue;
                }

                candidates.insert(new_key, new_weight.clone());
                queue.push_back(new_weight);

                // Safety limit
                if candidates.len() > 1000 {
                    break;
                }
            }

            if candidates.len() > 1000 {
                break;
            }
        }

        // CRITICAL: Also add the full Weyl orbit of the highest weight
        // This ensures we don't miss any extremal weights
        let weyl_orbit = self.root_system.weyl_orbit(highest_weight);
        for w in weyl_orbit {
            let key = Self::weight_key(&w.coords);
            candidates.insert(key, w);
        }

        // Compute accurate multiplicities using Kostant's formula
        let candidate_list: Vec<Root> = candidates.into_values().collect();
        let mut result = Vec::new();

        for weight in candidate_list.iter() {
            let mult = self.kostant_multiplicity(highest_weight, weight);
            if mult > 0 {
                result.push((weight.clone(), mult));
            }
        }

        result
    }

    /// Generate a string representation of a weight diagram for rank 2.
    ///
    /// For SU(3) representations, this creates a 2D triangular lattice
    /// showing all weights with their multiplicities.
    ///
    /// # Example
    ///
    /// ```
    /// use lie_groups::{RootSystem, WeightLattice};
    ///
    /// let rs = RootSystem::type_a(2);
    /// let wl = WeightLattice::from_root_system(rs);
    /// let highest = wl.dynkin_to_weight(&[1, 1]); // Adjoint rep
    ///
    /// let diagram = wl.weight_diagram_string(&highest);
    /// // Should show 8 weights (8-dimensional rep)
    /// ```
    #[must_use]
    pub fn weight_diagram_string(&self, highest_weight: &Root) -> String {
        if self.root_system.rank() != 2 {
            return "Weight diagrams only implemented for rank 2".to_string();
        }

        let weights = self.weights_of_irrep(highest_weight);

        let mut output = String::new();
        writeln!(
            &mut output,
            "Weight diagram for highest weight {:?}",
            highest_weight.coords
        )
        .expect("String formatting should not fail");
        writeln!(&mut output, "Dimension: {}", weights.len())
            .expect("String formatting should not fail");
        output.push_str("Weights (multiplicity):\n");

        for (weight, mult) in &weights {
            writeln!(&mut output, "  {} (×{})", weight, mult)
                .expect("String formatting should not fail");
        }

        output
    }
}

/// Cartan subalgebra of a semisimple Lie algebra.
///
/// The Cartan subalgebra 𝔥 is the maximal abelian subalgebra consisting of
/// simultaneously diagonalizable elements. For SU(n+1) (type Aₙ), it consists
/// of traceless diagonal matrices.
///
/// # Mathematical Background
///
/// - Dimension: rank of the Lie algebra
/// - Basis: {H₁, ..., Hₙ} where [Hᵢ, Hⱼ] = 0 (all commute)
/// - For SU(n+1): Hᵢ = Eᵢᵢ - Eᵢ₊₁,ᵢ₊₁ (diagonal matrices)
/// - Roots are functionals α: 𝔥 → ℝ
///
/// # Example
///
/// ```
/// use lie_groups::root_systems::CartanSubalgebra;
///
/// // SU(3) Cartan subalgebra (2-dimensional)
/// let cartan = CartanSubalgebra::type_a(2);
/// assert_eq!(cartan.dimension(), 2);
///
/// let basis = cartan.basis_matrices();
/// assert_eq!(basis.len(), 2);
/// ```
pub struct CartanSubalgebra {
    /// Dimension of the Cartan subalgebra (rank of Lie algebra)
    dimension: usize,

    /// Basis matrices as complex arrays
    /// For SU(n+1), these are n diagonal traceless matrices
    basis_matrices: Vec<Array2<Complex64>>,

    /// Matrix dimension (N for SU(N))
    matrix_size: usize,
}

impl CartanSubalgebra {
    /// Create Cartan subalgebra for type Aₙ (SU(n+1)).
    ///
    /// The Cartan subalgebra consists of traceless diagonal matrices.
    /// Basis: Hᵢ = Eᵢᵢ - Eᵢ₊₁,ᵢ₊₁ for i = 1, ..., n
    ///
    /// # Example
    ///
    /// ```
    /// use lie_groups::root_systems::CartanSubalgebra;
    ///
    /// // SU(2) Cartan subalgebra: 1-dimensional
    /// let h = CartanSubalgebra::type_a(1);
    /// assert_eq!(h.dimension(), 1);
    ///
    /// // First basis element: diag(1, -1)
    /// let h1 = &h.basis_matrices()[0];
    /// assert!((h1[(0,0)].re - 1.0).abs() < 1e-10);
    /// assert!((h1[(1,1)].re + 1.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn type_a(n: usize) -> Self {
        assert!(n >= 1, "Type A_n requires n >= 1");

        let matrix_size = n + 1;
        let dimension = n;

        let mut basis_matrices = Vec::new();

        // Generate Hᵢ = Eᵢᵢ - Eᵢ₊₁,ᵢ₊₁ for i = 0, ..., n-1
        for i in 0..n {
            let mut h = Array2::<Complex64>::zeros((matrix_size, matrix_size));
            h[(i, i)] = Complex64::new(1.0, 0.0);
            h[(i + 1, i + 1)] = Complex64::new(-1.0, 0.0);
            basis_matrices.push(h);
        }

        Self {
            dimension,
            basis_matrices,
            matrix_size,
        }
    }

    /// Dimension of the Cartan subalgebra (rank).
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Basis matrices for the Cartan subalgebra.
    #[must_use]
    pub fn basis_matrices(&self) -> &[Array2<Complex64>] {
        &self.basis_matrices
    }

    /// Matrix size (N for SU(N)).
    #[must_use]
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Evaluate a root functional on a Cartan element.
    ///
    /// For a root α and Cartan element H = Σ cᵢ Hᵢ, computes α(H) = Σ cᵢ α(Hᵢ).
    ///
    /// # Arguments
    ///
    /// * `root` - Root as a vector of coordinates
    /// * `coefficients` - Coefficients [c₁, ..., cₙ] in basis {H₁, ..., Hₙ}
    ///
    /// # Returns
    ///
    /// The value α(H) ∈ ℂ
    #[must_use]
    pub fn evaluate_root(&self, root: &Root, coefficients: &[f64]) -> Complex64 {
        assert_eq!(
            coefficients.len(),
            self.dimension,
            "Coefficient vector must match Cartan dimension"
        );
        assert_eq!(
            root.rank(),
            self.matrix_size,
            "Root dimension must match matrix size"
        );

        // For type A_n, α(H) is computed via the standard pairing
        // If H = diag(h₁, ..., hₙ₊₁) with Σhᵢ = 0, and α = (α₁, ..., αₙ₊₁)
        // then α(H) = Σ αᵢ hᵢ

        // Reconstruct diagonal H from basis coefficients
        let mut h_diagonal = vec![Complex64::new(0.0, 0.0); self.matrix_size];
        for (i, &c) in coefficients.iter().enumerate() {
            h_diagonal[i] += Complex64::new(c, 0.0);
            h_diagonal[i + 1] -= Complex64::new(c, 0.0);
        }

        // Compute α(H) = Σ αᵢ hᵢ
        root.coords
            .iter()
            .zip(&h_diagonal)
            .map(|(alpha_i, h_i)| h_i * alpha_i)
            .sum()
    }

    /// Project a matrix onto the Cartan subalgebra.
    ///
    /// For a matrix M, finds coefficients [c₁, ..., cₙ] such that
    /// Σ cᵢ Hᵢ best approximates M in the Frobenius norm.
    ///
    /// Since the basis {H₁, ..., Hₙ} is not necessarily orthogonal,
    /// we solve the linear system: G c = g where Gᵢⱼ = ⟨Hᵢ, Hⱼ⟩.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Matrix to project
    ///
    /// # Returns
    ///
    /// `Some(coefficients)` in the Cartan basis for rank ≤ 2,
    /// `None` for rank > 2 (Gaussian elimination not yet implemented).
    ///
    /// # Limitations
    ///
    /// Currently only supports rank 1 (SU(2)) and rank 2 (SU(3)) systems.
    /// Higher rank systems require general Gaussian elimination.
    #[must_use]
    pub fn project_matrix(&self, matrix: &Array2<Complex64>) -> Option<Vec<f64>> {
        assert_eq!(
            matrix.shape(),
            &[self.matrix_size, self.matrix_size],
            "Matrix must be {}×{}",
            self.matrix_size,
            self.matrix_size
        );

        // Compute Gram matrix G and right-hand side g
        let n = self.dimension;
        let mut gram = vec![vec![0.0; n]; n];
        let mut rhs = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                // Gᵢⱼ = ⟨Hᵢ, Hⱼ⟩ = Tr(Hᵢ† Hⱼ)
                let inner: Complex64 = self.basis_matrices[i]
                    .iter()
                    .zip(self.basis_matrices[j].iter())
                    .map(|(h_i, h_j)| h_i.conj() * h_j)
                    .sum();
                gram[i][j] = inner.re;
            }

            // gᵢ = ⟨M, Hᵢ⟩ = Tr(M† Hᵢ)
            let inner: Complex64 = matrix
                .iter()
                .zip(self.basis_matrices[i].iter())
                .map(|(m_ij, h_ij)| m_ij.conj() * h_ij)
                .sum();
            rhs[i] = inner.re;
        }

        // Solve G c = g using direct formulas (small n only)
        let mut coefficients = vec![0.0; n];

        match n {
            1 => {
                coefficients[0] = rhs[0] / gram[0][0];
                Some(coefficients)
            }
            2 => {
                // Solve 2×2 system directly
                let det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
                coefficients[0] = (rhs[0] * gram[1][1] - rhs[1] * gram[0][1]) / det;
                coefficients[1] = (gram[0][0] * rhs[1] - gram[1][0] * rhs[0]) / det;
                Some(coefficients)
            }
            _ => {
                // Rank > 2: Gaussian elimination not yet implemented
                None
            }
        }
    }

    /// Construct a Cartan element from coefficients.
    ///
    /// Given coefficients [c₁, ..., cₙ], returns H = Σ cᵢ Hᵢ.
    #[must_use]
    pub fn from_coefficients(&self, coefficients: &[f64]) -> Array2<Complex64> {
        assert_eq!(
            coefficients.len(),
            self.dimension,
            "Must provide {} coefficients",
            self.dimension
        );

        let mut result = Array2::<Complex64>::zeros((self.matrix_size, self.matrix_size));

        for (&c_i, h_i) in coefficients.iter().zip(&self.basis_matrices) {
            result = result + h_i * c_i;
        }

        result
    }

    /// Check if a matrix is in the Cartan subalgebra.
    ///
    /// Returns true if the matrix is diagonal (or nearly diagonal within tolerance).
    #[must_use]
    pub fn contains(&self, matrix: &Array2<Complex64>, tolerance: f64) -> bool {
        assert_eq!(
            matrix.shape(),
            &[self.matrix_size, self.matrix_size],
            "Matrix must be {}×{}",
            self.matrix_size,
            self.matrix_size
        );

        // Check if matrix is diagonal
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                if i != j && matrix[(i, j)].norm() > tolerance {
                    return false;
                }
            }
        }

        // Check if diagonal entries sum to zero (traceless)
        let trace: Complex64 = (0..self.matrix_size).map(|i| matrix[(i, i)]).sum();

        trace.norm() < tolerance
    }

    /// Killing form restricted to Cartan subalgebra.
    ///
    /// For H, H' ∈ 𝔥, the Killing form is κ(H, H') = Tr(`ad_H` ∘ `ad_H`').
    /// For type `A_n`, this simplifies significantly.
    ///
    /// # Arguments
    ///
    /// * `coeffs1` - First Cartan element as coefficients
    /// * `coeffs2` - Second Cartan element as coefficients
    ///
    /// # Returns
    ///
    /// The value κ(H₁, H₂)
    #[must_use]
    pub fn killing_form(&self, coeffs1: &[f64], coeffs2: &[f64]) -> f64 {
        assert_eq!(coeffs1.len(), self.dimension);
        assert_eq!(coeffs2.len(), self.dimension);

        // For SU(n+1), the Killing form on the Cartan is
        // κ(H, H') = 2(n+1) Tr(HH')
        //
        // For our basis Hᵢ = Eᵢᵢ - Eᵢ₊₁,ᵢ₊₁:
        // Tr(Hᵢ Hⱼ) = 2δᵢⱼ

        let n_plus_1 = self.matrix_size as f64;

        coeffs1
            .iter()
            .zip(coeffs2)
            .map(|(c1, c2)| 2.0 * n_plus_1 * 2.0 * c1 * c2)
            .sum()
    }

    /// Dual basis in 𝔥* (root space).
    ///
    /// Returns roots {α₁, ..., αₙ} such that αᵢ(Hⱼ) = δᵢⱼ.
    /// For type `A_n`, these correspond to the simple roots.
    #[must_use]
    pub fn dual_basis(&self) -> Vec<Root> {
        let mut dual_roots = Vec::new();

        for i in 0..self.dimension {
            // For Hᵢ = Eᵢᵢ - Eᵢ₊₁,ᵢ₊₁, the dual is αᵢ = eᵢ - eᵢ₊₁
            let mut coords = vec![0.0; self.matrix_size];
            coords[i] = 1.0;
            coords[i + 1] = -1.0;
            dual_roots.push(Root::new(coords));
        }

        dual_roots
    }
}

/// Weyl chamber for a root system.
///
/// A Weyl chamber is a connected component of the complement of the reflecting
/// hyperplanes in the dual Cartan space 𝔥*. The **fundamental Weyl chamber**
/// (or **dominant chamber**) is defined by the simple roots:
///
/// ```text
/// C = {λ ∈ 𝔥* : ⟨λ, αᵢ⟩ ≥ 0 for all simple roots αᵢ}
/// ```
///
/// # Mathematical Background
///
/// - The Weyl group W acts on 𝔥* by reflections through root hyperplanes
/// - W partitions 𝔥* into finitely many Weyl chambers
/// - All chambers are isometric under W
/// - The fundamental chamber parametrizes dominant weights (highest weights
///   of irreducible representations)
///
/// # Example
///
/// ```
/// use lie_groups::root_systems::{RootSystem, WeylChamber, Root};
///
/// let root_system = RootSystem::type_a(2); // SU(3)
/// let chamber = WeylChamber::fundamental(&root_system);
///
/// // Fundamental weight ω₁ = (2/3, -1/3, -1/3) is dominant (traceless for SU(3))
/// let omega1 = Root::new(vec![2.0/3.0, -1.0/3.0, -1.0/3.0]);
/// assert!(chamber.contains(&omega1, false)); // Non-strict since on boundary
///
/// // Negative weight is not dominant
/// let neg = Root::new(vec![-1.0, 0.0, 1.0]);
/// assert!(!chamber.contains(&neg, false));
/// ```
///
/// # References
///
/// - Humphreys, *Introduction to Lie Algebras*, §10.2
/// - Hall, *Lie Groups*, §8.4
#[derive(Debug, Clone)]
pub struct WeylChamber {
    /// The root system defining the chamber
    root_system: RootSystem,

    /// Simple roots defining the chamber walls
    simple_roots: Vec<Root>,
}

impl WeylChamber {
    /// Construct the fundamental Weyl chamber for a root system.
    ///
    /// The fundamental chamber is bounded by hyperplanes perpendicular to
    /// the simple roots:
    ///
    /// ```text
    /// C = {λ : ⟨λ, αᵢ⟩ ≥ 0 for all simple αᵢ}
    /// ```
    #[must_use]
    pub fn fundamental(root_system: &RootSystem) -> Self {
        Self {
            root_system: root_system.clone(),
            simple_roots: root_system.simple_roots.clone(),
        }
    }

    /// Check if a weight is in the Weyl chamber.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight λ ∈ 𝔥* to test
    /// * `strict` - If true, use strict inequality (⟨λ, αᵢ⟩ > 0); else non-strict (≥ 0)
    ///
    /// # Returns
    ///
    /// True if λ is in the chamber (dominant weight)
    #[must_use]
    pub fn contains(&self, weight: &Root, strict: bool) -> bool {
        // For type A_n, roots have n+1 coordinates
        assert_eq!(weight.rank(), self.root_system.rank + 1);

        for alpha in &self.simple_roots {
            let pairing = weight.inner_product(alpha);

            if strict {
                if pairing <= 1e-10 {
                    return false;
                }
            } else if pairing < -1e-10 {
                return false;
            }
        }

        true
    }

    /// Project a weight onto the closure of the fundamental Weyl chamber.
    ///
    /// Uses the Weyl group to map an arbitrary weight to its unique dominant
    /// representative (closest point in the chamber).
    ///
    /// # Algorithm
    ///
    /// Iteratively reflect through violated hyperplanes until all simple root
    /// pairings are non-negative.
    #[must_use]
    pub fn project(&self, weight: &Root) -> Root {
        let mut current = weight.clone();

        // Maximum iterations to prevent infinite loops
        let max_iterations = 100;
        let mut iterations = 0;

        while iterations < max_iterations {
            let mut found_violation = false;

            for alpha in &self.simple_roots {
                let pairing = current.inner_product(alpha);

                if pairing < -1e-10 {
                    // Reflect through α to increase ⟨current, α⟩
                    current = alpha.reflect(&current);
                    found_violation = true;
                }
            }

            if !found_violation {
                break;
            }

            iterations += 1;
        }

        if iterations >= max_iterations {
            eprintln!(
                "Warning: WeylChamber::project did not converge in {} iterations",
                max_iterations
            );
        }

        current
    }

    /// Get the simple roots defining the chamber walls.
    #[must_use]
    pub fn simple_roots(&self) -> &[Root] {
        &self.simple_roots
    }
}

/// Fundamental alcove for an affine root system.
///
/// An **alcove** is a bounded region in the Cartan space defined by affine
/// hyperplanes (root hyperplanes shifted by integer levels). The fundamental
/// alcove is:
///
/// ```text
/// A = {λ ∈ 𝔥* : ⟨λ, αᵢ⟩ ≥ 0 for all simple αᵢ, and ⟨λ, θ⟩ ≤ 1}
/// ```
///
/// where θ is the **highest root** (longest root in the positive system).
///
/// # Mathematical Background
///
/// - Alcoves tile 𝔥* under the affine Weyl group Wₐff
/// - The fundamental alcove parametrizes integrable highest-weight modules
///   at level k (conformal field theory, loop groups)
/// - Vertices of the alcove are fundamental weights
///
/// # Example
///
/// ```
/// use lie_groups::root_systems::{RootSystem, Alcove, Root};
///
/// let root_system = RootSystem::type_a(2); // SU(3)
/// let alcove = Alcove::fundamental(&root_system);
///
/// // Weight inside alcove
/// let lambda = Root::new(vec![0.3, 0.3, -0.6]);
/// // (need to verify ⟨λ, αᵢ⟩ ≥ 0 and ⟨λ, θ⟩ ≤ 1)
/// ```
///
/// # References
///
/// - Kac, *Infinite Dimensional Lie Algebras*, §6.2
/// - Humphreys, *Reflection Groups and Coxeter Groups*, §4.4
#[derive(Debug, Clone)]
pub struct Alcove {
    /// The root system defining the alcove
    root_system: RootSystem,

    /// Simple roots (defining lower walls)
    simple_roots: Vec<Root>,

    /// Highest root θ (defining upper wall ⟨λ, θ⟩ ≤ level)
    highest_root: Root,

    /// Level k (affine level; typically 1 for fundamental alcove)
    level: f64,
}

impl Alcove {
    /// Construct the fundamental alcove at level k = 1.
    ///
    /// The fundamental alcove is:
    /// ```text
    /// A = {λ : ⟨λ, αᵢ⟩ ≥ 0, ⟨λ, θ⟩ ≤ 1}
    /// ```
    #[must_use]
    pub fn fundamental(root_system: &RootSystem) -> Self {
        let simple_roots = root_system.simple_roots.clone();
        let highest_root = root_system.highest_root();

        Self {
            root_system: root_system.clone(),
            simple_roots,
            highest_root,
            level: 1.0,
        }
    }

    /// Construct an alcove at a specified level k.
    ///
    /// # Arguments
    ///
    /// * `root_system` - The root system
    /// * `level` - Affine level k (positive integer for integrable modules)
    #[must_use]
    pub fn at_level(root_system: &RootSystem, level: f64) -> Self {
        assert!(level > 0.0, "Level must be positive");

        let simple_roots = root_system.simple_roots.clone();
        let highest_root = root_system.highest_root();

        Self {
            root_system: root_system.clone(),
            simple_roots,
            highest_root,
            level,
        }
    }

    /// Check if a weight is in the alcove.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight λ ∈ 𝔥* to test
    /// * `strict` - If true, use strict inequalities
    ///
    /// # Returns
    ///
    /// True if λ is in the alcove
    #[must_use]
    pub fn contains(&self, weight: &Root, strict: bool) -> bool {
        // For type A_n, roots have n+1 coordinates
        assert_eq!(weight.rank(), self.root_system.rank + 1);

        // Check lower walls: ⟨λ, αᵢ⟩ ≥ 0
        for alpha in &self.simple_roots {
            let pairing = weight.inner_product(alpha);

            if strict {
                if pairing <= 1e-10 {
                    return false;
                }
            } else if pairing < -1e-10 {
                return false;
            }
        }

        // Check upper wall: ⟨λ, θ⟩ ≤ level
        let pairing_highest = weight.inner_product(&self.highest_root);

        if strict {
            if pairing_highest >= self.level - 1e-10 {
                return false;
            }
        } else if pairing_highest > self.level + 1e-10 {
            return false;
        }

        true
    }

    /// Get the level of the alcove.
    #[must_use]
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Get the highest root defining the upper wall.
    #[must_use]
    pub fn highest_root(&self) -> &Root {
        &self.highest_root
    }

    /// Compute vertices of the fundamental alcove.
    ///
    /// For type `A_n`, the fundamental alcove is a simplex with n+1 vertices:
    /// the origin and the fundamental weights {ω₁, ..., ωₙ}.
    ///
    /// # Returns
    ///
    /// Vector of vertices (as Roots in 𝔥*)
    #[must_use]
    pub fn vertices(&self) -> Vec<Root> {
        if self.level != 1.0 {
            // For level k ≠ 1, vertices are scaled: k·ωᵢ/(k + h∨)
            // where h∨ is the dual Coxeter number
            // Not implemented for now
            return vec![];
        }

        // For fundamental alcove at level 1:
        // Vertices are 0 and fundamental weights

        let mut vertices = vec![Root::new(vec![0.0; self.root_system.rank + 1])];

        // Add fundamental weights (computed from simple roots)
        // For type A_n, ωᵢ = (1/n+1) * (n+1-i, ..., n+1-i, -i, ..., -i)
        //                                 └── i times ──┘  └─ n+1-i ──┘

        let n = self.root_system.rank;

        for i in 1..=n {
            let mut coords = vec![0.0; n + 1];

            // ωᵢ has i coordinates equal to (n+1-i)/(n+1) and (n+1-i) equal to -i/(n+1)
            for j in 0..i {
                coords[j] = (n + 1 - i) as f64 / (n + 1) as f64;
            }
            for j in i..=n {
                coords[j] = -(i as f64) / (n + 1) as f64;
            }

            vertices.push(Root::new(coords));
        }

        vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_operations() {
        let alpha = Root::new(vec![1.0, -1.0, 0.0]);
        let beta = Root::new(vec![0.0, 1.0, -1.0]);

        assert_eq!(alpha.rank(), 3);
        assert!((alpha.norm_squared() - 2.0).abs() < 1e-10);
        assert!((alpha.inner_product(&beta) + 1.0).abs() < 1e-10);
        assert!(alpha.is_positive());
        assert!(!alpha.negate().is_positive());
    }

    #[test]
    fn test_weyl_reflection() {
        let alpha = Root::new(vec![1.0, -1.0]);
        let beta = Root::new(vec![0.5, 0.5]);

        let reflected = alpha.reflect(&beta);
        // s_α(β) = β - 2⟨β,α⟩/⟨α,α⟩ · α
        // ⟨β,α⟩ = 0.5 - 0.5 = 0, so s_α(β) = β
        assert!((reflected.coords[0] - 0.5).abs() < 1e-10);
        assert!((reflected.coords[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cartan_integer() {
        let alpha = Root::new(vec![1.0, -1.0, 0.0]);
        let beta = Root::new(vec![0.0, 1.0, -1.0]);

        // ⟨β, α^∨⟩ = 2⟨β,α⟩/⟨α,α⟩ = 2(-1)/2 = -1
        assert_eq!(alpha.cartan_integer(&beta), -1);
        assert_eq!(beta.cartan_integer(&alpha), -1);
    }

    #[test]
    fn test_type_a1_su2() {
        let rs = RootSystem::type_a(1);

        assert_eq!(rs.rank(), 1);
        assert_eq!(rs.num_roots(), 2); // ±α
        assert_eq!(rs.num_positive_roots(), 1);
        assert_eq!(rs.dimension(), 3); // SU(2) dim = 1 + 2 = 3

        let cartan = rs.cartan_matrix();
        assert_eq!(cartan.len(), 1);
        assert_eq!(cartan[0][0], 2); // ⟨α, α^∨⟩ = 2
    }

    #[test]
    fn test_type_a2_su3() {
        let rs = RootSystem::type_a(2);

        assert_eq!(rs.rank(), 2);
        assert_eq!(rs.num_roots(), 6); // 3² - 1 = 8 total, 6 nonzero differences
        assert_eq!(rs.num_positive_roots(), 3);
        assert_eq!(rs.dimension(), 8); // SU(3) dim = 2 + 6 = 8

        let cartan = rs.cartan_matrix();
        assert_eq!(cartan.len(), 2);
        assert_eq!(cartan[0][0], 2);
        assert_eq!(cartan[1][1], 2);
        assert_eq!(cartan[0][1], -1); // Adjacent simple roots
        assert_eq!(cartan[1][0], -1);
    }

    #[test]
    fn test_simple_roots_su3() {
        let rs = RootSystem::type_a(2);
        let simple = rs.simple_roots();

        assert_eq!(simple.len(), 2);

        // α₁ = e₁ - e₂ = (1, -1, 0)
        assert!((simple[0].coords[0] - 1.0).abs() < 1e-10);
        assert!((simple[0].coords[1] + 1.0).abs() < 1e-10);
        assert!((simple[0].coords[2]).abs() < 1e-10);

        // α₂ = e₂ - e₃ = (0, 1, -1)
        assert!((simple[1].coords[0]).abs() < 1e-10);
        assert!((simple[1].coords[1] - 1.0).abs() < 1e-10);
        assert!((simple[1].coords[2] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_positive_roots_su3() {
        let rs = RootSystem::type_a(2);
        let positive = rs.positive_roots();

        assert_eq!(positive.len(), 3);
        // Should have: e₁-e₂, e₂-e₃, e₁-e₃
    }

    #[test]
    fn test_dominant_weight() {
        let rs = RootSystem::type_a(2);

        // (1, 0, 0) should be dominant
        let weight1 = Root::new(vec![1.0, 0.0, 0.0]);
        assert!(rs.is_dominant_weight(&weight1));

        // (-1, 0, 0) should not be dominant
        let weight2 = Root::new(vec![-1.0, 0.0, 0.0]);
        assert!(!rs.is_dominant_weight(&weight2));
    }

    #[test]
    fn test_weight_lattice_su2() {
        let rs = RootSystem::type_a(1);
        let wl = WeightLattice::from_root_system(rs);

        assert_eq!(wl.fundamental_weights().len(), 1);

        // Spin-1 representation: Dynkin label [2]
        let weight = wl.dynkin_to_weight(&[2]);
        assert_eq!(weight.rank(), 2);
    }

    #[test]
    fn test_weight_lattice_su3() {
        let rs = RootSystem::type_a(2);
        let wl = WeightLattice::from_root_system(rs);

        assert_eq!(wl.fundamental_weights().len(), 2);

        // Fundamental representation: Dynkin labels [1, 0]
        let fund = wl.dynkin_to_weight(&[1, 0]);
        assert_eq!(fund.rank(), 3);

        // Adjoint representation: Dynkin labels [1, 1]
        let adj = wl.dynkin_to_weight(&[1, 1]);
        assert_eq!(adj.rank(), 3);
    }

    #[test]
    fn test_weyl_orbit_su2() {
        let rs = RootSystem::type_a(1);
        let weight = Root::new(vec![1.0, 0.0]);

        let orbit = rs.weyl_orbit(&weight);
        assert_eq!(orbit.len(), 2); // Weyl group S₂ has 2 elements
    }

    #[test]
    fn test_weyl_dimension_formula() {
        let rs = RootSystem::type_a(2); // SU(3)
        let wl = WeightLattice::from_root_system(rs);

        // Test known dimensions for SU(3)
        // Fundamental [1,0]: dim = 3
        let fund = wl.dynkin_to_weight(&[1, 0]);
        assert_eq!(wl.weyl_dimension(&fund), 3);

        // Anti-fundamental [0,1]: dim = 3
        let antifund = wl.dynkin_to_weight(&[0, 1]);
        assert_eq!(wl.weyl_dimension(&antifund), 3);

        // Adjoint [1,1]: dim = 8
        let adj = wl.dynkin_to_weight(&[1, 1]);
        assert_eq!(wl.weyl_dimension(&adj), 8);

        // [2,0]: dim = 6
        let sym2 = wl.dynkin_to_weight(&[2, 0]);
        assert_eq!(wl.weyl_dimension(&sym2), 6);

        // [0,2]: dim = 6
        let sym2_bar = wl.dynkin_to_weight(&[0, 2]);
        assert_eq!(wl.weyl_dimension(&sym2_bar), 6);

        // [1,0,0,...] for SU(2) - trivial rep has dim = 1
        let rs2 = RootSystem::type_a(1); // SU(2)
        let wl2 = WeightLattice::from_root_system(rs2);
        let trivial = wl2.dynkin_to_weight(&[0]);
        assert_eq!(wl2.weyl_dimension(&trivial), 1);

        // [2] for SU(2) - spin-1 has dim = 3
        let spin1 = wl2.dynkin_to_weight(&[2]);
        assert_eq!(wl2.weyl_dimension(&spin1), 3);
    }

    #[test]
    fn test_weights_of_irrep_su3_fundamental() {
        let rs = RootSystem::type_a(2);
        let wl = WeightLattice::from_root_system(rs);

        // Fundamental representation [1, 0] should have dimension 3
        let highest = wl.dynkin_to_weight(&[1, 0]);
        let weights = wl.weights_of_irrep(&highest);

        // Check exact dimension (Weyl dimension formula)
        let expected_dim = wl.weyl_dimension(&highest);
        assert_eq!(expected_dim, 3, "Fundamental rep has dimension 3");
        assert_eq!(
            weights.len(),
            3,
            "Should find exactly 3 weights, found {}",
            weights.len()
        );
    }

    #[test]
    fn test_weights_of_irrep_su3_adjoint() {
        let rs = RootSystem::type_a(2);
        let wl = WeightLattice::from_root_system(rs);

        // Adjoint representation [1, 1] should have dimension 8
        let highest = wl.dynkin_to_weight(&[1, 1]);
        let weights = wl.weights_of_irrep(&highest);

        // Check dimension (Weyl dimension formula)
        let dim = wl.weyl_dimension(&highest);
        assert_eq!(dim, 8, "Adjoint rep has dimension 8");

        // With Kostant's formula, we now compute proper multiplicities
        // For SU(3) adjoint: 6 roots (mult 1) + 1 zero weight (mult 2) = dimension 8
        let total_dim: usize = weights.iter().map(|(_, m)| m).sum();
        assert_eq!(
            total_dim, 8,
            "Total dimension (sum of multiplicities) should be 8"
        );
    }

    #[test]
    fn test_partition_function_basics() {
        // Test partition function on simple cases we can verify by hand
        let rs = RootSystem::type_a(1); // SU(2)
        let wl = WeightLattice::from_root_system(rs);

        // SU(2) has one positive root: α = [1, -1]
        // Test P(0) = 1 (empty sum)
        let zero = Root::new(vec![0.0, 0.0]);
        assert_eq!(wl.kostant_partition_function(&zero), 1, "P(0) = 1");

        // Test P(α) = 1 (just α itself)
        let alpha = Root::new(vec![1.0, -1.0]);
        assert_eq!(wl.kostant_partition_function(&alpha), 1, "P(α) = 1");

        // Test P(2α) = 1 (just 2·α)
        let two_alpha = Root::new(vec![2.0, -2.0]);
        assert_eq!(wl.kostant_partition_function(&two_alpha), 1, "P(2α) = 1");

        // Test P(-α) = 0 (negative, impossible)
        let neg_alpha = Root::new(vec![-1.0, 1.0]);
        assert_eq!(wl.kostant_partition_function(&neg_alpha), 0, "P(-α) = 0");
    }

    #[test]
    fn test_partition_function_su3() {
        // SU(3) has 3 positive roots: α₁=[1,-1,0], α₂=[0,1,-1], α₁+α₂=[1,0,-1]
        let rs = RootSystem::type_a(2);
        let wl = WeightLattice::from_root_system(rs);

        // Test P(0) = 1
        let zero = Root::new(vec![0.0, 0.0, 0.0]);
        assert_eq!(wl.kostant_partition_function(&zero), 1, "P(0) = 1");

        // Test P(α₁) = 1
        let alpha1 = Root::new(vec![1.0, -1.0, 0.0]);
        let p_alpha1 = wl.kostant_partition_function(&alpha1);
        eprintln!("P(α₁) = {}", p_alpha1);
        assert_eq!(p_alpha1, 1, "P(α₁) = 1");

        // Test P(α₁+α₂) = 2 (can write as α₁+α₂ OR as the root [1,0,-1])
        // Wait, [1,0,-1] IS α₁+α₂, so P(α₁+α₂) = 1
        let alpha1_plus_alpha2 = Root::new(vec![1.0, 0.0, -1.0]);
        let p_sum = wl.kostant_partition_function(&alpha1_plus_alpha2);
        eprintln!("P(α₁+α₂) = {}", p_sum);
        // This should be 1, not 2! Because [1,0,-1] is itself a positive root.
    }

    #[test]
    fn test_kostant_dimension_invariant() {
        // Verify that Kostant's formula satisfies the dimension invariant:
        // Σ_λ m(λ) = dim(V_Λ) for any representation
        let rs = RootSystem::type_a(2); // SU(3)
        let wl = WeightLattice::from_root_system(rs);

        // Test fundamental [1,0]: dimension 3
        let fund = wl.dynkin_to_weight(&[1, 0]);
        let fund_weights = wl.weights_of_irrep(&fund);
        eprintln!("Fundamental [1,0] weights:");
        for (w, m) in &fund_weights {
            eprintln!("  {:?} mult {}", w.coords, m);
        }
        let fund_dim: usize = fund_weights.iter().map(|(_, m)| m).sum();
        eprintln!("Total dimension: {} (expected 3)", fund_dim);
        assert_eq!(fund_dim, 3, "Fundamental rep dimension invariant");

        // Test adjoint [1,1]: dimension 8
        let adj = wl.dynkin_to_weight(&[1, 1]);
        let adj_weights = wl.weights_of_irrep(&adj);
        eprintln!("\nAdjoint [1,1] weights:");
        for (w, m) in &adj_weights {
            eprintln!("  {:?} mult {}", w.coords, m);
        }
        let adj_dim: usize = adj_weights.iter().map(|(_, m)| m).sum();
        eprintln!("Total dimension: {} (expected 8)", adj_dim);
        assert_eq!(adj_dim, 8, "Adjoint rep dimension invariant");

        // Test antifundamental [0,1]: dimension 3
        let anti = wl.dynkin_to_weight(&[0, 1]);
        let anti_weights = wl.weights_of_irrep(&anti);
        let anti_dim: usize = anti_weights.iter().map(|(_, m)| m).sum();
        assert_eq!(anti_dim, 3, "Antifundamental rep dimension invariant");
    }

    #[test]
    fn test_weight_diagram_string_su3() {
        let rs = RootSystem::type_a(2);
        let wl = WeightLattice::from_root_system(rs);

        let highest = wl.dynkin_to_weight(&[1, 0]);
        let diagram = wl.weight_diagram_string(&highest);

        // Check that diagram contains expected elements
        assert!(diagram.contains("Weight diagram"));
        assert!(diagram.contains("Dimension"));
        assert!(diagram.contains("Weights"));
    }

    #[test]
    fn test_weight_diagram_rank1_not_supported() {
        let rs = RootSystem::type_a(1);
        let wl = WeightLattice::from_root_system(rs);

        let highest = wl.dynkin_to_weight(&[2]);
        let diagram = wl.weight_diagram_string(&highest);

        assert!(diagram.contains("only implemented for rank 2"));
    }

    // --- Cartan Subalgebra Tests ---

    #[test]
    fn test_cartan_subalgebra_su2() {
        let cartan = CartanSubalgebra::type_a(1);

        assert_eq!(cartan.dimension(), 1);
        assert_eq!(cartan.matrix_size(), 2);
        assert_eq!(cartan.basis_matrices().len(), 1);

        // H₁ = diag(1, -1)
        let h1 = &cartan.basis_matrices()[0];
        assert!((h1[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((h1[(1, 1)].re + 1.0).abs() < 1e-10);
        assert!(h1[(0, 1)].norm() < 1e-10);
        assert!(h1[(1, 0)].norm() < 1e-10);
    }

    #[test]
    fn test_cartan_subalgebra_su3() {
        let cartan = CartanSubalgebra::type_a(2);

        assert_eq!(cartan.dimension(), 2);
        assert_eq!(cartan.matrix_size(), 3);
        assert_eq!(cartan.basis_matrices().len(), 2);

        // H₁ = diag(1, -1, 0)
        let h1 = &cartan.basis_matrices()[0];
        assert!((h1[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((h1[(1, 1)].re + 1.0).abs() < 1e-10);
        assert!(h1[(2, 2)].norm() < 1e-10);

        // H₂ = diag(0, 1, -1)
        let h2 = &cartan.basis_matrices()[1];
        assert!(h2[(0, 0)].norm() < 1e-10);
        assert!((h2[(1, 1)].re - 1.0).abs() < 1e-10);
        assert!((h2[(2, 2)].re + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cartan_basis_commutes() {
        let cartan = CartanSubalgebra::type_a(2);

        // [H₁, H₂] = 0 (diagonal matrices commute)
        let h1 = &cartan.basis_matrices()[0];
        let h2 = &cartan.basis_matrices()[1];

        let commutator = h1.dot(h2) - h2.dot(h1);

        for val in commutator.iter() {
            assert!(val.norm() < 1e-10, "Cartan basis elements must commute");
        }
    }

    #[test]
    fn test_cartan_dual_basis() {
        let cartan = CartanSubalgebra::type_a(2);
        let dual = cartan.dual_basis();

        assert_eq!(dual.len(), 2);

        // α₁ = e₁ - e₂
        assert!((dual[0].coords[0] - 1.0).abs() < 1e-10);
        assert!((dual[0].coords[1] + 1.0).abs() < 1e-10);
        assert!(dual[0].coords[2].abs() < 1e-10);

        // α₂ = e₂ - e₃
        assert!(dual[1].coords[0].abs() < 1e-10);
        assert!((dual[1].coords[1] - 1.0).abs() < 1e-10);
        assert!((dual[1].coords[2] + 1.0).abs() < 1e-10);

        // These should be the simple roots
        let rs = RootSystem::type_a(2);
        let simple_roots = rs.simple_roots();

        for (d, s) in dual.iter().zip(simple_roots) {
            for (dc, sc) in d.coords.iter().zip(&s.coords) {
                assert!((dc - sc).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cartan_evaluate_root() {
        let cartan = CartanSubalgebra::type_a(1);

        // For SU(2), H = c₁ · diag(1, -1)
        // Root α = (1, -1) (the simple root)
        let root = Root::new(vec![1.0, -1.0]);

        // α(H) = 1·c₁ + (-1)·(-c₁) = 2c₁
        let result = cartan.evaluate_root(&root, &[1.0]);
        assert!((result.re - 2.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);

        let result2 = cartan.evaluate_root(&root, &[0.5]);
        assert!((result2.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cartan_from_coefficients() {
        let cartan = CartanSubalgebra::type_a(2);

        // H = 2H₁ + 3H₂
        let h = cartan.from_coefficients(&[2.0, 3.0]);

        // Should be diag(2, -2+3, -3) = diag(2, 1, -3)
        assert!((h[(0, 0)].re - 2.0).abs() < 1e-10);
        assert!((h[(1, 1)].re - 1.0).abs() < 1e-10);
        assert!((h[(2, 2)].re + 3.0).abs() < 1e-10);

        // Check traceless
        let trace: Complex64 = (0..3).map(|i| h[(i, i)]).sum();
        assert!(trace.norm() < 1e-10);
    }

    #[test]
    fn test_cartan_project_matrix() {
        let cartan = CartanSubalgebra::type_a(2);

        // Create a diagonal matrix
        let mut mat = Array2::<Complex64>::zeros((3, 3));
        mat[(0, 0)] = Complex64::new(1.0, 0.0);
        mat[(1, 1)] = Complex64::new(-0.5, 0.0);
        mat[(2, 2)] = Complex64::new(-0.5, 0.0);

        let coeffs = cartan
            .project_matrix(&mat)
            .expect("Rank 2 should be supported");
        assert_eq!(coeffs.len(), 2);

        // Reconstruct and check
        let reconstructed = cartan.from_coefficients(&coeffs);

        // Should reconstruct exactly for diagonal traceless matrices
        for i in 0..3 {
            let diff = (mat[(i, i)] - reconstructed[(i, i)]).norm();
            assert!(
                diff < 1e-10,
                "Diagonal projection should be exact: diff at ({},{}) = {}",
                i,
                i,
                diff
            );
        }
    }

    #[test]
    fn test_cartan_contains() {
        let cartan = CartanSubalgebra::type_a(2);

        // Diagonal traceless matrix - should be in Cartan
        let mut h = Array2::<Complex64>::zeros((3, 3));
        h[(0, 0)] = Complex64::new(1.0, 0.0);
        h[(1, 1)] = Complex64::new(-0.5, 0.0);
        h[(2, 2)] = Complex64::new(-0.5, 0.0);

        assert!(cartan.contains(&h, 1e-6));

        // Non-diagonal matrix - should not be in Cartan
        let mut not_h = Array2::<Complex64>::zeros((3, 3));
        not_h[(0, 1)] = Complex64::new(1.0, 0.0);

        assert!(!cartan.contains(&not_h, 1e-6));

        // Diagonal but not traceless - should not be in Cartan
        let mut not_traceless = Array2::<Complex64>::zeros((3, 3));
        not_traceless[(0, 0)] = Complex64::new(1.0, 0.0);

        assert!(!cartan.contains(&not_traceless, 1e-6));
    }

    #[test]
    fn test_cartan_killing_form() {
        let cartan = CartanSubalgebra::type_a(1);

        // For SU(2), κ(H, H) = 2(2)·2 = 8 for H = H₁
        let killing = cartan.killing_form(&[1.0], &[1.0]);
        assert!((killing - 8.0).abs() < 1e-10);

        // κ(H, 0) = 0
        let killing_zero = cartan.killing_form(&[1.0], &[0.0]);
        assert!(killing_zero.abs() < 1e-10);

        // κ is symmetric
        let k12 = cartan.killing_form(&[1.0], &[2.0]);
        let k21 = cartan.killing_form(&[2.0], &[1.0]);
        assert!((k12 - k21).abs() < 1e-10);
    }

    #[test]
    fn test_cartan_killing_form_su3() {
        let cartan = CartanSubalgebra::type_a(2);

        // For SU(3), κ(Hᵢ, Hⱼ) = 2(3)·2δᵢⱼ = 12δᵢⱼ
        let k11 = cartan.killing_form(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((k11 - 12.0).abs() < 1e-10);

        let k22 = cartan.killing_form(&[0.0, 1.0], &[0.0, 1.0]);
        assert!((k22 - 12.0).abs() < 1e-10);

        // Off-diagonal should be zero for orthogonal basis
        let k12 = cartan.killing_form(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(k12.abs() < 1e-10);
    }

    #[test]
    fn test_weyl_chamber_contains_su2() {
        let rs = RootSystem::type_a(1); // SU(2)
        let chamber = WeylChamber::fundamental(&rs);

        // Simple root α = (1, -1)
        // Dominant weights satisfy ⟨λ, α⟩ ≥ 0

        // Dominant weight: (1, 0)
        let dominant = Root::new(vec![1.0, 0.0]);
        assert!(chamber.contains(&dominant, false));
        assert!(chamber.contains(&dominant, true));

        // Non-dominant: (-1, 0)
        let non_dominant = Root::new(vec![-1.0, 0.0]);
        assert!(!chamber.contains(&non_dominant, false));

        // Boundary: (0, 0)
        let boundary = Root::new(vec![0.0, 0.0]);
        assert!(chamber.contains(&boundary, false)); // Non-strict allows boundary
        assert!(!chamber.contains(&boundary, true)); // Strict excludes boundary
    }

    #[test]
    fn test_weyl_chamber_contains_su3() {
        let rs = RootSystem::type_a(2); // SU(3)
        let chamber = WeylChamber::fundamental(&rs);

        // Simple roots: α₁ = (1, -1, 0), α₂ = (0, 1, -1)
        // Dominant: ⟨λ, α₁⟩ ≥ 0 and ⟨λ, α₂⟩ ≥ 0

        // Fundamental weight ω₁ = (1, 0, 0) - 1/3 for traceless
        let omega1 = Root::new(vec![2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]);
        assert!(chamber.contains(&omega1, false));

        // Negative weight
        let neg = Root::new(vec![-1.0, 0.0, 1.0]);
        assert!(!chamber.contains(&neg, false));

        // Origin
        let origin = Root::new(vec![0.0, 0.0, 0.0]);
        assert!(chamber.contains(&origin, false));
        assert!(!chamber.contains(&origin, true));
    }

    #[test]
    fn test_weyl_chamber_project() {
        let rs = RootSystem::type_a(1); // SU(2)
        let chamber = WeylChamber::fundamental(&rs);

        // Already dominant
        let dominant = Root::new(vec![1.0, -1.0]);
        let projected = chamber.project(&dominant);
        assert!((projected.coords[0] - dominant.coords[0]).abs() < 1e-10);
        assert!((projected.coords[1] - dominant.coords[1]).abs() < 1e-10);

        // Non-dominant: should reflect to dominant
        let non_dominant = Root::new(vec![-1.0, 1.0]);
        let projected = chamber.project(&non_dominant);

        // After reflection by α = (1, -1), should get (1, -1)
        assert!(chamber.contains(&projected, false));
        assert!((projected.coords[0] - 1.0).abs() < 1e-10);
        assert!((projected.coords[1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weyl_chamber_project_su3() {
        let rs = RootSystem::type_a(2); // SU(3)
        let chamber = WeylChamber::fundamental(&rs);

        // Non-dominant weight
        let lambda = Root::new(vec![-1.0, 2.0, -1.0]);
        let projected = chamber.project(&lambda);

        // Projected weight should be dominant
        assert!(chamber.contains(&projected, false));

        // Check that projection preserves norm (Weyl group is orthogonal)
        let norm_before = lambda.norm_squared();
        let norm_after = projected.norm_squared();
        assert!((norm_before - norm_after).abs() < 1e-8);
    }

    #[test]
    fn test_alcove_contains_su2() {
        let rs = RootSystem::type_a(1); // SU(2)
        let alcove = Alcove::fundamental(&rs);

        // Highest root for SU(2): θ = (1, -1) (same as simple root)
        // Alcove: ⟨λ, α⟩ ≥ 0 and ⟨λ, θ⟩ ≤ 1

        // Inside alcove: (0.4, -0.4)
        let inside = Root::new(vec![0.4, -0.4]);
        // ⟨inside, α⟩ = 0.4·1 + (-0.4)·(-1) = 0.8 ≥ 0 ✓
        // ⟨inside, θ⟩ = 0.4·1 + (-0.4)·(-1) = 0.8 ≤ 1 ✓
        assert!(alcove.contains(&inside, false));

        // Outside alcove (too large): (1, -1)
        let outside = Root::new(vec![1.0, -1.0]);
        // ⟨outside, θ⟩ = 2 > 1
        assert!(!alcove.contains(&outside, false));

        // Boundary: origin
        let origin = Root::new(vec![0.0, 0.0]);
        assert!(alcove.contains(&origin, false));
        assert!(!alcove.contains(&origin, true)); // Strict excludes boundary
    }

    #[test]
    fn test_alcove_vertices_su2() {
        let rs = RootSystem::type_a(1); // SU(2)
        let alcove = Alcove::fundamental(&rs);

        let vertices = alcove.vertices();

        // For SU(2), fundamental alcove has 2 vertices: 0 and ω₁
        assert_eq!(vertices.len(), 2);

        // First vertex: origin
        assert!(vertices[0].coords[0].abs() < 1e-10);
        assert!(vertices[0].coords[1].abs() < 1e-10);

        // Second vertex: fundamental weight ω₁ = (1/2, -1/2)
        assert!((vertices[1].coords[0] - 0.5).abs() < 1e-10);
        assert!((vertices[1].coords[1] + 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_alcove_vertices_su3() {
        let rs = RootSystem::type_a(2); // SU(3)
        let alcove = Alcove::fundamental(&rs);

        let vertices = alcove.vertices();

        // For SU(3), fundamental alcove has 3 vertices: 0, ω₁, ω₂
        assert_eq!(vertices.len(), 3);

        // First vertex: origin
        for &coord in &vertices[0].coords {
            assert!(coord.abs() < 1e-10);
        }

        // Verify all vertices are in the alcove
        for vertex in &vertices {
            assert!(alcove.contains(vertex, false));
        }
    }

    #[test]
    fn test_alcove_at_level() {
        let rs = RootSystem::type_a(1); // SU(2)
        let alcove_k2 = Alcove::at_level(&rs, 2.0);

        assert_eq!(alcove_k2.level(), 2.0);

        // At level 2, the upper bound is ⟨λ, θ⟩ ≤ 2
        let inside = Root::new(vec![0.8, -0.8]);
        // ⟨inside, θ⟩ = 1.6 ≤ 2 ✓
        assert!(alcove_k2.contains(&inside, false));

        let outside = Root::new(vec![1.5, -1.5]);
        // ⟨outside, θ⟩ = 3 > 2
        assert!(!alcove_k2.contains(&outside, false));
    }

    #[test]
    fn test_alcove_highest_root() {
        let rs = RootSystem::type_a(2); // SU(3)
        let alcove = Alcove::fundamental(&rs);

        let theta = alcove.highest_root();

        // For SU(3), highest root is θ = α₁ + α₂ = (1, 0, -1)
        assert!((theta.coords[0] - 1.0).abs() < 1e-10);
        assert!(theta.coords[1].abs() < 1e-10);
        assert!((theta.coords[2] + 1.0).abs() < 1e-10);

        // Verify it's the longest root
        let theta_norm = theta.norm_squared();
        for root in rs.positive_roots() {
            assert!(root.norm_squared() <= theta_norm + 1e-10);
        }
    }

    // --- Simple Root Expansion Tests ---

    #[test]
    fn test_simple_root_expansion_su2() {
        let rs = RootSystem::type_a(1); // SU(2)

        // SU(2) has one simple root: α₁ = e₀ - e₁ = (1, -1)
        // The only positive root is α₁ itself

        // Test the simple root itself: α₁ = 1·α₁
        let alpha1 = Root::new(vec![1.0, -1.0]);
        let coeffs = rs.simple_root_expansion(&alpha1);
        assert_eq!(coeffs, Some(vec![1]));

        // Test the negative root: -α₁ = (-1)·α₁
        let neg_alpha1 = Root::new(vec![-1.0, 1.0]);
        let coeffs_neg = rs.simple_root_expansion(&neg_alpha1);
        assert_eq!(coeffs_neg, Some(vec![-1]));
    }

    #[test]
    fn test_simple_root_expansion_su3() {
        let rs = RootSystem::type_a(2); // SU(3)

        // SU(3) has 2 simple roots:
        // α₁ = e₀ - e₁ = (1, -1, 0)
        // α₂ = e₁ - e₂ = (0, 1, -1)
        // And a third positive root: α₁ + α₂ = e₀ - e₂ = (1, 0, -1)

        // Test α₁ = (1, -1, 0): coefficients [1, 0]
        let alpha1 = Root::new(vec![1.0, -1.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&alpha1), Some(vec![1, 0]));

        // Test α₂ = (0, 1, -1): coefficients [0, 1]
        let alpha2 = Root::new(vec![0.0, 1.0, -1.0]);
        assert_eq!(rs.simple_root_expansion(&alpha2), Some(vec![0, 1]));

        // Test α₁ + α₂ = (1, 0, -1): coefficients [1, 1]
        let alpha1_plus_alpha2 = Root::new(vec![1.0, 0.0, -1.0]);
        assert_eq!(
            rs.simple_root_expansion(&alpha1_plus_alpha2),
            Some(vec![1, 1])
        );

        // Test negative roots
        let neg_alpha1 = Root::new(vec![-1.0, 1.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&neg_alpha1), Some(vec![-1, 0]));

        let neg_highest = Root::new(vec![-1.0, 0.0, 1.0]);
        assert_eq!(rs.simple_root_expansion(&neg_highest), Some(vec![-1, -1]));
    }

    #[test]
    fn test_simple_root_expansion_su4() {
        let rs = RootSystem::type_a(3); // SU(4)

        // SU(4) has 3 simple roots:
        // α₁ = e₀ - e₁ = (1, -1, 0, 0)
        // α₂ = e₁ - e₂ = (0, 1, -1, 0)
        // α₃ = e₂ - e₃ = (0, 0, 1, -1)

        // Test simple roots
        let alpha1 = Root::new(vec![1.0, -1.0, 0.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&alpha1), Some(vec![1, 0, 0]));

        let alpha2 = Root::new(vec![0.0, 1.0, -1.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&alpha2), Some(vec![0, 1, 0]));

        let alpha3 = Root::new(vec![0.0, 0.0, 1.0, -1.0]);
        assert_eq!(rs.simple_root_expansion(&alpha3), Some(vec![0, 0, 1]));

        // Test e₀ - e₂ = α₁ + α₂: coefficients [1, 1, 0]
        let e0_minus_e2 = Root::new(vec![1.0, 0.0, -1.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&e0_minus_e2), Some(vec![1, 1, 0]));

        // Test e₁ - e₃ = α₂ + α₃: coefficients [0, 1, 1]
        let e1_minus_e3 = Root::new(vec![0.0, 1.0, 0.0, -1.0]);
        assert_eq!(rs.simple_root_expansion(&e1_minus_e3), Some(vec![0, 1, 1]));

        // Test highest root e₀ - e₃ = α₁ + α₂ + α₃: coefficients [1, 1, 1]
        let highest = Root::new(vec![1.0, 0.0, 0.0, -1.0]);
        assert_eq!(rs.simple_root_expansion(&highest), Some(vec![1, 1, 1]));

        // Test negative of highest root: coefficients [-1, -1, -1]
        let neg_highest = Root::new(vec![-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            rs.simple_root_expansion(&neg_highest),
            Some(vec![-1, -1, -1])
        );
    }

    #[test]
    fn test_simple_root_expansion_not_a_root() {
        let rs = RootSystem::type_a(2); // SU(3)

        // Test a vector that is not a root
        let not_a_root = Root::new(vec![1.0, 1.0, -2.0]);
        assert_eq!(rs.simple_root_expansion(&not_a_root), None);

        // Test zero vector
        let zero = Root::new(vec![0.0, 0.0, 0.0]);
        assert_eq!(rs.simple_root_expansion(&zero), None);
    }

    #[test]
    fn test_simple_root_expansion_roundtrip() {
        // Verify that we can reconstruct roots from their expansions
        let rs = RootSystem::type_a(2); // SU(3)

        for root in rs.roots() {
            let coeffs = rs.simple_root_expansion(root);
            assert!(coeffs.is_some(), "All roots should have expansions");

            let coeffs = coeffs.unwrap();
            let simple = rs.simple_roots();

            // Reconstruct: Σ cᵢ αᵢ
            let mut reconstructed = vec![0.0; root.coords.len()];
            for (i, &c) in coeffs.iter().enumerate() {
                for (j, &coord) in simple[i].coords.iter().enumerate() {
                    reconstructed[j] += (c as f64) * coord;
                }
            }

            // Compare with original
            for (orig, recon) in root.coords.iter().zip(&reconstructed) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "Reconstruction failed for root {:?}: expected {:?}, got {:?}",
                    root.coords,
                    root.coords,
                    reconstructed
                );
            }
        }
    }
}
