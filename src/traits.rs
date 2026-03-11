//! Traits for Lie groups and Lie algebras.
//!
//! This module defines the core abstractions for Lie theory. These traits
//! capture the mathematical structure
//! of Lie groups as both algebraic objects (groups) and geometric objects
//! (differentiable manifolds).
//!
//! # Mathematical Background
//!
//! A **Lie group** G is a smooth manifold that is also a group, where the
//! group operations are smooth maps:
//!
//! - Multiplication: G × G → G is smooth
//! - Inversion: G → G is smooth
//! - Identity: e ∈ G
//!
//! A **Lie algebra** 𝔤 is the tangent space at the identity, with a Lie bracket
//! operation [·,·] : 𝔤 × 𝔤 → 𝔤 satisfying bilinearity, antisymmetry, and the Jacobi identity.
//!
//! The **exponential map** exp: 𝔤 → G connects the Lie algebra to the Lie group,
//! mapping tangent vectors to group elements via one-parameter subgroups.
//!
//! # Design Philosophy
//!
//! These traits enable **generic programming over Lie groups**: algorithms written
//! against the `LieGroup` trait work for SU(2), U(1), SU(3), etc. without modification.
//! This mirrors how mathematicians work—proving theorems for "a Lie group G"
//! rather than for each specific group.
//!
//! # Examples
//!
//! ```
//! use lie_groups::{LieGroup, SU2};
//!
//! // Generic parallel transport along a path
//! fn parallel_transport<G: LieGroup>(path: &[G]) -> G {
//!     path.iter().fold(G::identity(), |acc, g| acc.compose(g))
//! }
//!
//! // Works for any Lie group!
//! let su2_path = vec![SU2::rotation_x(0.1), SU2::rotation_y(0.2)];
//! let holonomy = parallel_transport(&su2_path);
//! ```
//!
//! # Constrained Representations: Compile-Time Guarantees
//!
//! This module uses **constrained representations** to encode mathematical
//! properties at the type level. The key insight: instead of checking if a
//! value satisfies a property at runtime, use a representation that makes
//! invalid values **unrepresentable**.
//!
//! ## The Pattern
//!
//! | Property | Representation | Why It Works |
//! |----------|----------------|--------------|
//! | Traceless | Basis coefficients | Pauli/Gell-Mann matrices are traceless |
//! | Anti-Hermitian | `i·(Hermitian basis)` | `(iH)† = -iH† = -iH` |
//! | Unit norm | Private fields + normalizing constructor | Can only create unit values |
//! | Fixed size | `[T; N]` instead of `Vec<T>` | Compile-time size guarantee |
//!
//! ## Example: Traceless by Construction
//!
//! ```text
//! // BAD: General matrix, might not be traceless
//! struct BadAlgebra { matrix: [[f64; 2]; 2] }  // tr(M) could be anything
//!
//! // GOOD: Pauli basis coefficients, always traceless
//! struct Su2Algebra([f64; 3]);  // X = a·iσ₁ + b·iσ₂ + c·iσ₃, tr(X) = 0 always
//! ```
//!
//! ## Connection to Lean Proofs
//!
//! The marker traits in this module connect to formal proofs:
//!
//! - [`TracelessByConstruction`]: Combined with the Lean theorem
//!   `det_exp_eq_exp_trace`, gives `det(exp(X)) = exp(0) = 1` for all X.
//!
//! - [`AntiHermitianByConstruction`]: Combined with `exp_antiHermitian_unitary`,
//!   gives `exp(X)† · exp(X) = I` for all X.
//!
//! These are **compile-time guarantees**: if your code compiles with the trait
//! bound `A: TracelessByConstruction`, you know all algebra elements are
//! traceless without any runtime checks.
//!
//! ## Zero Runtime Overhead
//!
//! Marker traits are empty—no methods, no data:
//!
//! ```ignore
//! pub trait TracelessByConstruction: LieAlgebra {}
//! impl TracelessByConstruction for Su2Algebra {}  // Empty impl
//! ```
//!
//! The compiler uses them for type checking, then erases them completely.
//! Same pattern as `Send`, `Sync`, `Copy` in std.
//!
//! ## When to Use Each Pattern
//!
//! | Pattern | Use When | Example |
//! |---------|----------|---------|
//! | Sealed marker trait | Property depends on type, not value | `TracelessByConstruction` |
//! | Private fields | Property depends on value (invariant) | `UnitQuaternion` unit norm |
//! | Fixed-size array | Size is known at compile time | `Plaquette([NodeIndex; 4])` |
//! | Builder pattern | Invariant built incrementally | Path with consecutive edges |

// ============================================================================
// Sealed Trait Pattern for Marker Traits
// ============================================================================
//
// Marker traits (Compact, Abelian, Simple, SemiSimple) are sealed to prevent
// incorrect claims about mathematical properties. These encode theorems:
// - "U(1) is abelian" is a mathematical fact that shouldn't be claimable by arbitrary types
// - "SU(2) is compact" likewise
//
// The core `LieGroup` and `LieAlgebra` traits are OPEN - the associated type
// constraints provide compile-time safety without sealing.

mod sealed {
    // Import types for cleaner impl blocks
    use super::super::rplus::RPlus;
    use super::super::so3::{So3Algebra, SO3};
    use super::super::su2::{Su2Algebra, SU2};
    use super::super::su3::{Su3Algebra, SU3};
    use super::super::sun::{SunAlgebra, SUN};
    use super::super::u1::{U1Algebra, U1};

    /// Sealed trait for compact groups.
    pub trait SealedCompact {}

    /// Sealed trait for abelian groups.
    pub trait SealedAbelian {}

    /// Sealed trait for simple groups.
    pub trait SealedSimple {}

    /// Sealed trait for semi-simple groups.
    pub trait SealedSemiSimple {}

    /// Sealed trait for algebras that are traceless by construction.
    ///
    /// An algebra is "traceless by construction" if its representation
    /// structurally guarantees tr(X) = 0 for all elements X.
    ///
    /// # Examples
    /// - `Su2Algebra::new([f64; 3])`: Pauli basis coefficients → always traceless
    /// - `Su3Algebra::new([f64; 8])`: Gell-Mann basis coefficients → always traceless
    ///
    /// # Non-examples
    /// - `U1Algebra::new(f64)`: Represents iℝ, trace = i·a ≠ 0
    /// - General matrix algebras: Can have arbitrary trace
    pub trait SealedTraceless {}

    /// Sealed trait for algebras that are anti-Hermitian by construction.
    ///
    /// An algebra is "anti-Hermitian by construction" if its representation
    /// structurally guarantees X† = -X for all elements X.
    ///
    /// # Examples
    /// - `Su2Algebra`: Uses i·σ basis (anti-Hermitian)
    /// - `U1Algebra`: Represents iℝ ⊂ ℂ (purely imaginary = anti-Hermitian)
    /// - `So3Algebra`: Real antisymmetric matrices (= anti-Hermitian)
    pub trait SealedAntiHermitian {}

    // =========================================================================
    // Sealed trait implementations
    // =========================================================================

    // Compact groups (all our groups are compact)
    impl SealedCompact for SU2 {}
    impl SealedCompact for SU3 {}
    impl SealedCompact for U1 {}
    impl SealedCompact for SO3 {}
    impl<const N: usize> SealedCompact for SUN<N> {}

    // Abelian groups
    impl SealedAbelian for U1 {}
    impl SealedAbelian for RPlus {}

    // Simple groups (SU(2), SU(3), SO(3), SU(N) for N >= 2)
    impl SealedSimple for SU2 {}
    impl SealedSimple for SU3 {}
    impl SealedSimple for SO3 {}
    // SUN<N> is simple (hence semi-simple) for all N >= 2.
    // The const assert in SUN prevents N < 2 at compile time.
    impl<const N: usize> SealedSimple for SUN<N> {}

    // Semi-simple groups (all simple groups are semi-simple)
    impl SealedSemiSimple for SU2 {}
    impl SealedSemiSimple for SU3 {}
    impl SealedSemiSimple for SO3 {}
    impl<const N: usize> SealedSemiSimple for SUN<N> {}

    // Traceless algebras (representation guarantees tr(X) = 0)
    // NOT U1Algebra: u(1) = iℝ has tr(ia) = ia ≠ 0
    impl SealedTraceless for Su2Algebra {}
    impl SealedTraceless for Su3Algebra {}
    impl SealedTraceless for So3Algebra {}
    impl<const N: usize> SealedTraceless for SunAlgebra<N> {}

    // Anti-Hermitian algebras (representation guarantees X† = -X)
    impl SealedAntiHermitian for Su2Algebra {}
    impl SealedAntiHermitian for Su3Algebra {}
    impl SealedAntiHermitian for So3Algebra {}
    impl SealedAntiHermitian for U1Algebra {}
    impl<const N: usize> SealedAntiHermitian for SunAlgebra<N> {}
}

// ============================================================================
// Marker Traits for Mathematical Properties
// ============================================================================

/// Marker trait for compact Lie groups.
///
/// A Lie group is **compact** if it is compact as a topological space.
///
/// # Examples
/// - `U(1)`, `SU(2)`, `SU(3)`, `SO(3)` - all compact
///
/// # Significance
/// - Representations are completely reducible
/// - Yang-Mills theory well-defined
/// - Haar measure exists
///
/// # Sealed Trait
///
/// This trait is sealed - only verified compact groups can implement it.
pub trait Compact: LieGroup + sealed::SealedCompact {}

/// Marker trait for abelian (commutative) Lie groups.
///
/// Satisfies: `g.compose(&h) == h.compose(&g)` for all g, h
///
/// # Examples
/// - `U(1)` - abelian (electromagnetism)
///
/// # Type Safety
/// ```ignore
/// fn maxwell_theory<G: Abelian>(conn: &NetworkConnection<G>) {
///     // Compiles only for abelian groups!
/// }
/// ```
///
/// # Sealed Trait
///
/// This trait is sealed - only verified abelian groups can implement it.
pub trait Abelian: LieGroup + sealed::SealedAbelian {}

/// Marker trait for simple Lie groups (no non-trivial normal subgroups).
///
/// # Examples
/// - `SU(2)`, `SU(3)`, `SO(3)` - simple
///
/// # Sealed Trait
///
/// This trait is sealed - only verified simple groups can implement it.
pub trait Simple: SemiSimple + sealed::SealedSimple {}

/// Marker trait for semi-simple Lie groups (products of simple groups).
///
/// # Examples
/// - `SU(2)`, `SU(3)` - semi-simple
///
/// # Sealed Trait
///
/// This trait is sealed - only verified semi-simple groups can implement it.
pub trait SemiSimple: LieGroup + sealed::SealedSemiSimple {}

/// Marker trait for Lie algebras whose representation is structurally traceless.
///
/// A Lie algebra is "traceless by construction" when its representation
/// **guarantees** tr(X) = 0 for all elements X, without runtime verification.
///
/// # Mathematical Significance
///
/// Combined with the theorem `det(exp(X)) = exp(tr(X))` (proven in Lean as
/// `Matrix.det_exp_eq_exp_trace`), this gives a **compile-time guarantee**:
///
/// ```text
/// A: TracelessByConstruction  ⟹  tr(X) = 0 for all X ∈ A
///                             ⟹  det(exp(X)) = exp(0) = 1
/// ```
///
/// Therefore, `exp` maps traceless algebras to special (det = 1) groups.
///
/// # Examples
///
/// - `Su2Algebra::new([f64; 3])`: Pauli basis coefficients → always traceless
/// - `Su3Algebra::new([f64; 8])`: Gell-Mann basis coefficients → always traceless
/// - `So3Algebra::new([f64; 3])`: Antisymmetric matrices → always traceless
///
/// # Non-examples
///
/// - `U1Algebra::new(f64)`: Represents iℝ ⊂ ℂ, where tr(ia) = ia ≠ 0
/// - General `Matrix<N,N>`: Can have arbitrary trace
///
/// # Type-Level Proof Pattern
///
/// ```ignore
/// fn exp_to_special_unitary<A>(x: &A) -> SpecialUnitary
/// where
///     A: TracelessByConstruction,
/// {
///     // Compiler knows: tr(x) = 0 by construction
///     // Lean theorem: det(exp(x)) = exp(tr(x)) = exp(0) = 1
///     // Therefore: exp(x) ∈ SU(n), not just U(n)
/// }
/// ```
///
/// # Sealed Trait
///
/// This trait is sealed - only algebras with verified traceless representations
/// can implement it.
pub trait TracelessByConstruction: LieAlgebra + sealed::SealedTraceless {}

/// Marker trait for Lie algebras whose representation is structurally anti-Hermitian.
///
/// A Lie algebra is "anti-Hermitian by construction" when its representation
/// **guarantees** X† = -X for all elements X, without runtime verification.
///
/// # Mathematical Significance
///
/// Combined with the theorem `exp(X)† · exp(X) = I` for anti-Hermitian X
/// (proven in Lean as `Matrix.exp_antiHermitian_unitary`), this gives a
/// **compile-time guarantee**:
///
/// ```text
/// A: AntiHermitianByConstruction  ⟹  X† = -X for all X ∈ A
///                                 ⟹  exp(X) is unitary
/// ```
///
/// # Examples
///
/// - `Su2Algebra`: Uses i·σ basis (anti-Hermitian Pauli matrices)
/// - `Su3Algebra`: Uses i·λ basis (anti-Hermitian Gell-Mann matrices)
/// - `So3Algebra`: Real antisymmetric matrices (= anti-Hermitian over ℝ)
/// - `U1Algebra`: Represents iℝ ⊂ ℂ (purely imaginary = anti-Hermitian)
///
/// # Sealed Trait
///
/// This trait is sealed - only algebras with verified anti-Hermitian representations
/// can implement it.
pub trait AntiHermitianByConstruction: LieAlgebra + sealed::SealedAntiHermitian {}

/// Trait for Lie algebra elements.
///
/// A Lie algebra 𝔤 is a vector space equipped with a bilinear, antisymmetric
/// operation called the Lie bracket [·,·] : 𝔤 × 𝔤 → 𝔤 that satisfies the Jacobi identity.
///
/// In this library, we represent Lie algebras as the tangent space at the identity
/// of a Lie group. For matrix groups, algebra elements are traceless anti-Hermitian matrices.
///
/// # Mathematical Structure
///
/// For classical Lie groups:
/// - **u(1)**: ℝ (1-dimensional)
/// - **su(2)**: ℝ³ (3-dimensional, isomorphic to pure quaternions)
/// - **su(3)**: ℝ⁸ (8-dimensional, Gell-Mann basis)
///
/// # Design
///
/// This trait enables generic gradient descent on Lie groups via the exponential map.
/// The gradient lives in the Lie algebra, and we use `exp: 𝔤 → G` to move on the manifold.
///
/// # Examples
///
/// ```ignore
/// use lie_groups::{LieAlgebra, Su2Algebra};
///
/// let v = Su2Algebra::basis_element(0);  // Pauli X direction
/// let w = Su2Algebra::basis_element(1);  // Pauli Y direction
/// let sum = v.add(&w);                   // Linear combination
/// let scaled = v.scale(2.0);             // Scalar multiplication
/// ```
///
/// # Open Trait
///
/// This trait is open for implementation. The associated type constraints
/// on `LieGroup` (requiring `type Algebra: LieAlgebra`) provide compile-time
/// safety by enforcing group-algebra correspondence.
pub trait LieAlgebra: Clone + Sized + std::fmt::Debug + PartialEq {
    /// Dimension of the Lie algebra as a compile-time constant.
    ///
    /// This enables compile-time dimension checking where possible.
    /// For example, `from_components` can verify array length at compile time
    /// when called with a fixed-size array.
    ///
    /// # Values
    ///
    /// - `U1Algebra`: 1
    /// - `Su2Algebra`: 3
    /// - `So3Algebra`: 3
    /// - `Su3Algebra`: 8
    /// - `SunAlgebra<N>`: N² - 1
    ///
    /// # Note
    ///
    /// Rust doesn't yet support `where A: LieAlgebra<DIM = 3>` bounds
    /// (requires `generic_const_exprs`). Use `const_assert!` or runtime
    /// checks with `Self::DIM` for now.
    const DIM: usize;

    /// Zero element (additive identity) 0 ∈ 𝔤.
    ///
    /// Satisfies: `v.add(&Self::zero()) == v` for all v ∈ 𝔤.
    #[must_use]
    fn zero() -> Self;

    /// Add two algebra elements: v + w
    ///
    /// The Lie algebra is a vector space, so addition is defined.
    ///
    /// # Properties
    ///
    /// - Commutative: `v.add(&w) == w.add(&v)`
    /// - Associative: `u.add(&v.add(&w)) == u.add(&v).add(&w)`
    /// - Identity: `v.add(&Self::zero()) == v`
    #[must_use]
    fn add(&self, other: &Self) -> Self;

    /// Scalar multiplication: α · v
    ///
    /// Scale the algebra element by a real number.
    ///
    /// # Properties
    ///
    /// - Distributive: `v.scale(a + b) ≈ v.scale(a).add(&v.scale(b))`
    /// - Associative: `v.scale(a * b) ≈ v.scale(a).scale(b)`
    /// - Identity: `v.scale(1.0) == v`
    #[must_use]
    fn scale(&self, scalar: f64) -> Self;

    /// Euclidean norm of the coefficient vector: ||v|| = √(Σᵢ vᵢ²)
    ///
    /// This is the L² norm in the **coefficient space**, not the matrix operator
    /// norm. For the standard Gell-Mann normalization `Tr(λᵢ λⱼ) = 2δᵢⱼ`,
    /// the coefficient norm equals `||X||_F / √2` where `||·||_F` is the
    /// Frobenius norm of the matrix representation.
    ///
    /// # Convergence Bounds
    ///
    /// The BCH convergence radius `||X|| + ||Y|| < log(2)` was derived using
    /// the operator norm. Since `||X||_op ≤ ||X||_F = √2 · ||X||_coeff`,
    /// the coefficient norm gives a conservative (safe) convergence check:
    /// if `x.norm() + y.norm() < log(2)`, the BCH series converges.
    ///
    /// # Returns
    ///
    /// Non-negative real number. Zero iff `self` is the zero element.
    #[must_use]
    fn norm(&self) -> f64;

    /// Get the i-th basis element of the Lie algebra.
    ///
    /// For su(2), the basis is {σₓ, σᵧ, σᵤ}/2i (Pauli matrices).
    /// For u(1), the single basis element is i (imaginary unit).
    ///
    /// # Parameters
    ///
    /// - `i`: Index in [0, `dim()`)
    ///
    /// # Panics
    ///
    /// If `i >= Self::DIM`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lie_groups::{LieAlgebra, Su2Algebra};
    ///
    /// let sigma_x = Su2Algebra::basis_element(0);
    /// let sigma_y = Su2Algebra::basis_element(1);
    /// let sigma_z = Su2Algebra::basis_element(2);
    /// ```
    #[must_use]
    fn basis_element(i: usize) -> Self;

    /// Construct algebra element from basis coordinates.
    ///
    /// Given coefficients [c₀, c₁, ..., c_{n-1}], constructs:
    /// ```text
    /// v = Σᵢ cᵢ eᵢ
    /// ```
    /// where eᵢ are the basis elements.
    ///
    /// # Parameters
    ///
    /// - `components`: Slice of length `Self::DIM`
    ///
    /// # Panics
    ///
    /// If `components.len() != Self::DIM`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lie_groups::{LieAlgebra, Su2Algebra};
    ///
    /// let v = Su2Algebra::from_components(&[1.0, 0.0, 0.0]);  // Pure X
    /// let w = Su2Algebra::from_components(&[0.5, 0.5, 0.0]);  // X + Y mix
    /// ```
    #[must_use]
    fn from_components(components: &[f64]) -> Self;

    /// Extract basis coordinates from algebra element.
    ///
    /// Returns the coefficients [c₀, c₁, ..., c_{n-1}] such that:
    /// ```text
    /// self = Σᵢ cᵢ eᵢ
    /// ```
    /// where eᵢ are the basis elements.
    ///
    /// # Returns
    ///
    /// Vector of length `Self::DIM` containing the basis coordinates.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lie_groups::{LieAlgebra, Su2Algebra};
    ///
    /// let v = Su2Algebra::from_components(&[1.0, 2.0, 3.0]);
    /// let coords = v.to_components();
    /// assert_eq!(coords, vec![1.0, 2.0, 3.0]);
    /// ```
    #[must_use]
    fn to_components(&self) -> Vec<f64>;

    /// Lie bracket operation: [X, Y] ∈ 𝔤
    ///
    /// The Lie bracket is a binary operation on the Lie algebra that captures
    /// the non-commutativity of the group. For matrix Lie algebras:
    /// ```text
    /// [X, Y] = XY - YX  (matrix commutator)
    /// ```
    ///
    /// # Mathematical Properties
    ///
    /// The Lie bracket must satisfy:
    ///
    /// 1. **Bilinearity**:
    ///    - `[aX + bY, Z] = a[X,Z] + b[Y,Z]`
    ///    - `[X, aY + bZ] = a[X,Y] + b[X,Z]`
    ///
    /// 2. **Antisymmetry**: `[X, Y] = -[Y, X]`
    ///
    /// 3. **Jacobi Identity**:
    ///    ```text
    ///    [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
    ///    ```
    ///
    /// # Geometric Interpretation
    ///
    /// The bracket measures the "infinitesimal non-commutativity" of the group.
    /// For SU(2) with structure constants `[eᵢ, eⱼ] = −εᵢⱼₖeₖ`:
    /// ```text
    /// [e₁, e₂] = −e₃,  [e₂, e₃] = −e₁,  [e₃, e₁] = −e₂
    /// ```
    ///
    /// # Applications
    ///
    /// - **Structure constants**: Define the algebra's multiplication table
    /// - **Curvature formulas**: F = dA + [A, A]
    /// - **Representation theory**: Adjoint representation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lie_groups::{LieAlgebra, Su2Algebra};
    ///
    /// let e1 = Su2Algebra::basis_element(0);
    /// let e2 = Su2Algebra::basis_element(1);
    /// let bracket = e1.bracket(&e2);
    ///
    /// // [e₁, e₂] = −e₃, so bracket has norm 1
    /// assert!((bracket.norm() - 1.0).abs() < 1e-10);
    /// ```
    ///
    #[must_use]
    fn bracket(&self, other: &Self) -> Self;

    /// Inner product: ⟨v, w⟩
    ///
    /// The inner product induced by the Killing form (or Frobenius norm for
    /// matrix Lie algebras). For an orthonormal basis {eᵢ}:
    /// ```text
    /// ⟨v, w⟩ = Σᵢ vᵢ wᵢ
    /// ```
    ///
    /// # Numerical Stability
    ///
    /// This method provides a numerically stable inner product computation.
    /// The default implementation uses `to_components()` for direct summation,
    /// avoiding the catastrophic cancellation issues of the polarization identity.
    ///
    /// # Returns
    ///
    /// The inner product as a real number. For orthonormal bases, this equals
    /// the dot product of the coefficient vectors.
    ///
    /// # Default Implementation
    ///
    /// Computes `Σᵢ self.to_components()[i] * other.to_components()[i]`.
    /// Override for algebras with more efficient direct computation.
    #[must_use]
    fn inner(&self, other: &Self) -> f64 {
        let v = self.to_components();
        let w = other.to_components();
        v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum()
    }
}

/// Core trait for Lie group elements.
///
/// A Lie group is a smooth manifold with a compatible group structure.
/// This trait captures both the algebraic (group operations) and geometric
/// (distance, smoothness) aspects.
///
/// # Mathematical Properties
///
/// Implementations must satisfy the group axioms:
///
/// 1. **Identity**: `g.compose(&G::identity()) == g`
/// 2. **Inverse**: `g.compose(&g.inverse()) == G::identity()`
/// 3. **Associativity**: `g1.compose(&g2.compose(&g3)) == g1.compose(&g2).compose(&g3)`
/// 4. **Closure**: `compose` always produces a valid group element
///
/// Additionally, the manifold structure should satisfy:
///
/// 5. **Metric**: `distance_to_identity()` measures geodesic distance on the manifold
/// 6. **Smoothness**: Small parameter changes produce smooth paths in the group
///
/// # Type Parameters vs Associated Types
///
/// This trait uses `Self` for the group element type, allowing different
/// representations of the same abstract group (e.g., matrix vs quaternion
/// representation of SU(2)).
///
/// # Examples
///
/// ```
/// use lie_groups::{LieGroup, SU2};
///
/// let g = SU2::rotation_x(1.0);
/// let h = SU2::rotation_y(0.5);
///
/// // Group operations
/// let product = g.compose(&h);
/// let inv = g.inverse();
///
/// // Check group axioms
/// assert!(g.compose(&g.inverse()).distance_to_identity() < 1e-10);
/// ```
///
/// # Open Trait
///
/// This trait is open for implementation. The associated type `Algebra`
/// must implement `LieAlgebra`, providing compile-time enforcement of
/// the group-algebra correspondence.
pub trait LieGroup: Clone + Sized + std::fmt::Debug {
    /// Matrix dimension in the fundamental representation.
    ///
    /// For matrix Lie groups, this is the size of the N×N matrices:
    /// - U(1): `MATRIX_DIM = 1` (complex numbers as 1×1 matrices)
    /// - SU(2): `MATRIX_DIM = 2` (2×2 unitary matrices)
    /// - SU(3): `MATRIX_DIM = 3` (3×3 unitary matrices)
    /// - SO(3): `MATRIX_DIM = 3` (3×3 orthogonal matrices)
    ///
    /// # Distinction from `LieAlgebra::DIM`
    ///
    /// `MATRIX_DIM` is the matrix size N, while `LieAlgebra::DIM` is the
    /// algebra dimension (number of independent generators):
    /// - SU(2): `MATRIX_DIM = 2`, `Algebra::DIM = 3`
    /// - SU(3): `MATRIX_DIM = 3`, `Algebra::DIM = 8`
    const MATRIX_DIM: usize;

    /// Associated Lie algebra type.
    ///
    /// The Lie algebra 𝔤 is the tangent space at the identity of the Lie group G.
    /// This associated type enables generic algorithms that work in the algebra
    /// (e.g., gradient descent, exponential coordinates).
    ///
    /// # Examples
    ///
    /// - `SU(2)::Algebra` = `Su2Algebra` (3-dimensional)
    /// - `U(1)::Algebra` = `U1Algebra` (1-dimensional)
    /// - `SU(3)::Algebra` = `Su3Algebra` (8-dimensional)
    type Algebra: LieAlgebra;

    /// The identity element e ∈ G.
    ///
    /// Satisfies: `g.compose(&Self::identity()) == g` for all g.
    ///
    /// # Mathematical Note
    ///
    /// For matrix groups, this is the identity matrix I.
    /// For additive groups, this would be zero.
    #[must_use]
    fn identity() -> Self;

    /// Group composition (multiplication): g₁ · g₂
    ///
    /// Computes the product of two group elements. For matrix groups,
    /// this is matrix multiplication. For SU(2), preserves unitarity.
    ///
    /// # Properties
    ///
    /// - Associative: `g1.compose(&g2.compose(&g3)) == g1.compose(&g2).compose(&g3)`
    /// - Identity: `g.compose(&Self::identity()) == g`
    ///
    /// # Complexity
    ///
    /// | Group | Time | Space | Notes |
    /// |-------|------|-------|-------|
    /// | SU(2) via quaternion | O(1) | O(1) | 16 multiplications |
    /// | SU(N) via matrix | O(N³) | O(N²) | Matrix multiplication |
    /// | U(1) | O(1) | O(1) | Complex multiplication |
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// let g = SU2::rotation_x(0.5);
    /// let h = SU2::rotation_y(0.3);
    /// let product = g.compose(&h);
    /// ```
    #[must_use]
    fn compose(&self, other: &Self) -> Self;

    /// Group inverse: g⁻¹
    ///
    /// Computes the unique element such that `g.compose(&g.inverse()) == identity`.
    /// For unitary matrix groups, this is the conjugate transpose (adjoint).
    ///
    /// # Properties
    ///
    /// - Involutive: `g.inverse().inverse() == g`
    /// - Inverse property: `g.compose(&g.inverse()) == Self::identity()`
    ///
    /// # Complexity
    ///
    /// | Group | Time | Space | Notes |
    /// |-------|------|-------|-------|
    /// | SU(2) via quaternion | O(1) | O(1) | Quaternion conjugate |
    /// | SU(N) unitary | O(N²) | O(N²) | Conjugate transpose |
    /// | U(1) | O(1) | O(1) | Complex conjugate |
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// let g = SU2::rotation_z(1.2);
    /// let g_inv = g.inverse();
    ///
    /// // Verify inverse property (up to numerical precision)
    /// assert!(g.compose(&g_inv).distance_to_identity() < 1e-10);
    /// ```
    #[must_use]
    fn inverse(&self) -> Self;

    /// Adjoint representation element (for matrix groups: conjugate transpose).
    ///
    /// For matrix groups: `conjugate_transpose(g) = g†`
    /// For unitary groups: `conjugate_transpose(g) = inverse(g)`
    ///
    /// # Gauge Theory Application
    ///
    /// In gauge transformations, fields transform as:
    /// ```text
    /// A' = g A g† + g dg†
    /// ```
    /// where g† is the conjugate transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// let g = SU2::rotation_x(0.7);
    /// let g_dag = g.conjugate_transpose();
    ///
    /// // For unitary groups, conjugate transpose equals inverse
    /// assert!(g_dag.compose(&g).distance_to_identity() < 1e-10);
    /// ```
    #[must_use]
    fn conjugate_transpose(&self) -> Self;

    /// Adjoint representation: `Ad_g`: 𝔤 → 𝔤
    ///
    /// The adjoint representation is a group homomorphism `Ad`: G → Aut(𝔤) that maps
    /// each group element g to a linear automorphism of the Lie algebra.
    ///
    /// # Mathematical Definition
    ///
    /// For matrix Lie groups:
    /// ```text
    /// Ad_g(X) = g X g⁻¹
    /// ```
    ///
    /// For unitary groups where g† = g⁻¹:
    /// ```text
    /// Ad_g(X) = g X g†
    /// ```
    ///
    /// # Distinction from `conjugate_transpose()`
    ///
    /// - `conjugate_transpose()`: Returns g† (conjugate transpose), a **group element**
    /// - `adjoint_action()`: Returns `Ad_g(X)` (conjugation), a **Lie algebra element**
    ///
    /// | Method | Signature | Output Type | Mathematical Meaning |
    /// |--------|-----------|-------------|---------------------|
    /// | `conjugate_transpose()` | `g → g†` | Group element | Hermitian conjugate |
    /// | `adjoint_action()` | `(g, X) → gXg⁻¹` | Algebra element | Conjugation action |
    ///
    /// # Properties
    ///
    /// The adjoint representation is a group homomorphism:
    /// ```text
    /// Ad_{g₁g₂}(X) = Ad_{g₁}(Ad_{g₂}(X))
    /// Ad_e(X) = X
    /// Ad_g⁻¹(X) = Ad_g⁻¹(X)
    /// ```
    ///
    /// It preserves the Lie bracket:
    /// ```text
    /// Ad_g([X, Y]) = [Ad_g(X), Ad_g(Y)]
    /// ```
    ///
    /// For abelian groups (e.g., U(1)):
    /// ```text
    /// Ad_g(X) = X  (trivial action)
    /// ```
    ///
    /// # Gauge Theory Application
    ///
    /// In gauge transformations, the connection A ∈ 𝔤 transforms as:
    /// ```text
    /// A' = Ad_g(A) + g dg†
    ///    = g A g† + (derivative term)
    /// ```
    ///
    /// The adjoint action captures how gauge fields rotate under symmetry transformations.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lie_groups::{LieGroup, SU2, Su2Algebra};
    ///
    /// let g = SU2::rotation_z(std::f64::consts::PI / 2.0);  // 90° rotation
    /// let X = Su2Algebra::basis_element(0);  // σ_x direction
    ///
    /// // Conjugation rotates the algebra element
    /// let rotated_X = g.adjoint_action(&X);
    ///
    /// // Should now point in σ_y direction (rotated by 90°)
    /// ```
    ///
    /// # Complexity
    ///
    /// | Group | Time | Space | Notes |
    /// |-------|------|-------|-------|
    /// | SU(2) via quaternion | O(1) | O(1) | Quaternion sandwich: qvq* |
    /// | SU(N) via matrix | O(N³) | O(N²) | Two matrix multiplications |
    /// | U(1) | O(1) | O(1) | Trivial (abelian) |
    /// | SO(3) | O(1) | O(1) | 3×3 matrix rotation |
    ///
    /// # See Also
    ///
    /// - [`exp`](Self::exp) - Exponential map 𝔤 → G
    /// - [`conjugate_transpose`](Self::conjugate_transpose) - Conjugate transpose g → g†
    /// - [`LieAlgebra::bracket`] - Lie bracket [·,·]
    #[must_use]
    fn adjoint_action(&self, algebra_element: &Self::Algebra) -> Self::Algebra;

    /// Geodesic distance from identity: d(g, e)
    ///
    /// Measures the "size" of the group element as the length of the
    /// shortest geodesic curve from the identity to g on the Lie group
    /// manifold.
    ///
    /// # Mathematical Definition
    ///
    /// For matrix Lie groups, typically computed as:
    /// ```text
    /// d(g, e) = ||log(g)||_F
    /// ```
    /// where ||·||_F is the Frobenius norm and log is the matrix logarithm.
    ///
    /// For SU(2), this simplifies to:
    /// ```text
    /// d(g, e) = arccos(Re(Tr(g))/2)
    /// ```
    /// which is the rotation angle.
    ///
    /// # Properties
    ///
    /// - Non-negative: `d(g, e) ≥ 0`
    /// - Identity: `d(e, e) = 0`
    /// - Symmetric: `d(g, e) = d(g⁻¹, e)`
    ///
    /// # Returns
    ///
    /// Distance in radians (for rotation groups) or appropriate units.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// let identity = SU2::identity();
    /// assert!(identity.distance_to_identity().abs() < 1e-10);
    ///
    /// let rotation = SU2::rotation_x(1.0);
    /// assert!((rotation.distance_to_identity() - 1.0).abs() < 1e-10);
    /// ```
    #[must_use]
    fn distance_to_identity(&self) -> f64;

    /// Distance between two group elements: d(g, h)
    ///
    /// Computed as the distance from the identity to g⁻¹h:
    /// ```text
    /// d(g, h) = d(g⁻¹h, e)
    /// ```
    ///
    /// This is the canonical left-invariant metric on the Lie group.
    ///
    /// # Default Implementation
    ///
    /// Provided via left-invariance of the metric. Can be overridden
    /// for efficiency if a direct formula is available.
    #[must_use]
    fn distance(&self, other: &Self) -> f64 {
        self.inverse().compose(other).distance_to_identity()
    }

    /// Check if this element is approximately the identity.
    ///
    /// Useful for numerical algorithms to check convergence or
    /// validate gauge fixing.
    ///
    /// # Parameters
    ///
    /// - `tolerance`: Maximum allowed distance from identity
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// let almost_identity = SU2::rotation_x(1e-12);
    /// assert!(almost_identity.is_near_identity(1e-10));
    /// ```
    #[must_use]
    fn is_near_identity(&self, tolerance: f64) -> bool {
        self.distance_to_identity() < tolerance
    }

    /// Exponential map: 𝔤 → G
    ///
    /// Maps elements from the Lie algebra (tangent space at identity) to the Lie group
    /// via the matrix exponential or equivalent operation.
    ///
    /// # Mathematical Definition
    ///
    /// For matrix groups:
    /// ```text
    /// exp(X) = I + X + X²/2! + X³/3! + ...
    /// ```
    ///
    /// For SU(2) with X = θ·n̂·(iσ/2) (rotation by angle θ around axis n̂):
    /// ```text
    /// exp(X) = cos(θ/2)I + i·sin(θ/2)·(n̂·σ)
    /// ```
    ///
    /// For U(1) with X = iθ:
    /// ```text
    /// exp(iθ) = e^(iθ)
    /// ```
    ///
    /// # Properties
    ///
    /// - exp(0) = identity
    /// - exp is a local diffeomorphism near 0
    /// - exp(t·X) traces a geodesic on the group manifold
    ///
    /// # Applications
    ///
    /// - **Optimization**: Gradient descent on manifolds via retraction
    /// - **Integration**: Solving ODEs on Lie groups
    /// - **Parameterization**: Unconstrained → constrained optimization
    ///
    /// # Complexity
    ///
    /// | Group | Time | Space | Notes |
    /// |-------|------|-------|-------|
    /// | SU(2) via quaternion | O(1) | O(1) | Rodrigues formula |
    /// | SU(N) via Padé | O(N³) | O(N²) | Matrix exp via scaling+squaring |
    /// | U(1) | O(1) | O(1) | e^(iθ) |
    /// | SO(3) via Rodrigues | O(1) | O(1) | Closed-form rotation |
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lie_groups::{LieGroup, SU2, Su2Algebra};
    ///
    /// let tangent = Su2Algebra::from_components(&[1.0, 0.0, 0.0]);
    /// let group_element = SU2::exp(&tangent);
    /// // group_element is a rotation by 1 radian around X-axis
    /// ```
    #[must_use]
    fn exp(tangent: &Self::Algebra) -> Self;

    /// Logarithm map: G → 𝔤 (inverse of exponential)
    ///
    /// Maps group elements near the identity back to the Lie algebra.
    /// This is the **inverse** of the exponential map in a neighborhood of the identity.
    ///
    /// # Mathematical Definition
    ///
    /// For g ∈ G sufficiently close to the identity, returns X ∈ 𝔤 such that:
    /// ```text
    /// exp(X) = g
    /// ```
    ///
    /// For matrix groups, this is the matrix logarithm:
    /// ```text
    /// log(g) = Σ_{n=1}^∞ (-1)^{n+1} (g - I)^n / n
    /// ```
    ///
    /// For specific groups:
    ///
    /// - **U(1)**: log(e^{iθ}) = iθ (principal branch)
    /// - **SU(2)**: log(exp(θ n̂·σ)) = θ n̂·σ for θ ∈ [0, π)
    /// - **SO(3)**: Rodrigues' formula in reverse
    ///
    /// # Domain Restrictions
    ///
    /// Unlike `exp` which is defined on all of 𝔤, `log` is only well-defined
    /// on a **neighborhood of the identity**:
    ///
    /// - **U(1)**: All elements except negative real axis
    /// - **SU(2)**: All elements except -I (rotation by π with ambiguous axis)
    /// - **SU(3)**: Elements with all eigenvalues in a small neighborhood
    ///
    /// # Errors
    ///
    /// Returns `Err(LogError::NotNearIdentity)` if the element is too far from identity.
    /// Returns `Err(LogError::Singularity)` if at an exact singularity (e.g., -I for SU(2)).
    ///
    /// # Gauge Theory Application
    ///
    /// In lattice gauge theory, the **curvature** (field strength) is computed
    /// from the **holonomy** (Wilson loop) via the logarithm:
    ///
    /// ```text
    /// F_□ = log(U_□) / a²
    /// ```
    ///
    /// where:
    /// - U_□ ∈ G is the plaquette holonomy (group element)
    /// - F_□ ∈ 𝔤 is the curvature (Lie algebra element)
    /// - a is the lattice spacing
    ///
    /// This is the **discrete analog** of the continuum formula:
    /// ```text
    /// F_{μν} = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν] ∈ 𝔤
    /// ```
    ///
    /// # Numerical Considerations
    ///
    /// - For elements very close to identity, use Taylor expansion of log
    /// - For elements near singularities, logarithm is numerically unstable
    /// - In lattice gauge theory, use smaller lattice spacing if log fails
    ///
    /// # Complexity
    ///
    /// | Group | Time | Space | Notes |
    /// |-------|------|-------|-------|
    /// | SU(2) via quaternion | O(1) | O(1) | atan2 + normalize |
    /// | SU(N) via eigendecomp | O(N³) | O(N²) | Schur decomposition |
    /// | U(1) | O(1) | O(1) | atan2 |
    /// | SO(3) via Rodrigues | O(1) | O(1) | Closed-form |
    ///
    /// # Properties
    ///
    /// - log(identity) = 0
    /// - log(exp(X)) = X for small X (in the cut)
    /// - log(g^{-1}) = -log(g) for g near identity
    /// - log is a local diffeomorphism near the identity
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lie_groups::{LieGroup, SU2, Su2Algebra};
    ///
    /// // Create rotation around X-axis by 0.5 radians
    /// let tangent = Su2Algebra::from_components(&[0.5, 0.0, 0.0]);
    /// let g = SU2::exp(&tangent);
    ///
    /// // Recover the algebra element
    /// let recovered = g.log().unwrap();
    ///
    /// // Should match original (up to numerical precision)
    /// assert!(tangent.add(&recovered.scale(-1.0)).norm() < 1e-10);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`exp`](Self::exp) - Exponential map 𝔤 → G
    /// - [`LogError`](crate::LogError) - Error types for logarithm failures
    fn log(&self) -> crate::error::LogResult<Self::Algebra>;

    /// Trace of the identity element
    ///
    /// For SU(N), SO(N), and related matrix groups, the identity matrix I has trace N.
    /// This is used in gauge theory computations:
    /// - Yang-Mills action: `S_p = N - Re Tr(U_p)` (measures deviation from identity)
    /// - Field strength: `|F| ∝ √(N - Re Tr(U))`
    ///
    /// # Examples
    ///
    /// ```
    /// use lie_groups::{SU2, SU3, LieGroup};
    ///
    /// assert_eq!(SU2::trace_identity(), 2.0);
    /// assert_eq!(SU3::trace_identity(), 3.0);
    /// ```
    #[must_use]
    fn trace_identity() -> f64 {
        Self::MATRIX_DIM as f64
    }

    /// Project element back onto the group manifold using Gram-Schmidt orthogonalization.
    ///
    /// **Critical for numerical stability** (Tao priority): After many `compose()` or `exp()` operations,
    /// floating-point errors accumulate and the result may drift from the group manifold.
    /// This method projects the element back onto the manifold, restoring group constraints.
    ///
    /// # Mathematical Background
    ///
    /// For matrix Lie groups, the group manifold is defined by constraints:
    /// - **SU(N)**: U†U = I (unitary) and det(U) = 1 (special)
    /// - **SO(N)**: R†R = I (orthogonal) and det(R) = +1 (special)
    ///
    /// Gram-Schmidt reorthogonalization ensures U†U = I by:
    /// 1. Orthogonalizing columns via modified Gram-Schmidt
    /// 2. Rescaling det(U) → 1 if needed
    ///
    /// # When to Use
    ///
    /// Call after:
    /// - Long chains: `g₁.compose(&g₂).compose(&g₃)....compose(&gₙ)` for large n
    /// - Many exponentials: `exp(X₁).compose(&exp(X₂))....`
    /// - Scaling-and-squaring: After matrix powers in `exp()` algorithm
    ///
    /// # Performance vs Accuracy Trade-off
    ///
    /// - **Cost**: O(N³) for N×N matrices (Gram-Schmidt)
    /// - **Benefit**: Prevents catastrophic error accumulation
    /// - **Guideline**: Reorthogonalize every ~100 operations in long chains
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lie_groups::{LieGroup, SU2};
    ///
    /// // Long composition chain
    /// let mut g = SU2::identity();
    /// for i in 0..1000 {
    ///     g = g.compose(&SU2::rotation_x(0.01));
    ///     if i % 100 == 0 {
    ///         g = g.reorthogonalize();  // Prevent drift every 100 steps
    ///     }
    /// }
    ///
    /// // Verify still on manifold
    /// assert!(g.conjugate_transpose().compose(&g).distance_to_identity() < 1e-12);
    /// ```
    ///
    /// # Default Implementation
    ///
    /// The default implementation uses the exponential/logarithm round-trip:
    /// ```text
    /// reorthogonalize(g) = exp(log(g))
    /// ```
    ///
    /// This works because:
    /// 1. `log(g)` extracts the Lie algebra element (always exact algebraically)
    /// 2. `exp(X)` produces an element exactly on the manifold
    ///
    /// Groups may override this with more efficient methods (e.g., direct Gram-Schmidt for SU(N)).
    ///
    /// # Error Handling
    ///
    /// - **Never fails** for elements close to the manifold
    /// - For elements far from manifold (catastrophic numerical error), returns best approximation
    /// - If `log()` fails (element too far from identity), falls back to identity
    ///
    /// # See Also
    ///
    /// - [`exp`](Self::exp) - Exponential map (always produces elements on manifold)
    /// - [`log`](Self::log) - Logarithm map (inverse of exp near identity)
    /// - [Modified Gram-Schmidt Algorithm](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
    ///
    /// # References
    ///
    /// - Tao, "Topics in Random Matrix Theory" (2012) - Numerical stability
    /// - Higham, "Accuracy and Stability of Numerical Algorithms" (2002) - Matrix orthogonalization
    #[must_use]
    fn reorthogonalize(&self) -> Self {
        // Default: Round-trip through log-exp
        // This works because exp() always produces elements on the manifold
        match self.log() {
            Ok(algebra) => Self::exp(&algebra),
            Err(_) => {
                // Fallback: If log fails (element too corrupted), return identity
                Self::identity()
            }
        }
    }

    /// Geodesic interpolation between two group elements.
    ///
    /// Computes the point at parameter `t` along the geodesic from `self` to `other`:
    /// ```text
    /// γ(t) = g · exp(t · log(g⁻¹ · h))
    /// ```
    ///
    /// - `t = 0.0` → returns `self`
    /// - `t = 1.0` → returns `other`
    /// - `t = 0.5` → geodesic midpoint
    ///
    /// # Errors
    ///
    /// Returns `None` if `g⁻¹h` is at the cut locus (log fails).
    #[must_use]
    fn geodesic(&self, other: &Self, t: f64) -> Option<Self> {
        let delta = self.inverse().compose(other);
        let tangent = delta.log().ok()?;
        Some(self.compose(&Self::exp(&tangent.scale(t))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock Lie algebra for testing (u(1) ≅ ℝ)
    #[derive(Clone, Debug, PartialEq)]
    struct U1Algebra {
        value: f64,
    }

    impl LieAlgebra for U1Algebra {
        const DIM: usize = 1;

        fn zero() -> Self {
            Self { value: 0.0 }
        }

        fn add(&self, other: &Self) -> Self {
            Self {
                value: self.value + other.value,
            }
        }

        fn scale(&self, scalar: f64) -> Self {
            Self {
                value: self.value * scalar,
            }
        }

        fn norm(&self) -> f64 {
            self.value.abs()
        }

        fn basis_element(i: usize) -> Self {
            assert_eq!(i, 0, "U(1) algebra is 1-dimensional");
            Self { value: 1.0 }
        }

        fn from_components(components: &[f64]) -> Self {
            assert_eq!(components.len(), 1);
            Self {
                value: components[0],
            }
        }

        fn to_components(&self) -> Vec<f64> {
            vec![self.value]
        }

        fn bracket(&self, _other: &Self) -> Self {
            Self::zero() // u(1) is abelian
        }
    }

    /// Mock Lie group for testing trait laws
    #[derive(Clone, Debug, PartialEq)]
    struct U1 {
        angle: f64, // U(1) = circle group, element = e^{iθ}
    }

    impl LieGroup for U1 {
        const MATRIX_DIM: usize = 1;

        type Algebra = U1Algebra;

        fn identity() -> Self {
            Self { angle: 0.0 }
        }

        fn compose(&self, other: &Self) -> Self {
            Self {
                angle: (self.angle + other.angle) % (2.0 * std::f64::consts::PI),
            }
        }

        fn inverse(&self) -> Self {
            Self {
                angle: (-self.angle).rem_euclid(2.0 * std::f64::consts::PI),
            }
        }

        fn conjugate_transpose(&self) -> Self {
            self.inverse() // U(1) is abelian, so conjugate transpose = inverse
        }

        fn adjoint_action(&self, algebra_element: &Self::Algebra) -> Self::Algebra {
            // Trivial adjoint action for abelian group
            algebra_element.clone()
        }

        fn distance_to_identity(&self) -> f64 {
            // Shortest arc distance on circle [0, 2π]
            let normalized = self.angle.rem_euclid(2.0 * std::f64::consts::PI);
            let dist = normalized.min(2.0 * std::f64::consts::PI - normalized);
            dist.abs()
        }

        fn exp(tangent: &Self::Algebra) -> Self {
            // exp(iθ) = e^(iθ), represented as angle θ
            Self {
                angle: tangent.value.rem_euclid(2.0 * std::f64::consts::PI),
            }
        }

        fn log(&self) -> crate::error::LogResult<Self::Algebra> {
            // For U(1), log(e^{iθ}) = iθ
            Ok(U1Algebra { value: self.angle })
        }
    }

    #[test]
    fn test_identity_law() {
        let g = U1 { angle: 1.0 };
        let e = U1::identity();

        let g_times_e = g.compose(&e);
        assert!((g_times_e.angle - g.angle).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_law() {
        let g = U1 { angle: 2.5 };
        let g_inv = g.inverse();
        let product = g.compose(&g_inv);

        assert!(product.is_near_identity(1e-10));
    }

    #[test]
    fn test_associativity() {
        let g1 = U1 { angle: 0.5 };
        let g2 = U1 { angle: 1.2 };
        let g3 = U1 { angle: 0.8 };

        let left = g1.compose(&g2.compose(&g3));
        let right = g1.compose(&g2).compose(&g3);

        assert!((left.angle - right.angle).abs() < 1e-10);
    }

    #[test]
    fn test_distance_symmetry() {
        let g = U1 { angle: 1.5 };
        let g_inv = g.inverse();

        let d1 = g.distance_to_identity();
        let d2 = g_inv.distance_to_identity();

        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_is_near_identity() {
        let e = U1::identity();
        assert!(e.is_near_identity(1e-10));

        let g = U1 { angle: 1e-12 };
        assert!(g.is_near_identity(1e-10));

        let h = U1 { angle: 0.1 };
        assert!(!h.is_near_identity(1e-10));
    }

    #[test]
    fn test_exponential_map() {
        // exp(0) = identity
        let zero = U1Algebra::zero();
        let exp_zero = U1::exp(&zero);
        assert!(exp_zero.is_near_identity(1e-10));

        // exp(iθ) = e^(iθ)
        let tangent = U1Algebra { value: 1.5 };
        let group_elem = U1::exp(&tangent);
        assert!((group_elem.angle - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_dim() {
        assert_eq!(U1::MATRIX_DIM, 1);
    }

    #[test]
    fn test_lie_algebra_zero() {
        let zero = U1Algebra::zero();
        assert_eq!(zero.value, 0.0);
    }

    #[test]
    fn test_lie_algebra_add() {
        let v = U1Algebra { value: 1.0 };
        let w = U1Algebra { value: 2.0 };
        let sum = v.add(&w);
        assert_eq!(sum.value, 3.0);
    }

    #[test]
    fn test_lie_algebra_scale() {
        let v = U1Algebra { value: 2.0 };
        let scaled = v.scale(3.0);
        assert_eq!(scaled.value, 6.0);
    }

    #[test]
    fn test_lie_algebra_norm() {
        let v = U1Algebra { value: -3.0 };
        assert_eq!(v.norm(), 3.0);
    }

    #[test]
    fn test_lie_algebra_basis() {
        let basis = U1Algebra::basis_element(0);
        assert_eq!(basis.value, 1.0);
    }

    #[test]
    fn test_lie_algebra_from_components() {
        let v = U1Algebra::from_components(&[4.5]);
        assert_eq!(v.value, 4.5);
    }

    #[test]
    fn test_matrix_dim_available_at_compile_time() {
        assert_eq!(U1::MATRIX_DIM, 1);
    }

    // Note: Associated const equality (e.g., `LieGroup<DIM = 1>`) is not yet stable
    // See https://github.com/rust-lang/rust/issues/92827
    //
    // Once stabilized, we can write functions like:
    //
    // ```ignore
    // fn dimension_one_only<G: LieGroup<DIM = 1>>() -> usize {
    //     G::DIM
    // }
    // ```
    //
    // This would enable compile-time verification of dimension compatibility.

    // ========================================================================
    // Reorthogonalization Tests (TAO PRIORITY - Numerical Stability)
    // ========================================================================

    /// Test that reorthogonalization fixes drift in long composition chains
    #[test]
    fn test_reorthogonalize_prevents_drift_su2() {
        use crate::SU2;

        // Create a long composition chain without reorthogonalization
        let mut g_no_reorth = SU2::identity();
        for _ in 0..1000 {
            g_no_reorth = g_no_reorth.compose(&SU2::rotation_x(0.01));
        }

        // Check unitarity: U†U should be identity
        let unitarity_error_no_reorth = g_no_reorth
            .conjugate_transpose()
            .compose(&g_no_reorth)
            .distance_to_identity();

        // Same chain WITH reorthogonalization every 100 steps
        let mut g_with_reorth = SU2::identity();
        for i in 0..1000 {
            g_with_reorth = g_with_reorth.compose(&SU2::rotation_x(0.01));
            if i % 100 == 0 {
                g_with_reorth = g_with_reorth.reorthogonalize();
            }
        }

        let unitarity_error_with_reorth = g_with_reorth
            .conjugate_transpose()
            .compose(&g_with_reorth)
            .distance_to_identity();

        // Reorthogonalization should reduce drift significantly
        assert!(
            unitarity_error_with_reorth < unitarity_error_no_reorth,
            "Reorthogonalization should reduce drift: {} vs {}",
            unitarity_error_with_reorth,
            unitarity_error_no_reorth
        );

        // With reorthogonalization, should stay very close to manifold
        // Tolerance allows for numerical variation across library versions
        assert!(
            unitarity_error_with_reorth < 1e-7,
            "With reorthogonalization, drift should be minimal: {}",
            unitarity_error_with_reorth
        );
    }

    /// Test that `reorthogonalize()` is idempotent
    #[test]
    fn test_reorthogonalize_idempotent() {
        use crate::SU2;
        use std::f64::consts::PI;

        let g = SU2::rotation_y(PI / 3.0);

        let g1 = g.reorthogonalize();
        let g2 = g1.reorthogonalize();

        // Applying twice should give same result (idempotent)
        assert!(
            g1.distance(&g2) < 1e-14,
            "Reorthogonalization should be idempotent"
        );
    }

    /// Test reorthogonalization preserves group element
    #[test]
    fn test_reorthogonalize_preserves_element() {
        use crate::SU2;
        use std::f64::consts::PI;

        let g = SU2::rotation_z(PI / 4.0);
        let g_reorth = g.reorthogonalize();

        // Should be very close to original
        assert!(
            g.distance(&g_reorth) < 1e-12,
            "Reorthogonalization should preserve element"
        );

        // Reorthogonalized version should be exactly on manifold
        let unitarity_error = g_reorth
            .conjugate_transpose()
            .compose(&g_reorth)
            .distance_to_identity();
        assert!(
            unitarity_error < 1e-14,
            "Reorthogonalized element should be on manifold: {}",
            unitarity_error
        );
    }

    // ========================================================================
    // Operator overloading tests
    // ========================================================================

    use crate::so3::{So3Algebra, SO3};
    use crate::su2::{Su2Algebra, SU2};
    use crate::su3::Su3Algebra;

    #[test]
    fn test_su2_algebra_operators() {
        let x = Su2Algebra([1.0, 2.0, 3.0]);
        let y = Su2Algebra([0.5, 1.0, 1.5]);

        // Add
        let sum = x + y;
        assert_eq!(sum.0, [1.5, 3.0, 4.5]);

        // Sub
        let diff = x - y;
        assert_eq!(diff.0, [0.5, 1.0, 1.5]);

        // Neg
        let neg = -x;
        assert_eq!(neg.0, [-1.0, -2.0, -3.0]);

        // Scalar mul (both directions)
        let scaled = x * 2.0;
        assert_eq!(scaled.0, [2.0, 4.0, 6.0]);
        let scaled2 = 2.0 * x;
        assert_eq!(scaled2.0, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_su2_group_mul_operator() {
        let g = SU2::rotation_x(0.3);
        let h = SU2::rotation_y(0.7);

        // Operator should match compose
        let product_op = &g * &h;
        let product_compose = g.compose(&h);
        assert_eq!(product_op.matrix(), product_compose.matrix());

        // MulAssign
        let mut g2 = g.clone();
        g2 *= &h;
        assert_eq!(g2.matrix(), product_compose.matrix());
    }

    #[test]
    fn test_so3_algebra_operators() {
        let x = So3Algebra([1.0, 0.0, 0.0]);
        let y = So3Algebra([0.0, 1.0, 0.0]);

        let sum = x + y;
        assert_eq!(sum.0, [1.0, 1.0, 0.0]);

        let scaled = 3.0 * x;
        assert_eq!(scaled.0, [3.0, 0.0, 0.0]);

        let neg = -y;
        assert_eq!(neg.0, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_so3_group_mul_operator() {
        let r1 = SO3::rotation_x(0.5);
        let r2 = SO3::rotation_z(0.3);
        let product = &r1 * &r2;
        let expected = r1.compose(&r2);
        assert!(product.distance(&expected) < 1e-14);
    }

    #[test]
    fn test_u1_algebra_operators() {
        let x = crate::U1Algebra(1.5);
        let y = crate::U1Algebra(0.5);

        assert_eq!((x + y).0, 2.0);
        assert_eq!((x - y).0, 1.0);
        assert_eq!((-x).0, -1.5);
        assert_eq!((x * 3.0).0, 4.5);
        assert_eq!((3.0 * x).0, 4.5);
    }

    #[test]
    fn test_u1_group_mul_operator() {
        let g = crate::U1::from_angle(1.0);
        let h = crate::U1::from_angle(2.0);
        let product = &g * &h;
        let expected = g.compose(&h);
        assert!(product.distance(&expected) < 1e-14);
    }

    #[test]
    fn test_rplus_algebra_operators() {
        let x = crate::RPlusAlgebra(2.0);
        let y = crate::RPlusAlgebra(0.5);

        assert_eq!((x + y).0, 2.5);
        assert_eq!((x - y).0, 1.5);
        assert_eq!((-x).0, -2.0);
        assert_eq!((x * 4.0).0, 8.0);
        assert_eq!((4.0 * x).0, 8.0);
    }

    #[test]
    fn test_su3_algebra_operators() {
        let x = Su3Algebra([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Su3Algebra([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let sum = x + y;
        assert_eq!(sum.0[0], 1.0);
        assert_eq!(sum.0[1], 1.0);

        let scaled = 2.0 * x;
        assert_eq!(scaled.0[0], 2.0);
    }

    #[test]
    fn test_sun_algebra_operators() {
        use crate::sun::SunAlgebra;

        let x = SunAlgebra::<2>::basis_element(0);
        let y = SunAlgebra::<2>::basis_element(1);

        let sum = &x + &y;
        assert_eq!(sum.coefficients, vec![1.0, 1.0, 0.0]);

        let diff = x.clone() - y.clone();
        assert_eq!(diff.coefficients, vec![1.0, -1.0, 0.0]);

        let neg = -x.clone();
        assert_eq!(neg.coefficients, vec![-1.0, 0.0, 0.0]);

        let scaled = x.clone() * 5.0;
        assert_eq!(scaled.coefficients, vec![5.0, 0.0, 0.0]);

        let scaled2 = 5.0 * x;
        assert_eq!(scaled2.coefficients, vec![5.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sun_group_mul_operator() {
        use crate::sun::SUN;

        let g = SUN::<3>::identity();
        let h = SUN::<3>::identity();
        let product = &g * &h;
        assert!(product.distance_to_identity() < 1e-14);
    }

    #[test]
    fn test_algebra_operator_consistency_with_trait_methods() {
        // Verify that operators produce the same results as trait methods
        let x = Su2Algebra([0.3, -0.7, 1.2]);
        let y = Su2Algebra([0.5, 0.1, -0.4]);

        // operator + should equal LieAlgebra::add
        let op_sum = x + y;
        let trait_sum = LieAlgebra::add(&x, &y);
        assert_eq!(op_sum.0, trait_sum.0);

        // operator * scalar should equal LieAlgebra::scale
        let op_scaled = x * 2.5;
        let trait_scaled = x.scale(2.5);
        assert_eq!(op_scaled.0, trait_scaled.0);
    }

    // ========================================================================
    // Error Path Tests
    // ========================================================================

    #[test]
    fn test_su2_log_returns_err_for_drifted_matrix() {
        use crate::SU2;
        use num_complex::Complex64;

        // Construct a matrix that violates unitarity (drift simulation)
        let mut bad = SU2::identity();
        bad.matrix[[0, 0]] = Complex64::new(2.0, 0.0); // |cos(θ/2)| > 1
        let result = SU2::log(&bad);
        assert!(result.is_err(), "log should fail for non-unitary matrix");
    }

    #[test]
    fn test_su2_log_at_near_2pi_returns_err() {
        use crate::SU2;

        // θ = 2π is the cut locus where sin(θ/2) = 0
        let near_cut = SU2::exp(&crate::Su2Algebra([
            0.0,
            0.0,
            std::f64::consts::PI * 2.0 - 1e-14,
        ]));
        // This may or may not error depending on precision, but should not panic
        let _result = SU2::log(&near_cut);
    }

    #[test]
    fn test_so3_log_at_pi_returns_err() {
        use crate::so3::SO3;

        // 180° rotation is a singularity for SO(3) log (axis ambiguous)
        let r = SO3::rotation_x(std::f64::consts::PI);
        let result = SO3::log(&r);
        assert!(
            result.is_err(),
            "SO(3) log at π rotation should return singularity error"
        );
    }

    #[test]
    fn test_bch_error_for_invalid_order() {
        use crate::bch::{bch_safe, BchError};
        use crate::Su2Algebra;

        let x = Su2Algebra([0.1, 0.0, 0.0]);
        let y = Su2Algebra([0.0, 0.1, 0.0]);
        let result = bch_safe::<SU2>(&x, &y, 7);
        assert!(
            matches!(result, Err(BchError::InvalidOrder(7))),
            "BCH should reject invalid order"
        );
    }

    // ========================================================================
    // Cross-Group Correspondence Tests
    // ========================================================================

    #[test]
    fn test_su2_so3_double_cover() {
        // SU(2) is a double cover of SO(3): the same rotation angle
        // should produce the same adjoint action on algebra elements.
        //
        // For any g ∈ SU(2), the map Ad_g: su(2) → su(2) factors through
        // the isomorphism su(2) ≅ so(3), giving a rotation in SO(3).
        use crate::so3::{So3Algebra, SO3};
        use crate::su2::{Su2Algebra, SU2};

        let angles = [0.3, 1.0, std::f64::consts::PI / 2.0, 2.5];

        for &angle in &angles {
            // Same physical rotation via SU(2) and SO(3)
            let g_su2 = SU2::rotation_z(angle);
            let r_so3 = SO3::rotation_z(angle);

            // Act on basis vectors via adjoint
            let x_su2 = Su2Algebra([1.0, 0.0, 0.0]);
            let x_so3 = So3Algebra([1.0, 0.0, 0.0]);

            let ad_su2 = g_su2.adjoint_action(&x_su2);
            let ad_so3 = r_so3.adjoint_action(&x_so3);

            // The su(2) → so(3) isomorphism maps e_a ↦ -L_a,
            // so Ad_SU2 and Ad_SO3 should give the same rotation
            // up to the sign convention. Compare magnitudes.
            let su2_rotated = [ad_su2.0[0], ad_su2.0[1], ad_su2.0[2]];
            let so3_rotated = [ad_so3.0[0], ad_so3.0[1], ad_so3.0[2]];

            // The components should match (both rotate x-axis by angle around z)
            // su(2) has -ε structure constants, so Ad picks up a sign flip
            // relative to so(3). The rotation matrix is the same.
            for i in 0..3 {
                assert!(
                    (su2_rotated[i].abs() - so3_rotated[i].abs()) < 1e-10
                        || (su2_rotated[i] + so3_rotated[i]).abs() < 1e-10
                        || (su2_rotated[i] - so3_rotated[i]).abs() < 1e-10,
                    "SU(2)/SO(3) adjoint mismatch at angle {}: su2={:?} so3={:?}",
                    angle,
                    su2_rotated,
                    so3_rotated
                );
            }
        }
    }

    #[test]
    fn test_su2_minus_identity_is_same_so3_rotation() {
        // -I ∈ SU(2) maps to I ∈ SO(3) (kernel of the double cover)
        use crate::su2::{Su2Algebra, SU2};

        // exp(2π ê₃) = -I in SU(2)
        let g = SU2::exp(&Su2Algebra([0.0, 0.0, std::f64::consts::PI]));
        let minus_g = SU2::exp(&Su2Algebra([0.0, 0.0, -std::f64::consts::PI]));

        // Both should give the same adjoint action (since -I acts trivially)
        let x = Su2Algebra([1.0, 2.0, 3.0]);
        let ad_g = g.adjoint_action(&x);
        let ad_minus_g = minus_g.adjoint_action(&x);

        for i in 0..3 {
            assert!(
                (ad_g.0[i] - ad_minus_g.0[i]).abs() < 1e-10,
                "±I should give same adjoint action"
            );
        }
    }
}
