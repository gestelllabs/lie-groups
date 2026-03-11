# lie-groups

[![Crates.io](https://img.shields.io/crates/v/lie-groups.svg)](https://crates.io/crates/lie-groups)
[![docs.rs](https://docs.rs/lie-groups/badge.svg)](https://docs.rs/lie-groups)
[![License](https://img.shields.io/crates/l/lie-groups.svg)](LICENSE)

Concrete implementations of classical Lie groups and Lie algebras in Rust,
with emphasis on mathematical correctness and numerical stability.

## Why this crate?

Rust has good linear algebra libraries (`nalgebra`, `ndarray`) but no dedicated
support for Lie-theoretic computation. If you work in robotics, physics simulation,
geometric integration, or computational geometry, you need Lie groups — and you need
them to be correct.

`lie-groups` provides:

- **The groups you actually use** — U(1), SU(2), SO(3), SU(3), SU(N), R⁺ — with
  exponential maps, logarithms, and adjoints that satisfy the axioms
- **Operator overloading** — `g * h` for group multiplication, `x + y` and
  `2.0 * x` for algebra arithmetic — mathematical code reads like mathematics
- **Lie algebra operations** — brackets, Baker-Campbell-Hausdorff formula (to 5th
  order), with verified bilinearity, antisymmetry, and Jacobi identity
- **Representation theory** — irreducible representations, Casimir operators,
  characters, Clebsch-Gordan decomposition, root systems, weight lattices
- **Compile-time mathematical markers** — sealed traits for `Compact`, `Simple`,
  `SemiSimple`, `Abelian` prevent incorrect type-level claims about group properties
- **Numerical robustness** — quaternion-optimized SU(2), Higham inverse-scaling
  logarithm for SU(N), conditioned log with quality diagnostics

342 tests (290 unit + 52 doc) verify algebraic axioms — not just API surface —
including exp/log roundtrips, Jacobi identity, bracket bilinearity, and BCH
convergence across all groups.

## Installation

```toml
[dependencies]
lie-groups = "0.2"
```

To disable random sampling (removes `rand`/`rand_distr` dependencies):

```toml
[dependencies]
lie-groups = { version = "0.2", default-features = false }
```

## Quick start

```rust
use lie_groups::{LieGroup, LieAlgebra, SU2, Su2Algebra};

// Exponential map: algebra → group
let xi = Su2Algebra::new([0.1, 0.2, 0.3]);
let g = SU2::exp(&xi);

// Group operations — natural operator syntax
let h = SU2::exp(&Su2Algebra::new([0.4, 0.0, 0.0]));
let gh = &g * &h;        // group multiplication
let g_inv = g.inverse();

// Logarithm: group → algebra
let log_g = SU2::log(&g).unwrap();

// Algebra arithmetic — works like you'd expect
let x = Su2Algebra::new([1.0, 0.0, 0.0]);
let y = Su2Algebra::new([0.0, 1.0, 0.0]);
let sum = x + y;           // vector addition
let scaled = 2.0 * x;      // scalar multiplication
let diff = x - y;           // subtraction
let z = x.bracket(&y);     // [e₁, e₂] = -e₃
```

## Groups

| Group | Algebra | Dim | Representation | Notes |
|-------|---------|-----|----------------|-------|
| U(1)  | u(1)    | 1   | Phase on the unit circle | Abelian |
| SU(2) | su(2)   | 3   | 2×2 unitary, det = 1 | Quaternion-optimized |
| SO(3) | so(3)   | 3   | 3×3 orthogonal, det = 1 | Rotation group |
| SU(3) | su(3)   | 8   | 3×3 unitary, det = 1 | Gell-Mann basis |
| SU(N) | su(N)   | N²−1 | N×N unitary, det = 1 | Const generic N ≥ 2 |
| R⁺    | R       | 1   | Positive reals × | Abelian, non-compact |

## Generic programming with traits

Write algorithms once, run on any Lie group:

```rust
use lie_groups::{LieGroup, LieAlgebra, Compact};

/// Geodesic midpoint on any compact Lie group.
fn geodesic_midpoint<G: LieGroup + Compact>(a: &G, b: &G) -> Option<G> {
    // Built-in: γ(t) = g · exp(t · log(g⁻¹h))
    a.geodesic(b, 0.5)
}
```

## Baker-Campbell-Hausdorff

Compose Lie algebra elements without passing through the group:

```rust
use lie_groups::{bch_second_order, Su2Algebra};

let x = Su2Algebra::new([0.1, 0.0, 0.0]);
let y = Su2Algebra::new([0.0, 0.1, 0.0]);

// BCH: log(exp(X) · exp(Y)) ≈ X + Y + ½[X,Y] + ...
let z = bch_second_order(&x, &y);
```

Higher orders via `bch_third_order` through `bch_fifth_order`. Use `bch_checked`
for runtime convergence validation, or `bch_safe` for automatic fallback to
direct composition when inputs exceed the convergence radius.

## Representation theory

```rust
use lie_groups::{LieGroup, Spin, character_su2, clebsch_gordan_decomposition, SU2, Su2Algebra};

// SU(2) spin-1 character at a group element
let g = SU2::exp(&Su2Algebra::new([0.0, 0.0, std::f64::consts::PI / 3.0]));
let chi = character_su2(Spin::from_integer(1), &g);

// Clebsch-Gordan: spin-½ ⊗ spin-½ = spin-0 ⊕ spin-1
let decomp = clebsch_gordan_decomposition(
    Spin::from_half_integer(1),  // j = 1/2
    Spin::from_half_integer(1),  // j = 1/2
);
```

Root systems, Weyl chambers, weight lattices, and Casimir operators for SU(2) and
SU(3) are also available — see the [API docs](https://docs.rs/lie-groups).

## Conventions

- **su(2) basis**: `{iσ/2}` (Pauli matrices divided by 2), structure constants `fᵢⱼₖ = −εᵢⱼₖ`
- **so(3) basis**: angular momentum generators, structure constants `fᵢⱼₖ = +εᵢⱼₖ`
- **su(3) basis**: Gell-Mann matrices `{iλ/2}`, standard normalization `tr(λₐλᵦ) = 2δₐᵦ`
- **Exponential map**: `exp: 𝔤 → G` maps algebra elements to group elements
- **Logarithm**: `log: G → 𝔤` is the local inverse, returns `Result` with condition info

## License

BSD-3-Clause. See [LICENSE](LICENSE).
