# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-03-11

### Breaking Changes

**Renamed:**
- `LieGroup::adjoint()` → `conjugate_transpose()` (the old name was
  mathematically misleading — the adjoint representation is a different object)
- `LieGroup::DIM` → `MATRIX_DIM` (distinguishes matrix size N from algebra
  dimension N²−1)

**Encapsulated:**
- `SU2.matrix`, `SU3.matrix`, `SO3.matrix`, `SUN.matrix` — now `pub(crate)`;
  use `.matrix()` accessor
- `Su2Algebra`, `Su3Algebra`, `So3Algebra`, `SunAlgebra` tuple fields — now
  `pub(crate)`; use `.components()` / `.coefficients()` accessors
- `Root.coords` — now private; use `Root::coords()` → `&[f64]`
- `LogCondition.condition_number`, `.angle`, `.distance_to_cut_locus`,
  `.quality` — now private; use accessor methods (same names, with `()`)

**Removed from public API:**
- `LieAlgebra::dim()` — use `Self::DIM` const instead
- `LieGroup::trace()` — inherent methods remain on SU2, SU3
- Re-exports of `nalgebra::Matrix3`, `ndarray::Array2`, `num_complex::Complex64`
  — import these from their own crates

**New trait bounds:**
- `LieAlgebra` now requires `Debug + PartialEq`
- `LieGroup` now requires `Debug`

These bounds break downstream `impl LieAlgebra for MyType` if `MyType` doesn't
derive `Debug`/`PartialEq`. Add `#[derive(Debug, PartialEq)]`.

### Added

- `LieGroup::geodesic(other, t)` — geodesic interpolation γ(t) = g · exp(t · log(g⁻¹h))
- `bch_checked()` — BCH with runtime convergence validation
- `approx::AbsDiffEq` and `RelativeEq` impls for all algebra types

### Fixed

- SU(N) `exp()` now reorthogonalizes the result, improving unitarity for
  large algebra elements
- SU(2) determinant check no longer gives false negatives near θ = 2π
- Reduced heap allocations in algebra `add`/`scale`/`bracket` operations

### Documentation

- `LieAlgebra::inner()` — clarified this is the coefficient dot product
  Σᵢ vᵢwᵢ, not the Killing form (they are proportional but not equal)
- `LieAlgebra::norm()` — stated the basis normalization assumption for BCH
  convergence bounds
- Root systems — scoped to type Aₙ only (was incorrectly described as A–G)
- Compact marker trait — noted R⁺ exclusion

### Infrastructure

- CI: cargo-audit, cargo-deny, cargo-semver-checks
- Dependabot for cargo and github-actions dependencies
- `deny.toml` with license allowlist

## [0.1.0] — 2026-03-08

Initial release.

- Groups: U(1), SU(2), SO(3), SU(3), SU(N), R⁺
- Traits: `LieGroup`, `LieAlgebra`, sealed markers (`Compact`, `Simple`,
  `SemiSimple`, `Abelian`, `TracelessByConstruction`, `AntiHermitianByConstruction`)
- Exponential map and logarithm for all groups
- Lie bracket with verified Jacobi identity
- Baker-Campbell-Hausdorff formula to 5th order
- Representation theory: characters, Clebsch-Gordan (SU(2)), Casimir operators
- Root systems (type Aₙ), Weyl chambers, weight lattices
- Quaternion-optimized SU(2), Higham log for SU(N)
- 340 tests (288 unit + 52 doc)
