# Changelog

## 0.2.0

### Breaking Changes

- Renamed `adjoint()` → `conjugate_transpose()`, `DIM` → `MATRIX_DIM`
- Made matrix and algebra fields `pub(crate)` with public accessors
- Made `Root.coords` private; use `Root::coords()` accessor instead
- Made `LogCondition` fields private; use accessor methods instead
- Removed `dim()` and `trace()` from traits (inherent methods remain)
- Removed dependency re-exports (`Array2`, `Matrix3`, `Complex64`) from public API;
  import from `nalgebra`, `ndarray`, `num_complex` directly
- Added `Debug + PartialEq` supertraits on `LieAlgebra`, `Debug` on `LieGroup`

### Added

- `LieGroup::geodesic()` for geodesic interpolation on compact groups
- `bch_checked()` with runtime convergence validation
- `approx::AbsDiffEq` / `RelativeEq` impls for all algebra types

### Fixed

- SU(N) `exp()` reorthogonalization for improved unitarity
- SU(2) determinant check reliability
- Reduced heap allocations in algebra arithmetic
- Documentation: inner product is coefficient dot product, not Killing form
- Documentation: root systems limited to type Aₙ (not A–G)

### Infrastructure

- CI: cargo-audit, cargo-deny, cargo-semver-checks, dependabot

## 0.1.0

Initial release.
