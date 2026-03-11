# Changelog

## 0.2.0

- Renamed `adjoint()` to `conjugate_transpose()`, `DIM` to `MATRIX_DIM`
- Made matrix and algebra fields private with public accessors
- Made `Root.coords` private; use `Root::coords()` accessor instead
- Made `LogCondition` fields private; use accessor methods instead
- Removed dependency re-exports (`Array2`, `Matrix3`, `Complex64`) from public API
- Removed `dim()` and `trace()` from traits (inherent methods remain)
- Added `LieGroup::geodesic()`, `bch_checked()`, `approx` impls for all algebras
- Added `Debug + PartialEq` supertraits on `LieAlgebra`, `Debug` on `LieGroup`
- Fixed SU(N) exp() reorthogonalization, SU2 det check, heap allocations
- CI: cargo-audit, cargo-deny, cargo-semver-checks, dependabot

## 0.1.0

Initial release.
