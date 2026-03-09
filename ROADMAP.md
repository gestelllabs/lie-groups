# Roadmap

## v0.2.0 (in progress)

### Done
- `adjoint()` renamed to `conjugate_transpose()` — eliminates naming confusion
- Matrix fields made `pub(crate)` with `matrix()` accessor — fixes semver hazard
- `LieGroup::DIM` renamed to `MATRIX_DIM` — disambiguates from algebra dimension
- `dim()` and `trace()` removed from traits — use constants and inherent methods
- `to_matrix_array`/`from_matrix_array` renamed to `to_matrix`/`from_matrix` on SU2
- `SO3::to_matrix()`/`from_matrix()` and `trace()` added
- `SUN::trace()` inherent method added
- `approx::AbsDiffEq` and `RelativeEq` for all algebra types
- CI: cargo-audit, cargo-deny, cargo-semver-checks, dependabot, SHA-pinned actions
- Error path tests and cross-group correspondence tests (SU2/SO3 double cover)

### Remaining
- Property tests (proptest) for `SUN<N>` and `RPlus`
- `serde` feature gate for `Serialize`/`Deserialize` on all public types
- Crate-level examples directory with runnable programs

## Future (post-0.2.0)
- Consolidate `ndarray` and `nalgebra` to a single matrix backend
- Symplectic groups Sp(2n)
- Exceptional Lie algebras (G2, F4)
- `no_std` support (pending matrix backend feasibility)
- Parallel exp/log for batch operations
