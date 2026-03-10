# Roadmap

## v0.2.0 (done)

- Breaking API cleanup (naming, field visibility, trait supertraits)
- `geodesic()`, `bch_checked()`, `approx` impls, zero-alloc `inner()`
- CI hardening (audit, deny, semver-checks, dependabot)
- Correctness fixes (reorthogonalization, det check, sin_cos)

## Post-0.2.0

- Property tests (proptest) for `SUN<N>` and `RPlus`
- `serde` feature gate
- Crate-level examples directory
- Consolidate ndarray/nalgebra to single backend
- Symplectic groups Sp(2n)
- Exceptional Lie algebras (G2, F4)
- `no_std` support
- Parallel exp/log for batch operations
