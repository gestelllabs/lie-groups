# Roadmap

## v0.2.0 (done)

- Breaking API cleanup (naming, field visibility, trait supertraits)
- Unified basis normalization: tr(Tₐ†Tᵦ) = ½δₐᵦ across SU2, SU3, SUN
- `geodesic()`, `bch_checked()`, `approx` impls, zero-alloc `inner()`
- CI hardening (audit, deny, semver-checks, dependabot)
- Correctness fixes (reorthogonalization, det check, sin_cos)

## v0.3.0 — Feature parity across groups

Stabilize the API by closing gaps in the feature matrix. Every compact group
should support the same core operations.

### Numerical stability
- `log_with_condition()` for SO3, SU3, SUN (currently SU2-only)
- Public `renormalize()` for SU3, SUN (private helpers already exist)

### Random sampling
- `random_haar()` for SO3, SU3, SUN (currently SU2-only)
- `random_small()` for SU2, SO3, SU3, SUN (currently U1-only)

### Representation theory
- Casimir trait for So3Algebra (same eigenvalues as SU2 via algebra isomorphism)
- Casimir trait for SunAlgebra<N> (requires generic representation type)

### API consistency
- Reconcile SO3-only `interpolate()` with trait-level `geodesic()` (deprecate or document)
- Unify `verify_unitarity()`/`verify_orthogonality()` naming (trait method or consistent name)
- Add group-level `from_matrix()`/`to_matrix()` for SU3, SUN (currently algebra-only)
- Property tests (proptest) for SUN<N> and RPlus

## Post-0.3.0

- `serde` feature gate
- Crate-level examples directory
- Consolidate ndarray/nalgebra to single backend
- Symplectic groups Sp(2n)
- Exceptional Lie algebras (G2, F4)
- `no_std` support
- Parallel exp/log for batch operations
