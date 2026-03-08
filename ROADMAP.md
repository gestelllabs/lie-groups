# Roadmap

## v0.2.0

### Testing
- Property tests (proptest) for `SUN<N>` — the generic engine lacks axiom verification
- Property tests for `RPlus`
- Mutation testing baseline via `cargo-mutants`
- Numerical boundary fuzzing at scaling-and-squaring thresholds in `exp`/`log`

### API
- `serde` feature gate for `Serialize`/`Deserialize` on all public types
- `approx::AbsDiffEq` and `RelativeEq` impls on algebra and group types
- `MANIFOLD_DIM` associated constant on `LieGroup` (distinct from matrix `DIM`)
- `Display` impls for group types (`SU2`, `SU3`, `SO3`, `SUN`)

### Dependencies
- Evaluate consolidating `ndarray` and `nalgebra` to a single matrix backend

### Documentation
- Crate-level examples directory with runnable programs
- Performance characteristics and algorithm notes in module docs

## Future (post-0.2.0)
- Symplectic groups Sp(2n)
- Exceptional Lie algebras (G2, F4)
- `no_std` support (pending matrix backend feasibility)
- Parallel exp/log for batch operations
