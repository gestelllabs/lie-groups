#![allow(clippy::needless_range_loop)]

//! Lie groups and Lie algebras for computational mathematics.
//!
//! This crate provides concrete implementations of the classical Lie groups
//! used in physics and geometry, with emphasis on correctness, numerical
//! stability, and ergonomic APIs.
//!
//! # Implemented Groups
//!
//! | Group | Algebra | Representation | Dimension |
//! |-------|---------|----------------|-----------|
//! | [`U1`] | [`U1Algebra`] | Phase (complex unit) | 1 |
//! | [`SU2`] | [`Su2Algebra`] | 2×2 complex unitary | 3 |
//! | [`SO3`] | [`So3Algebra`] | 3×3 real orthogonal | 3 |
//! | [`SU3`] | [`Su3Algebra`] | 3×3 complex unitary | 8 |
//! | [`SUN`]`<N>` | [`SunAlgebra`]`<N>` | N×N complex unitary | N²−1 |
//! | [`RPlus`] | [`RPlusAlgebra`] | Positive reals | 1 |
//!
//! # Trait Abstractions
//!
//! The [`LieGroup`] and [`LieAlgebra`] traits provide a uniform interface:
//!
//! ```
//! use lie_groups::{LieGroup, LieAlgebra, SU2, Su2Algebra};
//!
//! let g = SU2::identity();
//! let h = SU2::exp(&Su2Algebra::new([0.1, 0.2, 0.3]));
//! let gh = g.compose(&h);
//! let inv = h.inverse();
//! ```
//!
//! # Features
//!
//! - **Quaternion-optimized SU(2)**: Rotation operations via unit quaternions
//! - **Baker-Campbell-Hausdorff**: Lie algebra composition up to 5th order
//! - **Root systems**: Cartan-Killing classification (A–G families)
//! - **Representation theory**: Casimir operators, characters, Clebsch-Gordan
//! - **Numerical stability**: Conditioned logarithms, scaling-and-squaring exp

pub mod bch;
pub mod error;
pub mod quaternion;
pub mod representation;
pub mod root_systems;
pub mod rplus;
pub mod so3;
pub mod su2;
pub mod su3;
pub mod sun;
pub mod traits;
pub mod u1;

pub use bch::{
    bch_checked, bch_error_bound, bch_fifth_order, bch_fourth_order, bch_is_practical, bch_safe,
    bch_second_order, bch_split, bch_third_order, bch_will_converge, BchError, BchMethod,
};
pub use error::{
    ConditionedLogResult, LogCondition, LogError, LogQuality, LogResult, RepresentationError,
    RepresentationResult,
};
pub use quaternion::UnitQuaternion;
pub use representation::casimir::Casimir;
pub use representation::su3_irrep::Su3Irrep;
pub use representation::{character, character_su2, clebsch_gordan_decomposition, Spin};
pub use root_systems::{Alcove, CartanSubalgebra, Root, RootSystem, WeightLattice, WeylChamber};
pub use rplus::{RPlus, RPlusAlgebra};
pub use so3::{So3Algebra, SO3};
pub use su2::{Su2Algebra, SU2};
pub use su3::{Su3Algebra, SU3};
pub use sun::{SU2Generic, SU3Generic, SunAlgebra, SU4, SU5, SUN};
pub use traits::{
    Abelian, AntiHermitianByConstruction, Compact, LieAlgebra, LieGroup, SemiSimple, Simple,
    TracelessByConstruction,
};
pub use u1::{U1Algebra, U1};
