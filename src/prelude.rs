//! Convenience re-exports for common usage.
//!
//! ```
//! use lie_groups::prelude::*;
//!
//! let g = SU2::rotation_x(0.5);
//! let h = SU2::rotation_y(0.3);
//! let product = &g * &h;
//! ```

pub use crate::bch::{bch_checked, BchError, BchMethod};
pub use crate::error::{LogError, LogResult};
pub use crate::quaternion::UnitQuaternion;
pub use crate::rplus::{RPlus, RPlusAlgebra};
pub use crate::so3::{So3Algebra, SO3};
pub use crate::su2::{Su2Algebra, SU2};
pub use crate::su3::{Su3Algebra, SU3};
pub use crate::sun::{SunAlgebra, SU4, SU5, SUN};
pub use crate::traits::{LieAlgebra, LieGroup};
pub use crate::u1::{U1Algebra, U1};
