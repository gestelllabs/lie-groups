//! Error types for Lie group and Lie algebra operations
//!
//! Provides principled error handling for Lie-theoretic computations.

use std::fmt;

/// Errors that can occur during Lie group logarithm computation
///
/// The logarithm map `log: G → 𝔤` is only well-defined on a neighborhood
/// of the identity. Elements far from the identity may not have a unique
/// or well-defined logarithm.
///
/// # Mathematical Background
///
/// For matrix Lie groups, the logarithm is the inverse of the exponential map.
/// However, unlike `exp: 𝔤 → G` which is always defined, `log: G → 𝔤` has
/// a restricted domain:
///
/// - **U(1)**: Log defined for all elements except exact multiples of e^{iπ}
/// - **SU(2)**: Log defined for all elements except -I
/// - **SU(3)**: Log defined in a neighborhood of the identity
/// - **SO(3)**: Log defined for rotations with angle < π
///
/// # Physical Interpretation
///
/// In lattice gauge theory, when computing curvature F = log(U_□) from
/// the Wilson loop U_□:
///
/// - If U_□ ≈ I (small plaquette, weak field), log is well-defined
/// - If U_□ far from I (large plaquette, strong field), log may fail
/// - Solution: Use smaller lattice spacing or smearing techniques
#[derive(Debug, Clone, PartialEq)]
pub enum LogError {
    /// Element is too far from identity for logarithm to be computed
    ///
    /// The logarithm map is only guaranteed to exist in a neighborhood
    /// of the identity. This error indicates the element is outside
    /// that neighborhood.
    NotNearIdentity {
        /// Distance from identity
        distance: f64,
        /// Maximum allowed distance for this group
        threshold: f64,
    },

    /// Element is exactly at a singularity of the log map
    ///
    /// For example:
    /// - SU(2): element = -I (rotation by π with ambiguous axis)
    /// - SO(3): rotation by exactly π (axis ambiguous)
    Singularity {
        /// Description of the singularity
        reason: String,
    },

    /// Numerical precision insufficient for accurate logarithm
    ///
    /// When the element is very close to a singularity, numerical
    /// errors can dominate. This indicates the computation would
    /// be unreliable.
    NumericalInstability {
        /// Description of the instability
        reason: String,
    },
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogError::NotNearIdentity {
                distance,
                threshold,
            } => write!(
                f,
                "Element too far from identity for log: distance {:.6} exceeds threshold {:.6}",
                distance, threshold
            ),
            LogError::Singularity { reason } => {
                write!(f, "Logarithm undefined at singularity: {}", reason)
            }
            LogError::NumericalInstability { reason } => {
                write!(f, "Logarithm numerically unstable: {}", reason)
            }
        }
    }
}

impl std::error::Error for LogError {}

/// Result type for Lie group logarithm operations
pub type LogResult<T> = Result<T, LogError>;

/// Errors that can occur during representation theory computations
///
/// Representation theory involves constructing explicit matrices for
/// group actions on vector spaces. Some representations require complex
/// tensor product constructions that may not be implemented.
#[derive(Debug, Clone, PartialEq)]
pub enum RepresentationError {
    /// The requested representation is not yet implemented
    ///
    /// This occurs for higher-dimensional representations that require
    /// tensor product construction or other advanced techniques.
    UnsupportedRepresentation {
        /// Description of the representation (e.g., Dynkin labels)
        representation: String,
        /// Reason why it's not supported
        reason: String,
    },

    /// Invalid representation parameters
    ///
    /// For example, negative Dynkin labels or inconsistent dimensions.
    InvalidParameters {
        /// Description of the invalid parameters
        description: String,
    },

    /// Integration method not implemented for this group
    ///
    /// Generic character integration requires group-specific Haar measure
    /// sampling, which must be implemented separately for each Lie group.
    UnsupportedIntegrationMethod {
        /// The integration method attempted
        method: String,
        /// The group type (e.g., "generic `LieGroup`")
        group: String,
        /// Suggested alternative
        suggestion: String,
    },
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RepresentationError::UnsupportedRepresentation {
                representation,
                reason,
            } => write!(
                f,
                "Representation {} not supported: {}",
                representation, reason
            ),
            RepresentationError::InvalidParameters { description } => {
                write!(f, "Invalid representation parameters: {}", description)
            }
            RepresentationError::UnsupportedIntegrationMethod {
                method,
                group,
                suggestion,
            } => write!(
                f,
                "{} integration not implemented for {}. {}",
                method, group, suggestion
            ),
        }
    }
}

impl std::error::Error for RepresentationError {}

/// Result type for representation theory operations
pub type RepresentationResult<T> = Result<T, RepresentationError>;

// ============================================================================
// Logarithm Conditioning
// ============================================================================

/// Conditioning information for logarithm computations.
///
/// The logarithm map on Lie groups becomes ill-conditioned near the cut locus
/// (e.g., θ → π for SU(2), where the rotation axis becomes ambiguous).
/// This structure provides quantitative information about the reliability
/// of the computed logarithm.
///
/// # Condition Number Interpretation
///
/// The condition number κ measures how sensitive the output is to input
/// perturbations: `|δ(log U)|/|log U| ≤ κ · |δU|/|U|`
///
/// - κ ≈ 1: Well-conditioned, result is reliable
/// - κ ~ 10: Mildly ill-conditioned, expect ~1 digit loss of precision
/// - κ > 100: Severely ill-conditioned, result may be unreliable
/// - κ = ∞: At singularity (cut locus), axis is undefined
///
/// # Example
///
/// ```ignore
/// let (log_u, cond) = g.log_with_condition()?;
/// if cond.is_well_conditioned() {
///     // Safe to use log_u
/// } else if cond.is_usable() {
///     // Use with caution, reduced precision
///     eprintln!("Warning: log condition number = {:.1}", cond.condition_number);
/// } else {
///     // Result unreliable
///     return Err(...);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LogCondition {
    /// Condition number for the logarithm computation.
    ///
    /// For SU(2), this is approximately 1/sin(θ/2) for rotation angle θ,
    /// which diverges as θ → π (approaching the cut locus at -I).
    pub condition_number: f64,

    /// Rotation angle (for SO(3)/SU(2)) or equivalent metric.
    ///
    /// Useful for understanding why conditioning is poor.
    pub angle: f64,

    /// Distance from the cut locus (θ = 2π for SU(2), θ = π for SO(3)).
    ///
    /// Values close to 0 indicate the element is near the cut locus
    /// where the logarithm is ill-defined or ill-conditioned.
    pub distance_to_cut_locus: f64,

    /// Quality assessment of the logarithm computation.
    pub quality: LogQuality,
}

/// Quality levels for logarithm computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogQuality {
    /// Excellent conditioning: κ < 2, full precision expected
    Excellent,
    /// Good conditioning: κ < 10, reliable results
    Good,
    /// Acceptable conditioning: κ < 100, ~2 digits precision loss
    Acceptable,
    /// Poor conditioning: κ < 1000, significant precision loss
    Poor,
    /// At or near singularity, result unreliable
    AtSingularity,
}

impl std::fmt::Display for LogQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogQuality::Excellent => write!(f, "Excellent"),
            LogQuality::Good => write!(f, "Good"),
            LogQuality::Acceptable => write!(f, "Acceptable"),
            LogQuality::Poor => write!(f, "Poor"),
            LogQuality::AtSingularity => write!(f, "AtSingularity"),
        }
    }
}

impl LogCondition {
    /// Create a new `LogCondition` from the rotation angle.
    ///
    /// For SU(2)/SO(3), the condition number is approximately 1/sin(θ/2)
    /// where θ is the rotation angle.
    pub fn from_angle(angle: f64) -> Self {
        let half_angle = angle / 2.0;
        let sin_half = half_angle.sin().abs();
        // For SU(2), the cut locus is at θ = 2π (U = -I); for SO(3) it's at θ = π.
        // We use 2π here as this is primarily used by SU(2)'s log_with_condition.
        let distance_to_cut_locus = (2.0 * std::f64::consts::PI - angle).abs();

        // Condition number: For extracting axis from sin(θ/2) division
        // κ ≈ 1/sin(θ/2), but bounded by practical considerations
        let condition_number = if sin_half > 1e-10 {
            1.0 / sin_half
        } else {
            f64::INFINITY
        };

        let quality = if condition_number < 2.0 {
            LogQuality::Excellent
        } else if condition_number < 10.0 {
            LogQuality::Good
        } else if condition_number < 100.0 {
            LogQuality::Acceptable
        } else if condition_number < 1000.0 {
            LogQuality::Poor
        } else {
            LogQuality::AtSingularity
        };

        Self {
            condition_number,
            angle,
            distance_to_cut_locus,
            quality,
        }
    }

    /// Returns true if the result is well-conditioned (κ < 10).
    #[must_use]
    pub fn is_well_conditioned(&self) -> bool {
        matches!(self.quality, LogQuality::Excellent | LogQuality::Good)
    }

    /// Returns true if the result is usable (κ < 100).
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(
            self.quality,
            LogQuality::Excellent | LogQuality::Good | LogQuality::Acceptable
        )
    }

    /// Returns true if the result is at or near a singularity.
    #[must_use]
    pub fn is_singular(&self) -> bool {
        matches!(self.quality, LogQuality::AtSingularity)
    }
}

/// Result type for conditioned logarithm operations.
pub type ConditionedLogResult<T> = Result<(T, LogCondition), LogError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ========================================================================
    // Condition Number Formula Tests
    // ========================================================================

    /// Verify condition number formula: κ = 1/sin(θ/2)
    ///
    /// The condition number captures the difficulty of extracting the rotation
    /// axis from sin(θ/2). It diverges as θ → 0 (small angle, axis hard to
    /// determine) and is well-conditioned near θ = π.
    #[test]
    fn test_condition_number_formula() {
        // For θ = π/2, κ = 1/sin(π/4) = √2 ≈ 1.414
        let cond_pi_2 = LogCondition::from_angle(PI / 2.0);
        let expected = 1.0 / (PI / 4.0).sin();
        assert!(
            (cond_pi_2.condition_number - expected).abs() < 1e-10,
            "θ=π/2: got {}, expected {}",
            cond_pi_2.condition_number,
            expected
        );
        assert_eq!(cond_pi_2.quality, LogQuality::Excellent);

        // For θ = π/3, κ = 1/sin(π/6) = 2.0 (at boundary, classified as Good)
        let cond_pi_3 = LogCondition::from_angle(PI / 3.0);
        let expected = 1.0 / (PI / 6.0).sin();
        assert!(
            (cond_pi_3.condition_number - expected).abs() < 1e-10,
            "θ=π/3: got {}, expected {}",
            cond_pi_3.condition_number,
            expected
        );
        // κ = 2.0 exactly, which is at the boundary (κ < 2 is Excellent, κ ≥ 2 is Good)
        assert_eq!(cond_pi_3.quality, LogQuality::Good);

        // For θ = 0.1, κ = 1/sin(0.05) ≈ 20.0
        let cond_small = LogCondition::from_angle(0.1);
        let expected = 1.0 / (0.05_f64).sin();
        assert!(
            (cond_small.condition_number - expected).abs() < 1e-8,
            "θ=0.1: got {}, expected {}",
            cond_small.condition_number,
            expected
        );
    }

    /// Verify quality classification thresholds
    #[test]
    fn test_quality_classification() {
        // Excellent: κ < 2 → need sin(θ/2) > 0.5 → θ/2 > π/6 → θ > π/3
        let excellent = LogCondition::from_angle(PI * 0.7); // θ/2 = 0.35π, sin ≈ 0.89
        assert_eq!(excellent.quality, LogQuality::Excellent);
        assert!(excellent.is_well_conditioned());
        assert!(excellent.is_usable());

        // Good: 2 ≤ κ < 10 → 0.1 < sin(θ/2) ≤ 0.5
        let good = LogCondition::from_angle(PI * 0.3); // θ/2 = 0.15π, sin ≈ 0.45
        assert_eq!(good.quality, LogQuality::Good);
        assert!(good.is_well_conditioned());

        // Acceptable: 10 ≤ κ < 100 → 0.01 < sin(θ/2) ≤ 0.1
        let acceptable = LogCondition::from_angle(0.1); // θ/2 = 0.05, sin ≈ 0.05
        assert_eq!(acceptable.quality, LogQuality::Acceptable);
        assert!(!acceptable.is_well_conditioned());
        assert!(acceptable.is_usable());

        // Poor: 100 ≤ κ < 1000 → 0.001 < sin(θ/2) ≤ 0.01
        let poor = LogCondition::from_angle(0.01); // θ/2 = 0.005, sin ≈ 0.005
        assert_eq!(poor.quality, LogQuality::Poor);
        assert!(!poor.is_well_conditioned());
        assert!(!poor.is_usable());

        // AtSingularity: κ ≥ 1000 → sin(θ/2) ≤ 0.001
        let singular = LogCondition::from_angle(0.001); // θ/2 = 0.0005, sin ≈ 0.0005
        assert_eq!(singular.quality, LogQuality::AtSingularity);
        assert!(singular.is_singular());
    }

    /// Verify distance to cut locus computation
    ///
    /// For SU(2), the cut locus is at θ = 2π (U = -I), not θ = π.
    #[test]
    fn test_distance_to_cut_locus() {
        // At θ = π/2, distance to cut locus = 2π - π/2 = 3π/2
        let cond = LogCondition::from_angle(PI / 2.0);
        assert!(
            (cond.distance_to_cut_locus - 3.0 * PI / 2.0).abs() < 1e-10,
            "Distance to cut locus mismatch"
        );

        // At θ = 2π - 0.1, distance = 0.1
        let cond_near = LogCondition::from_angle(2.0 * PI - 0.1);
        assert!(
            (cond_near.distance_to_cut_locus - 0.1).abs() < 1e-10,
            "Near cut locus distance mismatch"
        );

        // At θ ≈ 0, distance ≈ 2π
        let cond_far = LogCondition::from_angle(0.01);
        assert!(
            (cond_far.distance_to_cut_locus - (2.0 * PI - 0.01)).abs() < 1e-10,
            "Far from cut locus distance mismatch"
        );
    }

    /// Verify behavior near θ = 0 (where axis extraction becomes ill-conditioned)
    ///
    /// The condition number κ = 1/sin(θ/2) diverges at both θ → 0 and θ → 2π.
    /// Near θ = 0, the rotation is near-identity; near θ = 2π, U → -I.
    #[test]
    fn test_small_angle_conditioning() {
        // As θ → 0, condition number → ∞
        let small_angles = [0.5, 0.1, 0.01, 0.001, 0.0001];
        let mut prev_kappa = 0.0;

        for &angle in &small_angles {
            let cond = LogCondition::from_angle(angle);

            // Condition number should increase monotonically as angle decreases
            assert!(
                cond.condition_number > prev_kappa,
                "Condition number should increase as θ→0: θ={}, κ={}",
                angle,
                cond.condition_number
            );
            prev_kappa = cond.condition_number;
        }

        // At θ = 1e-11, should be effectively singular
        let cond_singular = LogCondition::from_angle(1e-11);
        assert_eq!(cond_singular.quality, LogQuality::AtSingularity);
    }

    /// Verify behavior near θ = π and θ = 2π
    ///
    /// For SU(2): θ = π is well-conditioned (sin(π/2) = 1, axis extractable).
    /// The true cut locus is at θ = 2π (U = -I, sin(π) = 0).
    #[test]
    fn test_near_cut_locus() {
        // At θ = π, κ = 1/sin(π/2) = 1 (well-conditioned!)
        let cond_pi = LogCondition::from_angle(PI);
        assert!(
            (cond_pi.condition_number - 1.0).abs() < 1e-10,
            "At θ=π, κ should be 1: got {}",
            cond_pi.condition_number
        );
        assert_eq!(cond_pi.quality, LogQuality::Excellent);

        // At θ = π - 0.1, still well-conditioned
        let cond_near_pi = LogCondition::from_angle(PI - 0.1);
        assert!(cond_near_pi.is_well_conditioned());

        // Distance to cut locus at θ = π should be π (since cut locus is at 2π)
        assert!(
            (cond_pi.distance_to_cut_locus - PI).abs() < 1e-10,
            "Distance to cut locus at θ=π should be π: got {}",
            cond_pi.distance_to_cut_locus
        );

        // At θ = 2π, distance to cut locus should be 0
        let cond_2pi = LogCondition::from_angle(2.0 * PI);
        assert!(
            cond_2pi.distance_to_cut_locus < 1e-10,
            "Distance to cut locus at θ=2π should be 0"
        );
    }

    // ========================================================================
    // Precision Loss Estimation Tests
    // ========================================================================

    /// Estimate actual precision loss based on condition number
    ///
    /// If κ is the condition number and ε is machine epsilon (~2.2e-16),
    /// then the expected error bound is roughly κ·ε.
    /// We expect to lose log10(κ) digits of precision.
    #[test]
    fn test_precision_loss_estimate() {
        let machine_eps = f64::EPSILON;

        // For θ = π/2: κ ≈ 1.41, expect ~0 digits lost
        let cond_good = LogCondition::from_angle(PI / 2.0);
        let expected_error = cond_good.condition_number * machine_eps;
        let digits_lost = cond_good.condition_number.log10();
        assert!(digits_lost < 1.0, "Should lose < 1 digit at θ=π/2");
        println!(
            "θ=π/2: κ={:.2}, expected error={:.2e}, digits lost={:.2}",
            cond_good.condition_number, expected_error, digits_lost
        );

        // For θ = 0.1: κ ≈ 20, expect ~1.3 digits lost
        let cond_moderate = LogCondition::from_angle(0.1);
        let digits_lost = cond_moderate.condition_number.log10();
        assert!(
            digits_lost > 1.0 && digits_lost < 2.0,
            "Should lose ~1-2 digits at θ=0.1"
        );
        println!(
            "θ=0.1: κ={:.2}, digits lost={:.2}",
            cond_moderate.condition_number, digits_lost
        );

        // For θ = 0.01: κ ≈ 200, expect ~2.3 digits lost
        let cond_poor = LogCondition::from_angle(0.01);
        let digits_lost = cond_poor.condition_number.log10();
        assert!(
            digits_lost > 2.0 && digits_lost < 3.0,
            "Should lose ~2-3 digits at θ=0.01"
        );
        println!(
            "θ=0.01: κ={:.2}, digits lost={:.2}",
            cond_poor.condition_number, digits_lost
        );
    }

    // ========================================================================
    // LogError Tests
    // ========================================================================

    #[test]
    fn test_log_error_display() {
        let err = LogError::NotNearIdentity {
            distance: 2.5,
            threshold: 1.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("2.5"));
        assert!(msg.contains("1.0"));

        let err2 = LogError::Singularity {
            reason: "rotation by π".to_string(),
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("rotation by π"));
    }

    #[test]
    fn test_representation_error_display() {
        let err = RepresentationError::UnsupportedRepresentation {
            representation: "(3,3)".to_string(),
            reason: "too high dimensional".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("(3,3)"));
        assert!(msg.contains("too high dimensional"));
    }
}
