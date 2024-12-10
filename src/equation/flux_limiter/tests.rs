#[cfg(test)]
mod tests {
    use crate::equation::flux_limiter::flux_limiters::{
        BeamWarming, FluxLimiter, Koren, Minmod, Superbee, VanAlbada, VanLeer,
    };

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_minmod_limiter() {
        let minmod = Minmod;

        // Test with same signs
        assert!(approx_eq(minmod.limit(1.0, 0.5), 0.5, 1e-6));
        assert!(approx_eq(minmod.limit(0.5, 1.0), 0.5, 1e-6));

        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(minmod.limit(1.0, -0.5), 0.0, 1e-6));
        assert!(approx_eq(minmod.limit(-1.0, 0.5), 0.0, 1e-6));

        // Test with zero values
        assert!(approx_eq(minmod.limit(0.0, 1.0), 0.0, 1e-6));
        assert!(approx_eq(minmod.limit(1.0, 0.0), 0.0, 1e-6));

        // Test edge cases with very high and low values
        assert!(approx_eq(minmod.limit(1e6, 1e6), 1e6, 1e-6));
        assert!(approx_eq(minmod.limit(-1e6, -1e6), -1e6, 1e-6));
    }

    #[test]
    fn test_superbee_limiter() {
        let superbee = Superbee;

        // Test with same signs
        assert!(approx_eq(superbee.limit(1.0, 0.5), 1.0, 1e-6));
        assert!(approx_eq(superbee.limit(0.5, 1.0), 1.0, 1e-6));

        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(superbee.limit(1.0, -0.5), 0.0, 1e-6));
        assert!(approx_eq(superbee.limit(-1.0, 0.5), 0.0, 1e-6));

        // Test with zero values
        assert!(approx_eq(superbee.limit(0.0, 1.0), 0.0, 1e-6));
        assert!(approx_eq(superbee.limit(1.0, 0.0), 0.0, 1e-6));

        // Test edge cases with very high and low values
        assert!(approx_eq(superbee.limit(1e6, 1e6), 1e6, 1e-6));
        assert!(approx_eq(superbee.limit(-1e6, -1e6), -1e6, 1e-6));
    }

    #[test]
    fn test_vanleer_limiter() {
        let vanleer = VanLeer;
    
        // Test with same signs
        assert!(approx_eq(vanleer.limit(1.0, 2.0), 0.6666666666666666, 1e-6)); // Corrected precision
        assert!(approx_eq(vanleer.limit(-1.0, -2.0), 0.6666666666666666, 1e-6));
    
        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(vanleer.limit(1.0, -2.0), 0.0, 1e-6));
    
        // Test with zero values
        assert!(approx_eq(vanleer.limit(0.0, 2.0), 0.0, 1e-6));
        assert!(approx_eq(vanleer.limit(2.0, 0.0), 0.0, 1e-6));
    }
    
    #[test]
    fn test_vanalbada_limiter() {
        let vanalbada = VanAlbada;
    
        // Test with same signs
        assert!(approx_eq(vanalbada.limit(1.0, 2.0), 1.2, 1e-6)); // Corrected
        assert!(approx_eq(vanalbada.limit(-1.0, -2.0), -1.2, 1e-6));
    
        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(vanalbada.limit(1.0, -2.0), 0.0, 1e-6));
    
        // Test with zero values
        assert!(approx_eq(vanalbada.limit(0.0, 2.0), 0.0, 1e-6));
        assert!(approx_eq(vanalbada.limit(2.0, 0.0), 0.0, 1e-6));
    }
    
    #[test]
    fn test_koren_limiter() {
        let koren = Koren;
    
        // Test with same signs
        assert!(approx_eq(koren.limit(1.0, 2.0), 0.6666666666666666, 1e-6)); // Corrected
        assert!(approx_eq(koren.limit(-1.0, -2.0), 0.6666666666666666, 1e-6)); // Corrected
    
        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(koren.limit(1.0, -2.0), 0.0, 1e-6));
    
        // Test with zero values
        assert!(approx_eq(koren.limit(0.0, 2.0), 0.0, 1e-6));
        assert!(approx_eq(koren.limit(2.0, 0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_beamwarming_limiter() {
        let beamwarming = BeamWarming;

        // Test with same signs
        assert!(approx_eq(beamwarming.limit(1.0, 2.0), 1.0, 1e-6));
        assert!(approx_eq(beamwarming.limit(-1.0, -2.0), -1.0, 1e-6));

        // Test with opposite signs
        assert!(approx_eq(beamwarming.limit(1.0, -2.0), 1.0, 1e-6));

        // Test with zero values
        assert!(approx_eq(beamwarming.limit(0.0, 2.0), 0.0, 1e-6));
        assert!(approx_eq(beamwarming.limit(2.0, 0.0), 2.0, 1e-6));
    }
}
