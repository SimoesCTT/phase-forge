use num_complex::Complex64;
use ndarray::{Array, Ix2};

/// A complex tensor storing state in Cartesian and polar forms for phase manipulation.
#[derive(Clone, Debug)]
pub struct PhaseTensor {
    pub data: Array<Complex64, Ix2>,
}

impl PhaseTensor {
    /// Create a new phase tensor from a 2D complex array
    pub fn new(data: Array<Complex64, Ix2>) -> Self {
        Self { data }
    }

    /// Extract instantaneous phase angles (theta in radians) across the tensor
    pub fn phases(&self) -> Array<f64, Ix2> {
        self.data.mapv(|z| z.arg())
    }

    /// Extract magnitudes (r) across the tensor
    pub fn magnitudes(&self) -> Array<f64, Ix2> {
        self.data.mapv(|z| z.norm())
    }

    /// Apply a global phase rotation by multiplying with e^(i * angle)
    pub fn rotate(&mut self, angle_radians: f64) {
        let rotor = Complex64::cis(angle_radians);
        self.data.mapv_inplace(|z| z * rotor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_phase_and_magnitude() {
        // 1 + i1 -> magnitude = sqrt(2) ≈ 1.41421356, phase = pi / 4
        let data = arr2(&[[Complex64::new(1.0, 1.0)]]);
        let tensor = PhaseTensor::new(data);

        let mags = tensor.magnitudes();
        let phases = tensor.phases();

        assert!((mags[[0, 0]] - 2.0_f64.sqrt()).abs() < 1e-5);
        assert!((phases[[0, 0]] - std::f64::consts::FRAC_PI_4).abs() < 1e-5);
    }
}
