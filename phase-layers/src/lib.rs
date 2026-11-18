use num_complex::Complex64;
use ndarray::{Array2, Array1};
use phase_core::PhaseTensor;

/// ModReLU activation function: scales magnitude by max(0, |z| + b) while preserving phase angle.
pub fn mod_relu(z: Complex64, b: f64) -> Complex64 {
    let norm = z.norm();
    if norm == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    let scale = (norm + b).max(0.0) / norm;
    z * scale
}

/// A complex-valued dense layer with weights and bias.
#[derive(Clone, Debug)]
pub struct PhaseDense {
    pub weights: Array2<Complex64>,
    pub bias: Array1<Complex64>,
}

impl PhaseDense {
    /// Initialize a new complex dense layer with given dimensions
    pub fn new(weights: Array2<Complex64>, bias: Array1<Complex64>) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: Z_out = W * Z_in + b
    pub fn forward(&self, input: &PhaseTensor) -> PhaseTensor {
        // Matrix multiplication in complex space using ndarray dot
        let out_data = input.data.dot(&self.weights.t()) + &self.bias;
        PhaseTensor::new(out_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_mod_relu() {
        // Positive magnitude test with bias b = -0.5
        // z = 3 + 4i -> norm = 5. (5 - 0.5) / 5 = 0.9. Result = 2.7 + 3.6i
        let z = Complex64::new(3.0, 4.0);
        let res = mod_relu(z, -0.5);
        assert!((res.re - 2.7).abs() < 1e-5);
        assert!((res.im - 3.6).abs() < 1e-5);

        // Negative threshold test: norm < -b -> clamped to 0
        let z_small = Complex64::new(0.3, 0.4); // norm = 0.5
        let res_zero = mod_relu(z_small, -1.0);
        assert_eq!(res_zero, Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_phase_dense_forward() {
        let weights = arr2(&[[Complex64::new(1.0, 0.0)]]);
        let bias = arr1(&[Complex64::new(0.0, 1.0)]);
        let layer = PhaseDense::new(weights, bias);

        let input_tensor = PhaseTensor::new(arr2(&[[Complex64::new(2.0, 2.0)]]));
        let output = layer.forward(&input_tensor);

        // (2 + 2i) * 1 + (0 + 1i) = 2 + 3i
        assert!((output.data[[0, 0]].re - 2.0).abs() < 1e-5);
        assert!((output.data[[0, 0]].im - 3.0).abs() < 1e-5);
    }
}
