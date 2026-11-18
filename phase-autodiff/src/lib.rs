use num_complex::Complex64;
use ndarray::{Array, Ix2};
use phase_core::PhaseTensor;

/// A differentiable variable wrapping a PhaseTensor and its gradient.
#[derive(Clone, Debug)]
pub struct Var {
    pub node: PhaseTensor,
    pub grad: Array<Complex64, Ix2>,
}

impl Var {
    /// Create a new variable with zeroed gradients matching the tensor shape
    pub fn new(node: PhaseTensor) -> Self {
        let shape = node.data.raw_dim();
        let grad = Array::zeros(shape);
        Self { node, grad }
    }

    /// Perform a Wirtinger gradient descent step on the variable
    /// z_new = z - lr * conj(grad)
    pub fn step(&mut self, lr: f64) {
        self.node.data.zip_mut_with(&self.grad, |z, g| {
            // Wirtinger gradient descent update rule uses the complex conjugate of the gradient
            *z -= lr * g.conj();
        });
    }

    /// Zero out accumulated gradients
    pub fn zero_grad(&mut self) {
        self.grad.fill(Complex64::new(0.0, 0.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_wirtinger_update() {
        let data = arr2(&[[Complex64::new(1.0, 1.0)]]);
        let tensor = PhaseTensor::new(data);
        let mut var = Var::new(tensor);

        // Simulate a gradient of 1 + i1
        var.grad = arr2(&[[Complex64::new(1.0, 1.0)]]);

        // Step with learning rate 0.1
        // conj(1 + i1) = 1 - i1
        // z_new = (1 + i1) - 0.1 * (1 - i1) = 0.9 + i1.1
        var.step(0.1);

        assert!((var.node.data[[0, 0]].re - 0.9).abs() < 1e-5);
        assert!((var.node.data[[0, 0]].im - 1.1).abs() < 1e-5);
    }
}
