use num_complex::Complex64;
use ndarray::{arr2, arr1, Array2, Array1};
use phase_core::PhaseTensor;
use phase_autodiff::Var;
use phase_layers::PhaseDense;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct TelemetryRecord {
    pub timestamp: u64,
    pub cpu_load: f64,
    pub scheduler_jitter_ms: f64,
}

fn main() {
    println!("=== Phase-Forge Training Loop ===");
    println!("Initializing complex-valued neural network for phase telemetry...");

    // 1. Initialize complex weights and biases deterministically
    let init_weights: Array2<Complex64> = arr2(&[
        [Complex64::new(0.8, 0.1), Complex64::new(-0.2, 0.4)],
        [Complex64::new(0.3, -0.5), Complex64::new(0.7, 0.2)],
    ]);
    let init_bias: Array1<Complex64> = arr1(&[
        Complex64::new(0.1, 0.0),
        Complex64::new(0.0, 0.1),
    ]);

    let mut dense_layer = PhaseDense::new(init_weights, init_bias);
    let mut weight_var = Var::new(PhaseTensor::new(dense_layer.weights.clone()));

    // 2. Mock historical telemetry dataset (representing normal system states)
    let dataset = vec![
        TelemetryRecord { timestamp: 1, cpu_load: 0.40, scheduler_jitter_ms: 0.1 },
        TelemetryRecord { timestamp: 2, cpu_load: 0.45, scheduler_jitter_ms: 0.12 },
        TelemetryRecord { timestamp: 3, cpu_load: 0.50, scheduler_jitter_ms: 0.15 },
        TelemetryRecord { timestamp: 4, cpu_load: 0.42, scheduler_jitter_ms: 0.11 },
    ];

    let lr = 0.01;
    let epochs = 5;

    // 3. Multi-epoch training loop
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        for record in &dataset {
            // Map telemetry record to complex input vector
            let magnitude = record.cpu_load;
            let phase = record.scheduler_jitter_ms * 0.1;
            let input_z = Complex64::from_polar(magnitude, phase);
            
            // Expand into 2D tensor batch shape [1, 2]
            let input_tensor = PhaseTensor::new(arr2(&[[input_z, input_z]]));

            // Forward pass
            let output = dense_layer.forward(&input_tensor);

            // Compute target (steady state expectation)
            let target = Complex64::new(0.5, 0.0);

            // Calculate complex Mean Squared Error loss for this sample
            let diff = output.data[[0, 0]] - target;
            let loss = diff.norm_sqr();
            total_loss += loss;

            // Approximate Wirtinger gradient for weight update
            let grad_val = diff * 2.0;
            weight_var.grad = arr2(&[
                [grad_val, grad_val],
                [grad_val, grad_val],
            ]);

            // Apply Wirtinger gradient descent step: z_new = z - lr * conj(grad)
            weight_var.step(lr);
            dense_layer.weights = weight_var.node.data.clone();
        }

        println!("Epoch {}/{}: Training Loss = {:.6}", epoch, epochs, total_loss / dataset.len() as f64);
    }

    println!("Training complete! Complex neural weights successfully optimized on historical telemetry.");
}
