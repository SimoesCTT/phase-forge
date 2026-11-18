use phase_core::PhaseTensor;

/// Stream analyzer for detecting phase desynchronization and anomalies in real-time.
pub struct PhaseStreamAnalyzer {
    pub window_size: usize,
    history: Vec<PhaseTensor>,
}

impl PhaseStreamAnalyzer {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: Vec::with_capacity(window_size),
        }
    }

    /// Push a new phase state into the sliding window
    pub fn push(&mut self, state: PhaseTensor) {
        if self.history.len() >= self.window_size {
            self.history.remove(0);
        }
        self.history.push(state);
    }

    /// Calculate the Phase Locking Value (PLV) across the sliding window.
    /// PLV close to 1.0 indicates strong phase coherence; close to 0.0 indicates turbulence/anomaly.
    pub fn compute_plv(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }

        let num_steps = self.history.len();
        let shape = self.history[0].data.raw_dim();
        let mut sum_cos = 0.0;
        let mut count = 0;

        for indices in ndarray::indices(shape) {
            let mut phase_sum_cos = 0.0;
            let mut phase_sum_sin = 0.0;

            for state in &self.history {
                let theta = state.data[indices].arg();
                phase_sum_cos += theta.cos();
                phase_sum_sin += theta.sin();
            }

            let plv_elem = ((phase_sum_cos / num_steps as f64).powi(2)
                + (phase_sum_sin / num_steps as f64).powi(2))
            .sqrt();

            sum_cos += plv_elem;
            count += 1;
        }

        if count == 0 {
            1.0
        } else {
            sum_cos / count as f64
        }
    }

    /// Check if current stream exhibits an anomaly based on a PLV threshold
    pub fn detect_anomaly(&self, threshold: f64) -> bool {
        self.compute_plv() < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use ndarray::arr2;

    #[test]
    fn test_plv_coherence() {
        let mut analyzer = PhaseStreamAnalyzer::new(5);

        // Feed consistent phase states (high PLV expected)
        for i in 0..5 {
            let z = Complex64::from_polar(1.0, (i as f64) * 0.1);
            analyzer.push(PhaseTensor::new(arr2(&[[z]])));
        }

        let plv = analyzer.compute_plv();
        assert!(plv > 0.9, "Expected high phase coherence, got {}", plv);
        assert!(!analyzer.detect_anomaly(0.5));
    }
}
