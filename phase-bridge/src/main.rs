use num_complex::Complex64;
use ndarray::arr2;
use phase_core::PhaseTensor;
use phase_infer::PhaseStreamAnalyzer;
use serde::Deserialize;
use std::io::{self, BufRead};

/// Telemetry record structure matching chronos-pulse JSONL output
#[derive(Deserialize, Debug)]
pub struct TelemetryRecord {
    pub timestamp: u64,
    pub cpu_load: f64,
    pub scheduler_jitter_ms: f64,
}

fn main() {
    println!("=== Phase-Forge <-> Chronos-Pulse Real-Time Inference Bridge ===");
    println!("Listening for JSONL telemetry on stdin...");

    // Initialize streaming analyzer with a sliding window of 10 ticks
    let mut analyzer = PhaseStreamAnalyzer::new(10);
    let stdin = io::stdin();
    let locked_stdin = stdin.lock();

    for line_result in locked_stdin.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };

        // Parse incoming chronos-pulse JSONL record
        let record: TelemetryRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue, // Skip malformed lines
        };

        // Map linear telemetry (load & jitter) into complex phase space:
        // Magnitude = CPU load intensity, Phase angle = Scheduler jitter radians
        let magnitude = record.cpu_load.max(0.01);
        let phase_angle = record.scheduler_jitter_ms * 0.1; // Scale jitter into phase radians
        let z = Complex64::from_polar(magnitude, phase_angle);

        let tensor = PhaseTensor::new(arr2(&[[z]]));
        analyzer.push(tensor);

        // Compute real-time Phase Locking Value (PLV)
        let plv = analyzer.compute_plv();
        let is_anomaly = analyzer.detect_anomaly(0.6);

        if is_anomaly {
            println!(
                "[ANOMALY DETECTED] Timestamp: {} | PLV: {:.4} | Load: {:.2} | Jitter: {:.2}ms",
                record.timestamp, plv, record.cpu_load, record.scheduler_jitter_ms
            );
        } else {
            println!(
                "[HEALTHY] Timestamp: {} | PLV: {:.4} | Jitter: {:.2}ms",
                record.timestamp, plv, record.scheduler_jitter_ms
            );
        }
    }
}
