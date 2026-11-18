# Phase-Forge ⚡🌀

**Phase-Forge** is a 100% pure-Rust, zero-FFI phase-domain machine learning and real-time telemetry analytics ecosystem. 

Traditional neural networks operate in real-valued vector spaces, treating time-series signals as static magnitudes and discarding phase relationships. Phase-Forge models signals as **complex-valued trajectories on Riemannian manifolds** ($z = r e^{i\theta}$), making it natively suited for harmonic analysis, quantum circuit compilation metrics, audio phase-space processing, and real-time system telemetry anomaly detection.

---

## 🏗️ Workspace Architecture

The workspace is organized into modular, highly decoupled crates:

| Crate | Primary Responsibility | Key Mathematical & Technical Concepts |
| :--- | :--- | :--- |
| **`phase-core`** | Complex tensor primitives | Polar/Cartesian dual-representation, phase unwrapping, magnitude & angle extraction. |
| **`phase-autodiff`** | Complex backpropagation graph | Wirtinger calculus, conjugate gradients ($\bar{\partial}$), and non-analytic descent rules. |
| **`phase-layers`** | Neural network primitives | **ModReLU** activation functions and complex linear rotation weights ($W z + b$). |
| **`phase-infer`** | Real-time streaming inference | Sliding-window **Phase Locking Value (PLV)** computation for instant anomaly detection. |
| **`phase-bridge`** | Telemetry ingestion pipeline | Stdin JSONL parser connecting `chronos-pulse` system metrics into phase trajectories. |
| **`phase-train`** | Training engine | Multi-epoch optimization loops running gradient descent on historical system logs. |

---

## 📐 Mathematical Foundations

### 1. Polar Complex Encoding
Every state vector $z \in \mathbb{C}$ maintains its Cartesian and polar coordinates simultaneously:
$$z = x + iy = r e^{i\theta}$$
Weight multiplication $W z$ rotates incoming phase trajectories rather than merely scaling amplitudes:
$$z_{\text{out}} = (A + iB)(x + iy) = (Ax - By) + i(Ay + Bx)$$

### 2. ModReLU Activation
Standard ReLUs destroy phase angles. ModReLU scales the magnitude while preserving the instantaneous phase angle $\theta$:
$$\sigma_{\text{mod}}(z) = \max\left(0, 1 + \frac{b}{|z|}\right) z$$

### 3. Phase Locking Value (PLV) Loss & Inference
To measure phase synchronization across temporal windows without lagging behind moving averages:
$$\text{PLV} = \left| \frac{1}{N} \sum_{n=1}^{N} e^{i(\theta_{\text{pred}, n} - \theta_{\text{target}, n})} \right|$$

---

## 🚀 Quickstart & Usage

### 1. Run Workspace Tests (Offline Vendored)
Phase-Forge is fully self-contained with vendored dependencies for secure, air-gapped or restricted builds:
```bash
cargo test --workspace --offline

2. Run the Telemetry Inference Bridge
Pipe JSONL telemetry records (e.g., from chronos-pulse) into the bridge to monitor system health and catch scheduler stutters in real time:

cargo build --workspace --offline

printf '{"timestamp":1, "cpu_load":0.45, "scheduler_jitter_ms":0.1}\n{"timestamp":2, "cpu_load":0.48, "scheduler_jitter_ms":0.15}\n{"timestamp":3, "cpu_load":0.95, "scheduler_jitter_ms":8.4}\n' | ./target/debug/phase-bridge

3. Run the Training Loop
Train the complex neural layers on historical telemetry datasets:

cargo run --package phase-train --offline

📦 Fedora Copr Packaging
Phase-Forge includes an RPM spec file (phase-forge.spec) for native distribution on Fedora Linux via Copr (simoesctt/phase-forge).

📄 License
Licensed under the MIT License.
