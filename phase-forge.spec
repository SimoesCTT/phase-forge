%global debug_package %{nil}

Name:           phase-forge
Version:        0.1.0
Release:        1%{?dist}
Summary:        Phase-domain machine learning and telemetry analytics ecosystem in Rust

License:        MIT
URL:            https://github.com/SimoesCTT/phase-forge
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  rust
BuildRequires:  cargo

%description
Phase-Forge is a pure-Rust, zero-FFI phase-domain machine learning and 
real-time telemetry streaming package. It features complex-valued tensor math, 
Wirtinger autodiff, ModReLU activations, and streaming Phase Locking Value (PLV) 
anomaly detection.

%package -n rust-phase-core-devel
Summary:        Development library for phase-core crate
BuildArch:      noarch

%description -n rust-phase-core-devel
Complex tensor primitives and polar/Cartesian dual-representation for phase-forge.

%package -n rust-phase-layers-devel
Summary:        Development library for phase-layers crate
BuildArch:      noarch

%description -n rust-phase-layers-devel
ModReLU activation functions and complex linear rotation weights for phase-forge.

%prep
%autosetup -c -n %{name}-%{version}

%build
cargo build --workspace --release --offline

%install
mkdir -p %{buildroot}%{_bindir}
install -m 0755 target/release/phase-bridge %{buildroot}%{_bindir}/phase-bridge
install -m 0755 target/release/phase-train %{buildroot}%{_bindir}/phase-train

# Install library sources for development subpackages
mkdir -p %{buildroot}%{_datadir}/cargo/registry/phase-core
cp -rp phase-core/src phase-core/Cargo.toml %{buildroot}%{_datadir}/cargo/registry/phase-core/

mkdir -p %{buildroot}%{_datadir}/cargo/registry/phase-layers
cp -rp phase-layers/src phase-layers/Cargo.toml %{buildroot}%{_datadir}/cargo/registry/phase-layers/

%files
%{_bindir}/phase-bridge
%{_bindir}/phase-train

%files -n rust-phase-core-devel
%{_datadir}/cargo/registry/phase-core

%files -n rust-phase-layers-devel
%{_datadir}/cargo/registry/phase-layers

%changelog
* Thu Nov 19 2026 Americo Nuno Fonseca Simoes <simoesctt@github.com> - 0.1.0-1
- Add rust-prefixed library development subpackages for phase-core and phase-layers
