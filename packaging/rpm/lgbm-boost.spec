%global debug_package %{nil}
%global _build_id_links none
%global __os_install_post %{nil}
%global __requires_exclude ^python.*dist\\(

Name:           %{py_pkg_prefix}-lgbm-boost
Version:        %{pkg_version}
Release:        1%{?dist}
Summary:        High-throughput C++17 SIMD batch inference engine for LightGBM (AVX2 / AVX-512)
License:        MIT
URL:            https://github.com/1orgar/lgbm-boost
BuildArch:      x86_64

%description
lgbm-boost is a high-throughput C++17 SIMD batch inference engine for LightGBM
gradient boosted decision tree ensembles, featuring 16-row V-QuickScorer and
Compact Branchless AVX2/AVX-512 kernels with automatic runtime CPUID dispatch.

%prep
# Built from pre-compiled wheel verified on the target Oracle Linux distribution

%build
# Nothing to build here

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{py_sitearch}
%{py_bin} -m zipfile -e %{wheel_file} %{buildroot}%{py_sitearch}
%{py_bin} -m compileall -q %{buildroot}%{py_sitearch}/lgbm_boost

%files
%{py_sitearch}/lgbm_boost/
%{py_sitearch}/lgbm_boost-%{version}.dist-info/

%changelog
* Thu Nov 26 2026 Kirill Sholokhov <11299207+1orgar@users.noreply.github.com> - %{pkg_version}-1
- Build native x86_64 SIMD package for Oracle Linux
