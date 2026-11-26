#!/usr/bin/env bash
set -euo pipefail

# Build native x86_64 Python wheels (.whl) and RPMs (.rpm) inside Oracle Linux 8 (ol8) or Oracle Linux 9 (ol9).
# Usage inside oraclelinux:8 or oraclelinux:9 container:
#   bash packaging/rpm/build_ol_packages.sh ol8 /workspace /workspace/out_ol8
#   bash packaging/rpm/build_ol_packages.sh ol9 /workspace /workspace/out_ol9

OL_VER="${1:?Expected ol8 or ol9}"
SRC_DIR="${2:-/workspace}"
OUT_DIR="${3:-/workspace/out_${OL_VER}}"

mkdir -p "${OUT_DIR}/wheels" "${OUT_DIR}/rpms"

echo "=== Preparing Oracle Linux (${OL_VER}) build toolchain ==="
dnf install -y 'dnf-command(config-manager)' || true

if [[ "${OL_VER}" == "ol8" ]]; then
    dnf config-manager --set-enabled ol8_codeready_builder || true
    dnf install -y \
        gcc-toolset-13-gcc-c++ \
        make \
        rpm-build \
        which \
        findutils \
        python39 python39-devel python39-pip \
        python3.11 python3.11-devel python3.11-pip \
        python3.12 python3.12-devel python3.12-pip
    if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
        # shellcheck disable=SC1091
        source /opt/rh/gcc-toolset-13/enable
    fi
    MANYLINUX_TAG="manylinux_2_28_x86_64"
    RPM_DIST=".el8"
    TARGETS=(
        "/usr/bin/python3.9:python39:cp39"
        "/usr/bin/python3.11:python3.11:cp311"
        "/usr/bin/python3.12:python3.12:cp312"
    )
elif [[ "${OL_VER}" == "ol9" ]]; then
    dnf config-manager --set-enabled ol9_codeready_builder || true
    dnf install -y \
        gcc-c++ \
        make \
        rpm-build \
        which \
        findutils \
        python3 python3-devel python3-pip \
        python3.11 python3.11-devel python3.11-pip \
        python3.12 python3.12-devel python3.12-pip
    MANYLINUX_TAG="manylinux_2_34_x86_64"
    RPM_DIST=".el9"
    TARGETS=(
        "/usr/bin/python3:python3:cp39"
        "/usr/bin/python3.11:python3.11:cp311"
        "/usr/bin/python3.12:python3.12:cp312"
    )
else
    echo "Unsupported target '${OL_VER}'. Expected 'ol8' or 'ol9'." >&2
    exit 1
fi

g++ --version

PKG_VERSION=$(/usr/bin/python3.12 -c "import tomllib; print(tomllib.load(open('${SRC_DIR}/pyproject.toml', 'rb'))['project']['version'])")
echo "Package version: ${PKG_VERSION}"

# Install auditwheel + patchelf once in a Python 3.12 tool venv
AUDIT_VENV="/tmp/audit_venv"
rm -rf "${AUDIT_VENV}"
/usr/bin/python3.12 -m venv "${AUDIT_VENV}"
"${AUDIT_VENV}/bin/pip" install --upgrade pip auditwheel patchelf

# Copy source to clean temp build dir to avoid polluting mounted workspace
WORK_SRC="/tmp/src_copy"
rm -rf "${WORK_SRC}"
mkdir -p "${WORK_SRC}"
cp -a "${SRC_DIR}/CMakeLists.txt" "${SRC_DIR}/LICENSE" "${SRC_DIR}/README.md" "${SRC_DIR}/pyproject.toml" \
      "${SRC_DIR}/include" "${SRC_DIR}/src" "${SRC_DIR}/python" "${SRC_DIR}/tests" "${SRC_DIR}/packaging" \
      "${WORK_SRC}/"

for entry in "${TARGETS[@]}"; do
    IFS=":" read -r PY_BIN RPM_PREFIX PY_ABI <<< "${entry}"
    echo "------------------------------------------------------------"
    echo "Building for ${OL_VER}: ${PY_BIN} (RPM prefix=${RPM_PREFIX}, tag=${PY_ABI})"
    echo "------------------------------------------------------------"

    VENV_DIR="/tmp/venv_${OL_VER}_${PY_ABI}"
    RAW_WHL_DIR="/tmp/raw_whl_${OL_VER}_${PY_ABI}"
    REP_WHL_DIR="/tmp/rep_whl_${OL_VER}_${PY_ABI}"
    rm -rf "${VENV_DIR}" "${RAW_WHL_DIR}" "${REP_WHL_DIR}"
    mkdir -p "${RAW_WHL_DIR}" "${REP_WHL_DIR}"

    "${PY_BIN}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel "numpy>=1.21" "pytest>=8.0"

    # Build binary wheel (PEP 517 isolated build pulls cmake, ninja, scikit-build-core, nanobind)
    "${VENV_DIR}/bin/pip" wheel "${WORK_SRC}" --no-deps -w "${RAW_WHL_DIR}"

    RAW_WHL=$(ls "${RAW_WHL_DIR}"/lgbm_boost-*.whl)
    echo "Built raw wheel: ${RAW_WHL}"

    # Audit & tag wheel for glibc baseline (manylinux_2_28 for OL8, manylinux_2_34 for OL9)
    PATH="${AUDIT_VENV}/bin:${PATH}" "${AUDIT_VENV}/bin/auditwheel" repair "${RAW_WHL}" --plat "${MANYLINUX_TAG}" -w "${REP_WHL_DIR}"
    REPAIRED_WHL=$(ls "${REP_WHL_DIR}"/lgbm_boost-*.whl)
    cp -v "${REPAIRED_WHL}" "${OUT_DIR}/wheels/"

    # Verify wheel in isolated venv
    "${VENV_DIR}/bin/pip" install "${REPAIRED_WHL}"
    "${VENV_DIR}/bin/pytest" "${WORK_SRC}/tests/test_wheel_smoke.py" -v
    "${VENV_DIR}/bin/pip" uninstall -y lgbm-boost

    # Build native x86_64 RPM package for system Python
    PY_SITEARCH=$("${PY_BIN}" -c "import sysconfig; print(sysconfig.get_path('platlib', vars={'platbase': '/usr', 'base': '/usr'}))")
    RPM_TOPDIR="/tmp/rpmbuild_${OL_VER}_${PY_ABI}"
    rm -rf "${RPM_TOPDIR}"
    mkdir -p "${RPM_TOPDIR}"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

    rpmbuild -bb "${WORK_SRC}/packaging/rpm/lgbm-boost.spec" \
        --define "_topdir ${RPM_TOPDIR}" \
        --define "dist ${RPM_DIST}" \
        --define "pkg_version ${PKG_VERSION}" \
        --define "py_pkg_prefix ${RPM_PREFIX}" \
        --define "py_bin ${PY_BIN}" \
        --define "py_sitearch ${PY_SITEARCH}" \
        --define "wheel_file ${REPAIRED_WHL}"

    BUILT_RPM=$(ls "${RPM_TOPDIR}"/RPMS/x86_64/*.rpm)
    cp -v "${BUILT_RPM}" "${OUT_DIR}/rpms/"

    # Smoke-test installing the built RPM into the Oracle Linux container and importing it
    rpm -Uvh --force "${BUILT_RPM}"
    PYTHONPATH="${PY_SITEARCH}" "${VENV_DIR}/bin/python" -c "
import lgbm_boost
assert '${PY_SITEARCH}' in lgbm_boost.__file__, lgbm_boost.__file__
print('Verified RPM import from:', lgbm_boost.__file__, 'ISA:', lgbm_boost.detect_best_isa())
"
    PYTHONPATH="${PY_SITEARCH}" "${VENV_DIR}/bin/pytest" "${WORK_SRC}/tests/test_wheel_smoke.py" -v
done

echo "=== Completed ${OL_VER} artifacts ==="
ls -lh "${OUT_DIR}/wheels" "${OUT_DIR}/rpms"
