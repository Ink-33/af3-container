# AF3 最新版轻量容器 —— 供 apptainer 使用
#
# 设计要点（重量 vs 能力的取舍）：
#   1. alphafold3.cpp 是纯 C++/pybind11，编译只需 g++，不需要 CUDA toolkit
#   2. jax[cuda12] / rdkit / tokamax 都是预编译 wheel，装它们不需要 CUDA
#   3. 但 AF3 的 Pallas/Triton kernel 在**运行时**由 XLA 调 ptxas 编 PTX→SASS
#      （has_triton_support() 的 Ampere 检查被源码注释掉了，V100 也会走 Triton）
#      而 apptainer run --nv 只绑驱动库、不绑 toolkit → 镜像里必须带 ptxas
#   4. ptxas 通过 pip 装 nvidia-cuda-nvcc-cu12 获取，避免拉 6 GB CUDA devel 镜像
#
# 为什么 ubuntu24.04：glibc 2.39 满足 manylinux_2_28；仓库自带 Python 3.12。
# 集群是 RHEL7 / glibc 2.17，装不下 jax 0.10.2 / rdkit 2026.3.3 / tokamax。
#
# 本地构建：
#   podman build --squash -t af3-new:latest -f Dockerfile .
#   podman save -o af3-new.tar docker-archive:af3-new:latest
# 集群转 SIF：
#   apptainer build --fakeroot af3_new.sif docker-archive:af3-new.tar

# ---------- 阶段 1：编译 AF3 ----------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
# 注意：不要在这里写 http_proxy —— CI 有直连网络；本地构建用
# podman Desktop 的代理设置或 --network=host + 系统代理。
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_RETRIES=8
ENV PIP_TIMEOUT=120
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential cmake ninja-build \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# AF3 运行时依赖（版本按 af3_src/pyproject.toml 锁死）
# nvidia-cuda-nvcc-cu12 提供 ptxas（XLA JIT 需要），替代拉 CUDA devel 镜像
RUN pip install --prefer-binary \
        "absl-py>=2.3.1" \
        "dm-haiku==0.0.17" \
        "etils[epath]" \
        "jax==0.10.2" \
        "jax[cuda12]==0.10.2" \
        "msgpack" "numpy" \
        "rdkit==2026.3.3" \
        "tokamax==0.0.12" \
        "tqdm" "zstandard" \
        "scikit-build-core" "pybind11" "cmake>=3.28" "ninja" \
        "setuptools-scm" \
        "nvidia-cuda-nvcc-cu12"

# AF3 源码 + 编译安装
COPY src/alphafold3-main.tar.gz /tmp/af3.tar.gz
RUN mkdir -p /opt/af3 /opt/libcifpp \
    && tar -xzf /tmp/af3.tar.gz -C /opt/af3 --strip-components=1 \
    && rm /tmp/af3.tar.gz \
    # converters/ 没有 __init__.py，scikit_build_core 不会打进 wheel；
    # 补一个空文件让 resources.files() 能正确定位输出目录
    && touch /opt/af3/src/alphafold3/constants/converters/__init__.py \
    && cd /opt/af3 \
    && pip install --no-build-isolation --prefer-binary . \
    # CCD 化学组件字典：build_data 用它生成 ccd.msgpack /
    # chemical_component_sets.msgpack，否则 alphafold3.model.model 导入会 FileNotFoundError
    # 注意 files.rcsb.org 的 .gz 是 404，正确源是 files.wwpdb.org
    && wget -q --tries=3 -O /tmp/components.cif.gz \
         https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz \
    && gunzip -c /tmp/components.cif.gz > /opt/libcifpp/components.cif \
    && rm /tmp/components.cif.gz

# 单独一步跑 build_data：解析 120 MB components.cif 吃内存，和 pip install 挤一起会 OOM
# MALLOC_ARENA_MAX=1 减少 glibc 多 arena 内存碎片
RUN MALLOC_ARENA_MAX=1 LIBCIFPP_DATA_DIR=/opt/libcifpp build_data \
    # 显式验证 msgpack 已落在 site-packages 的 converters/ 里
    && DST=$(python -c "import alphafold3.common.resources as r; print(r.ROOT / 'constants/converters')") \
    && echo "DST=$DST" && ls -la "$DST/" \
    && test -f "$DST/ccd.msgpack" && test -f "$DST/chemical_component_sets.msgpack"

# ---------- 阶段 2：轻量运行时 ----------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LIBCIFPP_DATA_DIR=/opt/libcifpp

# 运行时系统库（python3.12 的共享库 + AF3 扩展会用到的 libstdc++/libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libpython3.12 libgomp1 libglib2.0-0 \
        libgl1 libxrender1 libxext6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 拷 venv、AF3 源码、CCD 数据
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/af3 /opt/af3
COPY --from=builder /opt/libcifpp /opt/libcifpp

# ptxas：从 pip nvidia-cuda-nvcc-cu12 包里拿到 XLA 标准搜索路径
RUN mkdir -p /usr/local/cuda/bin \
    && ln -sf /opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
              /usr/local/cuda/bin/ptxas

# 自检
RUN python -c "import importlib;\
mods=['alphafold3','alphafold3.cpp','alphafold3.common.folding_input','alphafold3.model.params','alphafold3.model.model','jax','haiku','etils','tokamax','rdkit'];\
[print('OK', m, getattr(importlib.import_module(m),'__version__','')) for m in mods];\
import shutil; print('ptxas:', shutil.which('ptxas'))"

WORKDIR /work
CMD ["python", "/opt/af3/run_alphafold.py", "--help"]
