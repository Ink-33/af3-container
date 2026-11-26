# af3-container

AlphaFold 3 **v3.0.5** 容器镜像（轻量多阶段构建），供集群 `apptainer` 使用。

## 为什么需要这个镜像

集群是 RHEL7 / **glibc 2.17**，装不下 manylinux_2_28 的 wheel（`jax 0.10.2`、
`rdkit 2026.3.3`、`tokamax 0.0.12`）。容器内用 Ubuntu 24.04（glibc 2.39）解决。

## 轻量化设计

| 组件 | 占用 | 说明 |
|------|------|------|
| `ubuntu:24.04` 基底 | 81 MB | 不用 `nvidia/cuda-devel`（7.36 GB） |
| `ptxas`（从 toolchain 阶段单独拷） | ~50 MB | XLA 运行时 JIT 需要；`--nv` 只绑驱动不绑 toolkit |
| jax + CUDA 库 | ~2.5 GB | 必需；`--nv` 不绑 cublas/cudnn |
| rdkit / tokamax / AF3 | ~600 MB | |
| **合计** | **约 3.2 GB** | |

多阶段构建：`toolchain` → `builder` → `runtime`，最终镜像不含编译工具链
也不含代理配置。

## 构建

由 GitHub Actions 自动构建并推到 `ghcr.io`。

## 集群上使用

```bash
# 转 SIF
apptainer build --fakeroot af3_new.sif docker-archive:af3-new.tar.gz
# 或从 ghcr.io 拉
apptainer build --fakeroot af3_new.sif docker://ghcr.io/ink-33/af3-container:latest

# 运行（--nv 绑宿主 NVIDIA 驱动）
apptainer run --nv \
  -B /share/apps/af3/alphafold3/database:/db:ro \
  af3_new.sif \
  python /opt/af3/run_alphafold.py --json_path=in.json --output_dir=out --model_dir=/db --db_dir=/db
```

## 关键实现细节

- **`ptxas` 必须在镜像里**：AF3 的 Pallas/Triton kernel 在运行时由 XLA JIT
  编译 PTX→SASS。`has_triton_support()` 的 Ampere 检查在源码里被注释掉了，
  V100 也会走 Triton 路径。`apptainer --nv` 只绑驱动库，不绑 toolkit。
- **CCD 数据**：`alphafold3.model.model` 导入需要
  `chemical_component_sets.msgpack`，由 `python -m alphafold3.build_data`
  从 `components.cif`（wwPDB）生成。
- **`--flash_attention_implementation xla`**：Triton/cuDNN flash attention 要
  Ampere (sm_80) 以上，V100 是 sm_70，会报
  `ptxas: Feature '.bf16' requires .target sm_80 or higher`。
- **PTM 写法**：本地 AF3（`dialect: "alphafold3"`）的 `ptmType` 不带 `CCD_`
  前缀，写 `"TPO"`；Server 格式才写 `"CCD_TPO"`。
