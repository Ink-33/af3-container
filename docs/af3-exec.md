# af3-exec.sh — sandbox magic wrapper

在集群上运行 AF3 sandbox 的轻量 wrapper：不经过 `apptainer exec`，直接调用
sandbox rootfs 自带的 `ld-linux-x86-64.so.2` 启动容器内命令。

## 背景

镜像构建出 docker-archive 后，可用

```bash
apptainer build --fakeroot --sandbox af3_new.sandbox docker-archive:af3-new.tar
```

展开成 sandbox 目录树。正常途径是 `apptainer exec` / `apptainer run` 进入
sandbox，但当宿主禁用了 user namespace（如 sysctl `max_user_namespaces=0`）
时这条路走不通。

sandbox 本质是一个完整的 Linux rootfs（Ubuntu 24.04 / glibc 2.39）；若宿主
glibc 远旧于容器（如 RHEL7 / glibc 2.17），宿主的加载器跑不了容器里的二进制。但容器内 ELF 记录的
解释器就是 `/lib64/ld-linux-x86-64.so.2`，它就在 sandbox 里：直接调用这个
动态加载器、用 `--library-path` 把 sandbox 的库目录喂给它，就能在不进
namespace、不要 root 的情况下跑容器里的 Python 和 AF3。

## 原理

- **加载器**：`$SB/lib64/ld-linux-x86-64.so.2 --library-path <sandbox 库> <目标二进制>`。
  容器自己的 glibc 2.39 加载容器自己的所有依赖，宿主 glibc 完全不参与。
- **库搜索顺序**：NVIDIA 用户态库（nvcc / cublas / cudnn / nccl）→ venv →
  sandbox 系统库，保证容器内 CUDA 库优先于宿主。
- **Python 定位**：`PYTHONHOME=$SB/usr`，`PYTHONPATH` 指向 sandbox 内的 venv
  site-packages 与 AF3 源码树，解释器和包都来自 sandbox。
- **命令映射**：`python` / `python3` / `python3.12` → sandbox 的
  `usr/bin/python3.12`；其他命令依次查 `usr/bin/`、`bin/`。
- 不需要 apptainer、不需要 proot、不需要 root、不需要 user namespace。

## 脚本

把 `SB` 一行改成你的 sandbox 实际路径即可用：

```bash
#!/bin/bash
# af3-exec.sh — 用容器自己的 ld-linux + glibc 直接跑 AF3 容器内命令
# 绕过 apptainer user namespace 限制（如宿主 max_user_namespaces=0）
#
# 用法: af3-exec.sh <command> [args...]
#   af3-exec.sh python --version
#   af3-exec.sh python /opt/af3/run_alphafold.py --help

set -e

SB=/path/to/af3_new.sandbox        # ← 改成你的 sandbox 路径
LD=$SB/lib64/ld-linux-x86-64.so.2

# 容器内共享库搜索路径（按优先级）
LIBPATH="$SB/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/lib"
LIBPATH="$LIBPATH:$SB/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib"
LIBPATH="$LIBPATH:$SB/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
LIBPATH="$LIBPATH:$SB/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib"
LIBPATH="$LIBPATH:$SB/opt/venv/lib"
LIBPATH="$LIBPATH:$SB/usr/lib/x86_64-linux-gnu"
LIBPATH="$LIBPATH:$SB/lib/x86_64-linux-gnu"
LIBPATH="$LIBPATH:$SB/usr/lib:$SB/lib"

# 环境变量
export VIRTUAL_ENV=/opt/venv
export PATH="$SB/usr/local/cuda/bin:$SB/opt/venv/bin:$SB/usr/bin:$SB/bin:$PATH"
export LIBCIFPP_DATA_DIR=$SB/opt/libcifpp
export PYTHONUNBUFFERED=1
export PYTHONHOME=$SB/usr
export PYTHONPATH="$SB/opt/venv/lib/python3.12/site-packages:$SB/opt/af3/src"

# 解析命令：把容器内绝对路径映射到 sandbox 路径
case "$1" in
  python|python3|python3.12)
    shift
    exec "$LD" --library-path "$LIBPATH" "$SB/usr/bin/python3.12" "$@"
    ;;
  *)
    CMD="$SB/usr/bin/$1"
    if [ ! -f "$CMD" ]; then
      CMD="$SB/bin/$1"
    fi
    shift
    exec "$LD" --library-path "$LIBPATH" "$CMD" "$@"
    ;;
esac
```

## 用法

```bash
./af3-exec.sh python --version
./af3-exec.sh python /opt/af3/run_alphafold.py \
    --json_path=in.json --output_dir=out --model_dir=/db --db_dir=/db
```

作业脚本里把 wrapper 当作启动器用（资源请求与业务命令解耦，调度器不限）：

```bash
#!/bin/bash
# 作业脚本（LSF/Slurm/直跑均同理），示例：

EXEC=/path/to/container/af3-exec.sh
$EXEC python /opt/af3/run_alphafold.py --json_path=in.json --output_dir=out
```

## 前提与注意

1. sandbox 先用 `apptainer build --fakeroot --sandbox` 从 docker-archive 转出
   （见上文背景），wrapper 只认展开后的目录树，不认 SIF 单文件。
2. 宿主内核需满足 Ubuntu 24.04 userland 的系统调用要求；NVIDIA 驱动装在宿主，
   CUDA 用户态库由镜像里的 pip 包提供（见 Dockerfile 的 ptxas 说明）。
3. `LIBCIFPP_DATA_DIR` 等镜像内约定随 sandbox 一起生效，不要在宿主重复配置。
