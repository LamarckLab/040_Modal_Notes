import modal

app = modal.App("alphafold3-inference")

# ============================================================
# 严格对照 AlphaFold3 官方 Dockerfile 构建 af3_image
# ============================================================
af3_image = (
    # 基础镜像:对齐官方的 CUDA 12.6.3 + Ubuntu 24.04 + Python 3.12
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-base-ubuntu24.04",  # 从 Docker Hub 拉镜像
        add_python="3.12",  # cuda 镜像默认不带 Python (额外装一个)
    )
    # 系统依赖:编译工具、zlib、patch 等
    .apt_install(
        "git",  # 等会要 git clone AF3的仓库
        "wget",  # 下载 HMMER 源码包
        "gcc", "g++", "make",  # 编译 HMMER (C语言写的)
        "zlib1g-dev",  # HMMER 和 AF3 都依赖的压缩库 (负责 .gz 和 .zip)
        "zstd",  # AF3 某些数据用到 (负责 .zst)
        "patch",  # 给 HMMER 打补丁要用
        "clang",  # 编译 AF3 自己的 C++ 拓展
    )
    # 安装 uv (AF3 官方用 uv 管理 Python 依赖)
    .pip_install("uv==0.9.24")
    # 设置 uv 和 PATH 环境变量
    .env({
        # 让 uv 在安装时预编译 .pyc 字节码文件，容器启动时 Python 不用现场编译，启动更快
        "UV_COMPILE_BYTECODE": "1",
        # 显式指定 uv 虚拟环境的路径
        "UV_PROJECT_ENVIRONMENT": "/alphafold3_venv",
        # 把 /hmmer/bin（HMMER 二进制）和 /alphafold3_venv/bin（Python venv 的可执行文件）加到最前面
        # 这样在命令行直接敲 jackhmmer 或 python 时，会找到我们装的版本，而不是系统自带的
        "PATH": "/hmmer/bin:/alphafold3_venv/bin:/usr/local/bin:/usr/bin:/bin",
    })
    # 创建 uv 虚拟环境
    .run_commands("uv venv /alphafold3_venv")
    # 克隆 AF3 源码 (后面需要从里面拿 HMMER 补丁文件)
    .run_commands(
        "git clone https://github.com/google-deepmind/alphafold3.git /app/alphafold",
    )
    # 下载 HMMER 3.4 源码并校验 sha256
    .run_commands(
        # 创建两个目录：/hmmer_build 是临时构建目录，/hmmer 是最终安装目录
        "mkdir -p /hmmer_build /hmmer",
        # 下载 HMMER 3.4 源码包
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build",
        # 校验 sha256 哈希，确保下载的文件没有损坏或被篡改 (这个哈希值是 DeepMind 验证过的，从 Dockerfile中取出来的)
        "cd /hmmer_build && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz' | sha256sum --check",
        # 解压后删除 tar 包，省空间
        "cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
    )
    # 应用 AF3 自带的 jackhmmer seq_limit 补丁(运行时必需)
    .run_commands(
        "cp /app/alphafold/docker/jackhmmer_seq_limit.patch /hmmer_build/",
        "cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch",
    )
    # 编译安装 HMMER 到 /hmmer
    .run_commands(
        "cd /hmmer_build/hmmer-3.4 && ./configure --prefix=/hmmer && make -j4",
        "cd /hmmer_build/hmmer-3.4 && make install",
        "cd /hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /hmmer_build",
    )
    # 用 uv sync 安装 AF3 的 Python 依赖(严格按 lockfile)
    .run_commands(
        "cd /app/alphafold && UV_HTTP_TIMEOUT=1800 uv sync --frozen --all-groups --no-editable",
    )
    # 构建化学组件数据库
    .run_commands(
        "cd /app/alphafold && uv run build_data",
    )
    # AF3 运行时必需的 XLA 环境变量(不设置会导致编译极慢或 OOM)
    .env({
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
        # JAX 一启动就把 GPU 显存的 95% 全部预占，
        # 设计原因：JAX 反复申请/释放 GPU 显存会导致内存碎片化，影响大模型推理性能。
        # 预占可以避免这个问题，代价是"看起来一直在用"。
    })
)

# ============================================================
# 挂载已经上传好的 volume (包含 /databases 和 /parameters)
# ============================================================
af3_volume = modal.Volume.from_name("alphafold3-data")


# ============================================================
# 远程函数定义
# ============================================================
@app.function(
    image=af3_image,
    volumes={"/data": af3_volume},  # 把 volume 挂在到容器的 /data 路径下
    gpu="H100",  # 申请 GPU
    cpu=16,  # 预留 CPU 核心
    memory=32768,  # 预留内存
    timeout=60 * 60 * 3,  # 函数最多跑 3 小时，超时会被强制终止
)
def run_alphafold3(fasta_json: str, job_name: str = "job"):  # 如果调用者没传这个参数，就自动用 "job" 作为它的值
    import subprocess
    import pathlib

    # 在容器的 /tmp 下创建两个目录，/tmp 是容器本地的临时存储，函数结束后会被清空
    input_dir = pathlib.Path("/tmp/af_input")
    output_dir = pathlib.Path("/tmp/af_output")
    # parents=True 表示缺父目录就自动建，exist_ok=True 表示已存在不报错
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把从本地传过来的 JSON 字符串写成容器里的一个文件
    input_path = input_dir / f"{job_name}.json"
    input_path.write_text(fasta_json)

    # 运行 AlphaFold3 主脚本
    # 把命令拆成列表更安全（避免路径里有空格、特殊字符时被 shell 误解析）
    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",  # 用 uv run 在之前建好的虚拟环境里启动 Python
        f"--json_path={input_path}",  # 输入文件位置
        "--model_dir=/data/parameters",  # AF3 模型权重路径
        "--db_dir=/data/databases",  # AF3 数据库路径
        f"--output_dir={output_dir}",  # 输出结果路径
    ]
    # subprocess 要求把命令拆成一个个"词"放进列表里，空格在哪里断开，列表就在哪里断开
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    # 收集输出文件(.cif 结构 + .json 置信度)
    results = {}  # 准备一个字典，键是文件名，值是文件内容
    # 如果文件是 .cif 或者 .json 结尾，就将其存入字典
    for p in output_dir.rglob("*"):
        if p.is_file() and p.suffix in {".cif", ".json"}:
            results[str(p.relative_to(output_dir))] = p.read_text()
    return results


# ============================================================
# 本地入口:读取 JSON、调用远程函数、保存结果
# ============================================================
@app.local_entrypoint()
def main():
    import pathlib

    # 读取本地输入 JSON
    input_file = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_fold_input.json")
    fasta_json = input_file.read_text(encoding="utf-8")

    # 提交到 Modal 运行
    print("Submitting job to Modal...")
    results = run_alphafold3.remote(fasta_json, job_name="2PV7")

    # 把 result 字典里每个文件写回桌面的 output 文件夹
    out = pathlib.Path(r"C:\Users\Lamarck\Desktop\output\2PV7")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        fp = out / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    print(f"Saved {len(results)} files to {out.resolve()}")