# af3_modal.py
import modal

app = modal.App("alphafold3-inference")

# ============================================================
# 镜像定义:严格对照 AlphaFold3 官方 Dockerfile 构建
# ============================================================
af3_image = (
    # 基础镜像:对齐官方的 CUDA 12.6.3 + Ubuntu 24.04 + Python 3.12
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
        add_python="3.12",
    )
    # 系统依赖:编译工具、zlib、patch 等
    .apt_install(
        "git", "wget", "gcc", "g++", "make",
        "zlib1g-dev", "zstd", "patch", "clang",
    )
    # 安装 uv (AF3 官方用 uv 管理 Python 依赖)
    .pip_install("uv==0.9.24")
    # 设置 uv 和 PATH 环境变量
    .env({
        "UV_COMPILE_BYTECODE": "1",
        "UV_PROJECT_ENVIRONMENT": "/alphafold3_venv",
        "PATH": "/hmmer/bin:/alphafold3_venv/bin:/usr/local/bin:/usr/bin:/bin",
    })
    # 创建 uv 虚拟环境
    .run_commands("uv venv /alphafold3_venv")
    # 克隆 AF3 源码(后面需要从里面拿 HMMER 补丁文件)
    .run_commands(
        "git clone https://github.com/google-deepmind/alphafold3.git /app/alphafold",
    )
    # 下载 HMMER 3.4 源码并校验 sha256
    .run_commands(
        "mkdir -p /hmmer_build /hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build",
        "cd /hmmer_build && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz' | sha256sum --check",
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
    })
)

# ============================================================
# 挂载已经上传好的 volume (包含 /databases 和 /parameters)
# ============================================================
af3_volume = modal.Volume.from_name("alphafold3-data")


# ============================================================
# 远程函数:在 H100 上跑 AlphaFold3 推理
# ============================================================
@app.function(
    image=af3_image,
    volumes={"/data": af3_volume},
    gpu="H100",
    cpu=16,
    memory=65536,         # 64 GB,MSA 阶段需要
    timeout=60 * 60 * 3,  # 3 小时超时
)
def run_alphafold3(fasta_json: str, job_name: str = "job"):
    import subprocess
    import pathlib

    # 准备输入输出目录
    input_dir = pathlib.Path("/tmp/af_input")
    output_dir = pathlib.Path("/tmp/af_output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入输入 JSON
    input_path = input_dir / f"{job_name}.json"
    input_path.write_text(fasta_json)

    # 运行 AlphaFold3 主脚本(用 uv run 确保在正确的虚拟环境里)
    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={input_path}",
        "--model_dir=/data/parameters",
        "--db_dir=/data/databases",
        f"--output_dir={output_dir}",
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    # 收集输出文件(.cif 结构 + .json 置信度)
    results = {}
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

    # 保存结果到桌面的 output 文件夹
    out = pathlib.Path(r"C:\Users\Lamarck\Desktop\output\2PV7")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        fp = out / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    print(f"Saved {len(results)} files to {out.resolve()}")