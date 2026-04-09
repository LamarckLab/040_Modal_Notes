import modal

app = modal.App("alphafold3-split")

# ============================================================
# 严格对照 AlphaFold3 官方 Dockerfile 构建 af3_image
# ============================================================

af3_image = (
    # 基础镜像:对齐官方的 CUDA 12.6.3 + Ubuntu 24.04 + Python 3.12
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-base-ubuntu24.04",  # 从 Docker Hub 拉镜像
        add_python="3.12",  # cuda 镜像默认不带 Python (额外装一个)
    )
    # 安装系统依赖
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
        # 把 /hmmer/bin（HMMER 二进制）和 /alphafold3_venv/bin (Python venv 的可执行文件) 加到最前面
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
        # JAX 一启动就把 GPU 显存的 95% 全部预占
        # 设计原因：JAX 反复申请/释放 GPU 显存会导致内存碎片化，影响大模型推理性能
        # 预占可以避免这个问题，代价是"看起来一直在用"
    })
)

# ============================================================
# Volume 声明
# - af3_volume: 已有的数据库和权重(只读)
# - msa_cache_volume: 存 MSA 中间结果(读写),首次运行会自动创建
# ============================================================
af3_volume = modal.Volume.from_name("alphafold3-data")
msa_cache_volume = modal.Volume.from_name(
    "alphafold3-msa-cache",
    create_if_missing=True,
)


# ============================================================
# 辅助函数:计算序列哈希,用于缓存去重
# ============================================================
def compute_sequence_hash(fasta_json: str) -> str:
    """
    根据输入 JSON 里的序列内容计算一个稳定的哈希值。
    同一组序列(顺序无关)总是得到相同的哈希。
    用作 MSA 缓存的 key,实现"同序列自动复用 MSA"。
    """
    import hashlib
    import json

    data = json.loads(fasta_json)
    # 提取所有序列,排序后拼接(保证顺序无关)
    seqs = []
    for entry in data.get("sequences", []):
        for mol_type in ("protein", "rna", "dna"):
            if mol_type in entry:
                seqs.append(f"{mol_type}:{entry[mol_type]['sequence']}")
    seqs.sort()
    joined = "|".join(seqs)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# ============================================================
# 函数 1: 数据管线(MSA 搜索)
# 资源画像: CPU 密集,无 GPU,中等内存
# ============================================================
@app.function(
    image=af3_image,
    volumes={
        "/data": af3_volume,
        "/msa_cache": msa_cache_volume,
    },
    cpu=16,
    memory=16384,
    timeout=60 * 60 * 2,
)
def run_data_pipeline(fasta_json: str, job_name: str) -> str:
    """
    跑 AlphaFold3 的数据管线阶段(MSA + 模板搜索)。

    Returns:
        seq_hash: 序列哈希,作为 MSA 缓存的 key,后续推理阶段用它找中间文件
    """
    import subprocess
    import pathlib
    import shutil

    seq_hash = compute_sequence_hash(fasta_json)
    cache_dir = pathlib.Path(f"/msa_cache/{seq_hash}")

    # 检查缓存命中:如果已经跑过,直接返回
    if cache_dir.exists() and any(cache_dir.rglob("*_data.json")):
        print(f"[cache hit] MSA already exists for seq_hash={seq_hash}, skipping data pipeline")
        return seq_hash

    print(f"[cache miss] Running data pipeline for seq_hash={seq_hash}")

    # 准备输入输出目录
    input_dir = pathlib.Path("/tmp/af_input")
    output_dir = pathlib.Path("/tmp/af_output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{job_name}.json"
    input_path.write_text(fasta_json)

    # 运行 AF3,只做数据管线,不做推理
    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={input_path}",
        "--db_dir=/data/databases",
        f"--output_dir={output_dir}",
        "--norun_inference",  # 关键 flag:跳过推理
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    # 把 AF3 生成的中间文件拷贝到 MSA 缓存 volume
    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_dir():
            # AF3 会在 output_dir 下建一个以 job_name 命名的子目录
            dest = cache_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # 重要:提交 volume 修改,让其他容器能看到新文件
    msa_cache_volume.commit()
    print(f"[done] MSA cached at /msa_cache/{seq_hash}")

    return seq_hash


# ============================================================
# 函数 2: 模型推理
# 资源画像: GPU 密集,少量 CPU,中等内存
# ============================================================
@app.function(
    image=af3_image,
    volumes={
        "/data": af3_volume,
        "/msa_cache": msa_cache_volume,
    },
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 1,
)
def run_inference(seq_hash: str, job_name: str) -> dict:
    """
    从 MSA 缓存读取中间文件,跑 diffusion 采样生成结构。

    Args:
        seq_hash: 由 run_data_pipeline 返回的序列哈希
        job_name: AF3 任务名,决定输出子目录名

    Returns:
        results: {相对路径: 文件内容} 字典,包含 .cif 和 .json
    """
    import subprocess
    import pathlib
    import shutil

    # 从 volume 刷新最新状态(可能是别的容器刚写的)
    msa_cache_volume.reload()

    cache_dir = pathlib.Path(f"/msa_cache/{seq_hash}")
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"MSA cache not found at {cache_dir}. "
            f"Did you run run_data_pipeline first?"
        )

    # 把缓存的中间文件拷贝到 AF3 的工作目录
    # AF3 要求 --json_path 指向包含 MSA 的中间 JSON
    work_dir = pathlib.Path("/tmp/af_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(cache_dir, work_dir)

    # 找到中间 JSON 文件(AF3 生成的是 <job_name>_data.json)
    data_jsons = list(work_dir.rglob("*_data.json"))
    if not data_jsons:
        raise FileNotFoundError(f"No *_data.json found in cache {cache_dir}")
    data_json_path = data_jsons[0]
    print(f"Using MSA data file: {data_json_path}")

    # 输出目录
    output_dir = pathlib.Path("/tmp/af_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行 AF3,跳过数据管线,只做推理
    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={data_json_path}",
        "--model_dir=/data/parameters",
        f"--output_dir={output_dir}",
        "--norun_data_pipeline",  # 关键 flag:跳过数据管线
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    # 收集输出(.cif 结构 + .json 置信度)
    results = {}
    for p in output_dir.rglob("*"):
        if p.is_file() and p.suffix in {".cif", ".json"}:
            results[str(p.relative_to(output_dir))] = p.read_text()
    return results


# ============================================================
# 辅助函数:把结果字典保存到本地
# ============================================================
def save_results_locally(results: dict, local_out_dir):
    """把 {相对路径: 内容} 字典写到本地目录。"""
    import pathlib
    out = pathlib.Path(local_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        fp = out / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    print(f"Saved {len(results)} files to {out.resolve()}")


# ============================================================
# 入口 1: 完整流程(MSA + 推理)
# ============================================================
@app.local_entrypoint()
def main():
    """完整流程:先跑数据管线(如果缓存未命中),再跑推理。"""
    import pathlib

    # 配置区:按需修改
    input_file = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_fold_input.json")
    job_name = "2PV7"
    output_base = pathlib.Path(r"C:\Users\Lamarck\Desktop\output")

    fasta_json = input_file.read_text(encoding="utf-8")

    # 阶段 1: 数据管线(可能命中缓存直接返回)
    print("=" * 60)
    print(f"[Stage 1/2] Running data pipeline for job: {job_name}")
    print("=" * 60)
    seq_hash = run_data_pipeline.remote(fasta_json, job_name)
    print(f"Data pipeline done. seq_hash = {seq_hash}")

    # 阶段 2: 推理
    print("=" * 60)
    print(f"[Stage 2/2] Running inference")
    print("=" * 60)
    results = run_inference.remote(seq_hash, job_name)

    # 保存结果
    save_results_locally(results, output_base / job_name)


# ============================================================
# 入口 2: 只跑推理(跳过 MSA 阶段)
# 用法: modal run af3_modal_v1_full.py::only_inference --job-name 2PV7
# 前提: 对应序列的 MSA 必须已经在缓存里
# ============================================================
@app.local_entrypoint()
def only_inference(
    job_name: str = "2PV7",
    input_json: str = r"C:\Users\Lamarck\Desktop\af3_fold_input.json",
):
    """
    只跑推理,跳过 MSA。
    需要提供原始输入 JSON 用来计算序列哈希,定位 MSA 缓存。
    """
    import pathlib

    input_file = pathlib.Path(input_json)
    fasta_json = input_file.read_text(encoding="utf-8")

    # 根据输入序列算出 seq_hash,定位缓存
    seq_hash = compute_sequence_hash(fasta_json)
    print(f"Looking up MSA cache for seq_hash = {seq_hash}")

    # 直接跑推理
    results = run_inference.remote(seq_hash, job_name)

    # 保存结果
    output_base = pathlib.Path(r"C:\Users\Lamarck\Desktop\output")
    save_results_locally(results, output_base / f"{job_name}_rerun")

# ============================================================
# 入口 3: 只跑 MSA(预热缓存,不推理)
# 用法: modal run af3_modal_v2_split.py::only_msa
# ============================================================
@app.local_entrypoint()
def only_msa(
    input_json: str = r"C:\Users\Lamarck\Desktop\af3_fold_input.json",
    job_name: str = "2PV7",
):
    """
    只跑数据管线(MSA + 模板搜索),把结果存到缓存 volume,不做推理。
    适合提前预热缓存,或批量预处理多个序列的 MSA。
    """
    import pathlib

    input_file = pathlib.Path(input_json)
    fasta_json = input_file.read_text(encoding="utf-8")

    print(f"Running MSA only for job: {job_name}")
    seq_hash = run_data_pipeline.remote(fasta_json, job_name)
    print(f"MSA done. seq_hash = {seq_hash}")
    print(f"You can now run inference with:")
    print(f"  modal run af3_modal_v2_split.py::only_inference --job-name {job_name}")