#  ☁️Modal Notes

---

### 001 -- Modal 介绍
```text
Modal 是一个无服务器云计算平台，允许用 Python 定义远程函数并调用云端 GPU/CPU 资源，无需管理服务器与环境。通过 Image 声明依赖，实现代码级调度，适合 AI 与高性能计算任务。
```

### 002 -- 官方的 Hello World 
<img src="images/pic1.png" alt="pic1" width="900">

```text
通过在终端运行 modal run xxx.py 来提交任务
```

### 003 -- Modal 的 App 类
```text
App 实例是 Modal 的"顶层组织单元"，用于注册、组织远程函数及其依赖资源，并作为任务提交到云端执行的入口
```

### 004 -- Modal 的 image 容器
```text
Modal image 是供远程函数运行的容器环境，这个远程环境里包括：
- 操作系统基础层
- Python 解释器
- Python 包
- 环境变量
- ...
```

### 005 -- Modal 的两个常用装饰器
```text
- @app.function - 定义云端执行函数 & 分配计算资源
- @app.local_entrypoint - 定义本地入口函数
```

### 006 -- Modal 的执行链路
```bash
App
 ├── Function (@app.function)
 │     └── 运行在远程容器（image）
 ├── Image（运行环境）
 ├── Volume / Secret / Queue（资源）
 └── Entrypoint（本地入口）
```

### 007 -- .remote( )方法
```text
.remote() 用于将函数调用并提交到云端执行
```

### 008 -- Modal 提供的 GPU 类型
| GPU 型号                            | 显存 (VRAM)              | 价格      |
| :---------------------------------- | :----------------------- | :-------- |
| T4                                  | 16 GB                    | $0.59 / h |
| L4                                  | 24 GB                    | $0.80 / h |
| A10                                 | 24 GB                    | $1.10 / h |
| L40S                                | 48 GB                    | $1.95 / h |
| A100-40GB                           | 40 GB                    | $2.10 / h |
| A100-80GB                           | 80 GB                    | $2.50 / h |
| RTX PRO 6000（一般指 RTX 6000 Ada） | 48 GB                    | $3.03 / h |
| H100                                | 80 GB                    | $3.95 / h |
| H200                                | 141 GB                   | $4.54 / h |
| B200 / B200+                        | 192 GB（Blackwell 架构） | $6.25 / h |

### 009 -- 指定远程函数的 GPU 类型和数量
1. 指定 GPU 类型
```python
@app.function(gpu="B200") # 使用 1 张 B200
```

2. 使用多张卡
```python
@app.function(gpu="H100:8")
def run_llama_405b_fp8():
    ...
```

3. 回退机制
```python
@app.function(gpu=["H100", "A100-40GB:2"]) # 优先使用 H100，如果没有可用资源，则回退到 2 张 A100-40GB
def run_on_80gb():
    ...
```
#### [Modal GPU Doc](https://modal.com/docs/guide/gpu)

### 010 -- Modal image 的定义
>**在 Modal 中定义 image 的典型流程是：从一个基础镜像 (base Image) 开始，通过方法链 (method chaining) 逐步构建，可以为每一个函数单独定义不同的运行环境**
```python
image = (
    modal.Image.debian_slim(python_version="3.13") # 创建一个基础镜像 (精简版 Debian Linux，Python version 3.13)
    .apt_install("git") # 用 apt 安装 git
    .uv_pip_install("torch<3") # 用 uv 安装 torch，要求版本小于 3
    .env({"HALT_AND_CATCH_FIRE": "0"}) # 给 image 设置环境变量
    .run_commands("git clone https://github.com/modal-labs/agi && echo 'ready to go!'") # 构建这个 image 时，运行这条 shell 命令
)
```
#### [直接使用已有镜像，比如从DockerFile导入](https://modal.com/docs/guide/existing-images)

### 011 -- 在 image 中添加 Python 包
> **可以通过将所需的所有包传给 image.uv_pip_install 方法，将 Python 包添加到环境中**

> **建议严格固定依赖版本，比如"pandas==2.2.0"、"torch<3"，以提高构建的可复现性**
```python
datascience_image = (
    modal.Image.debian_slim()
    .uv_pip_install("pandas==2.2.0", "numpy")
)

@app.function(image=datascience_image)
def my_function():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame()
    ...
```

> **如果在使用 image.uv_pip_install 时遇到问题，你可以回退使用 image.pip_install，它使用标准的 pip**

```python
datascience_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pandas==2.2.0", "numpy")
)
```

### 012 -- 把本地文件传递到 image 中
> **有时容器需要一些无法从互联网获取的依赖，可以使用 image.add_local_dir 和 image.add_local_file 方法，将本地系统中的文件传递到容器中**
>

```python
image = modal.Image.debian_slim().add_local_dir("/user/erikbern/.aws", remote_path="/root/.aws")
```

### 013 -- 导入包写在函数内部
> **若本地未安装某包（如 pandas），不要在脚本顶部全局 import，否则会报 ImportError；应将 import 写在函数内部，使其仅在远程容器运行时加载（容器中已安装该包）**

```python
datascience_image = (
    modal.Image.debian_slim()
    .uv_pip_install("pandas==2.2.0", "numpy")
)

@app.function(image=datascience_image)
def my_function():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame()
    ...
```

> **如果有多个函数，且共享部分依赖，可以使用 image.imports 实现全局作用**

```python
pandas_image = modal.Image.debian_slim().pip_install("pandas", "numpy")

with pandas_image.imports():
    import pandas as pd
    import numpy as np

@app.function(image=pandas_image)
def my_function():
    df = pd.DataFrame()
    ...
```
