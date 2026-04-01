#  ☁️Modal Notes

---

### 001 -- Modal介绍
```text
Modal 是一个无服务器云计算平台，允许用 Python 定义远程函数并调用云端 GPU/CPU 资源，无需管理服务器与环境。通过 Image 声明依赖，实现代码级调度，适合 AI 与高性能计算任务。
```

### 002 -- 官方的 Hello World 
<img src="images/pic1.jpg" alt="pic1" width="800">

```text
通过 modal run xxx.py 来提交任务
```

### 003 -- Modal 的 App 类
```text
App 实例是Modal的"顶层组织单元"，用于注册、组织远程函数及其依赖资源，并作为任务提交到云端执行的入口
```

### 004 -- Modal 的两个常用装饰器
```text
- @app.function - 定义云端执行函数 & 分配计算资源
- @app.local_entrypoint - 定义本地入口函数
```

### 005 -- Modal 的执行链路
```bash
App
 ├── Function (@app.function)
 │     └── 运行在远程容器（image）
 ├── Image（运行环境）
 ├── Volume / Secret / Queue（资源）
 └── Entrypoint（本地入口）
```

### 006 -- .remote()
```text
.remote() 用于将函数调用并提交到云端执行
```
