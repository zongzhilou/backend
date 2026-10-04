# backend

昇腾（Ascend NPU）平台插件。

## 安装

```bash
pip uninstall transformer_engine torch
pip install "backend @ git+https://github.com/zongzhilou/backend.git"
```

## 快速上手

训练脚本在 `import torch` 下新增一行 `import backend`，即可启用昇腾平台与
Megatron 适配（等价于 MegatronAdaptor 的 `import megatron_adaptor`，并自动
完成宿主平台注册）：

```python
import torch
import backend
```

在装有 CANN 的昇腾机器上直接可用。在无 CANN 的机器上 `import torch` 会因
torch_npu 自动加载失败而报错，需先执行：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`import backend` 写在 `import torch` 之前时，本插件会自动禁用该机制，
无需设置环境变量。
