# Nova LLM

从零实现的大语言模型训练与推理系统。包含模型架构、分词器、数据处理、训练流水线（预训练 / SFT / DPO）、推理引擎、工具调用与 OpenAI 兼容 API 服务。

> 架构参考 Llama 系列：RMSNorm + RoPE + SwiGLU + GQA，支持 1B / 7B / 13B / 70B 与 MoE、多模态扩展。

---

## 架构总览

```
                        ┌──────────────────────────────────────┐
                        │            nova_llm/                 │
                        ├──────────────────────────────────────┤
   ┌──────────┐         │  model/                             │
   │  configs │────────▶│    config.py   NovaConfig / presets │
   │  (yaml)  │         │    transformer.py  NovaTransformer   │
   └──────────┘         │    moe.py / multimodal.py           │
                        └──────────────┬───────────────────────┘
                                      │
   ┌──────────┐         ┌──────────────┴───────────────┐
   │  data/   │────────▶│  train/                      │
   │ dataset  │         │    pretrain / sft / dpo      │
   └──────────┘         └──────────────┬───────────────┘
                                      │ checkpoints/
                                      ▼
                        ┌──────────────────────────────┐
                        │  inference/                  │
                        │  TextGenerator / Sampler     │◀──── tokenizer/
                        │  KV-Cache / quantize         │      (BPE)
                        └──────┬───────────────┬───────┘
                               │               │
                    ┌──────────▼─────┐   ┌─────▼─────────┐
                    │  tools/         │   │  server/      │
                    │  ToolRegistry  │   │  FastAPI      │
                    │  FunctionCall   │   │  /v1/chat...  │
                    │  CodeInterp.   │   │  (OpenAI 兼容)│
                    └─────────────────┘   └───────────────┘
```

## 目录结构

```
nova-llm/
├── nova_llm/
│   ├── model/          # Transformer、MoE、多模态、配置
│   ├── tokenizer/      # BPE 分词器
│   ├── data/           # 数据处理
│   ├── train/          # pretrain / sft / dpo
│   ├── inference/      # 生成器、采样、KV cache、量化
│   ├── tools/          # 工具注册、Function Calling、代码解释器
│   └── server/        # OpenAI 兼容 API
├── configs/           # 预训练 / SFT / DPO / 推理 YAML
├── scripts/           # 启动脚本与工具脚本
├── examples/          # 6 个可运行示例
└── checkpoints/        # 模型权重输出
```

## 安装

```bash
git clone <repo> && cd nova-llm
pip install -e .
# 服务端额外依赖
pip install -e .[server]
```

## 快速开始

### 1. 前向传播

```bash
python examples/01_forward_pass.py
```

### 2. 训练一步

```bash
python examples/02_train_one_step.py
```

### 3. 文本生成

```bash
# 随机权重演示接口
python examples/03_generate_text.py --prompt "你好"
# 加载真实权重
python examples/03_generate_text.py --model checkpoints/sft-1b
```

### 4. 工具调用

```bash
python examples/04_function_calling.py
```

### 5. 量化对比

```bash
python examples/05_quantize.py
```

### 6. API 客户端

```bash
# 终端 1：启动服务
python -m nova_llm.server.api_server --model checkpoints/sft-1b --port 8000
# 终端 2：调用
python examples/06_api_client.py
```

## 模型架构

| 规模 | hidden | layers | heads | KV heads | intermediate | 备注 |
|------|--------|--------|-------|----------|--------------|------|
| 1B   | 2048   | 24     | 32    | 8        | 5632         | GQA  |
| 7B   | 4096   | 32     | 32    | 8        | 11008        | GQA  |
| 13B  | 5120   | 40     | 40    | 10       | 13824        | GQA  |
| 70B  | 8192   | 80     | 64    | 8        | 28672        | 密集 |
| 8x7B-MoE | 4096 | 32  | 32    | 8        | 1408/expert  | top-2 |

核心组件：RMSNorm、旋转位置编码 RoPE、SwiGLU 激活、分组查询注意力 GQA。

```python
from nova_llm.model import NovaConfig, NovaTransformer, get_config

cfg = get_config("1B")
model = NovaTransformer(cfg)
```

## 训练指南

### 预训练

```bash
# 单卡
bash scripts/train_pretrain.sh configs/pretrain_1b.yaml checkpoints/pretrain-1b
# 多卡
NPROC=4 bash scripts/train_pretrain.sh configs/pretrain_7b.yaml checkpoints/pretrain-7b
```

### SFT 监督微调

```bash
RESUME=checkpoints/pretrain-1b bash scripts/train_sft.sh configs/sft.yaml checkpoints/sft-1b
```

### DPO 偏好优化

```bash
RESUME=checkpoints/sft-1b bash scripts/train_dpo.sh configs/dpo.yaml checkpoints/dpo-1b
```

## 推理部署

```bash
python -m nova_llm.inference.generator --model checkpoints/sft-1b --interactive
# 或
bash scripts/generate.sh checkpoints/sft-1b
```

## API 文档（OpenAI 兼容）

启动：

```bash
python -m nova_llm.server.api_server --model checkpoints/sft-1b --host 0.0.0.0 --port 8000
# 或
bash scripts/serve.sh checkpoints/sft-1b
```

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| GET  | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 对话补全（支持 `stream`、`tools`） |
| POST | `/v1/completions` | 文本补全 |
| POST | `/v1/embeddings` | 向量嵌入 |

### 示例：cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nova","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

### 示例：Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
resp = client.chat.completions.create(
    model="nova",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in resp:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 流式格式

SSE 每个 chunk：

```
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

```

结束时：`data: [DONE]\n\n`

## 工具调用

内置工具：`get_current_time`、`calculator`、`web_search`（DuckDuckGo）、`url_fetch`、`weather`（Open-Meteo）。

```python
from nova_llm.tools import ToolRegistry, FunctionCallingLoop

reg = ToolRegistry()
@tool
def my_tool(query: str) -> str:
    """自定义工具"""
    return f"got: {query}"

loop = FunctionCallingLoop(generator, reg)
result = loop.run("现在几点？", max_iterations=5)
print(result["reply"], result["tool_calls"])
```

模型输出支持两种格式：

````
```json
{"name": "calculator", "arguments": {"expression": "1+2*3"}}
```
````

或标签格式：

```
<tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(16)"}}</tool_call>
```

## 配置说明

所有训练超参集中在 `configs/*.yaml`，关键字段：

```yaml
model:
  size: 1B              # 1B / 7B / 13B / 70B / 8x7B-MoE
  vocab_size: 32000
  max_position_embeddings: 4096
train:
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 1000
  max_steps: 100000
  bf16: true
```

## 工具脚本

| 脚本 | 作用 |
|------|------|
| `scripts/convert_checkpoint.py` | safetensors ↔ pytorch 权重互转 |
| `scripts/eval_perplexity.py` | 在文本文件上计算困惑度 |
| `scripts/benchmark_inference.py` | 测量 TTFT、每 token 延迟、吞吐 tokens/s |

```bash
python scripts/eval_perplexity.py --model checkpoints/sft-1b --data eval.txt
python scripts/benchmark_inference.py --model checkpoints/sft-1b --new-tokens 128
```

## 性能指标

基准脚本输出示例（CPU 小模型参考）：

```
[bench] avg TTFT = xx.x ms
[bench] avg throughput = xx.x tokens/s
```

完整 GPU 基准需在 CUDA 环境下开启 flash-attn 与 bf16 后运行 `benchmark_inference.py` 获取。

## License

MIT
