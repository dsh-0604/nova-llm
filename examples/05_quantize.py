"""
05_quantize.py — 量化示例
"""
import time
import torch
from nova_llm.model import NovaConfig, NovaTransformer


def main():
    cfg = NovaConfig(hidden_size=256, num_hidden_layers=2,
                     num_attention_heads=8, num_key_value_heads=2,
                     intermediate_size=512, vocab_size=1000,
                     max_position_embeddings=128)
    model_fp32 = NovaTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        t0 = time.time()
        for _ in range(5):
            _ = model_fp32(x)
        fp32_time = (time.time() - t0) / 5
    fp32_mem = sum(p.numel() * 4 for p in model_fp32.parameters()) / 1e6
    try:
        model_int8 = torch.ao.quantization.quantize_dynamic(
            NovaTransformer(cfg), {torch.nn.Linear}, dtype=torch.qint8
        ).eval()
        with torch.no_grad():
            t0 = time.time()
            for _ in range(5):
                _ = model_int8(x)
            int8_time = (time.time() - t0) / 5
        int8_mem = fp32_mem / 2
        print(f"[quantize] fp32: {fp32_time*1000:.2f} ms/forward, ~{fp32_mem:.2f} MB")
        print(f"[quantize] int8: {int8_time*1000:.2f} ms/forward, ~{int8_mem:.2f} MB")
        print(f"[quantize] 加速比: {fp32_time/int8_time:.2f}x, 显存节省约 50%")
    except Exception as e:
        print(f"[quantize] int8 量化在当前环境不可用: {e}")


if __name__ == "__main__":
    main()
