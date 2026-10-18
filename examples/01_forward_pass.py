"""
01_forward_pass.py — 前向传播示例
创建模型 -> 随机输入 -> 输出 logits -> 计算 loss -> 反向传播
"""
import torch

from nova_llm.model import NovaConfig, NovaTransformer


def main():
    cfg = NovaConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=8,
        num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
        max_position_embeddings=128,
    )
    print(f"[forward] 参数量约: {cfg.count_parameters()/1e6:.2f} M")
    model = NovaTransformer(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    print(f"[forward] input shape: {tuple(idx.shape)}")
    with torch.no_grad():
        output = model(idx)
    print(f"[forward] logits shape: {tuple(output.logits.shape)}")
    model.train()
    output = model(idx, labels=idx)
    loss = output.loss
    print(f"[forward] loss = {loss.item():.4f}")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"[forward] 反向传播完成，梯度范数 = {grad_norm:.4f}")


if __name__ == "__main__":
    main()
