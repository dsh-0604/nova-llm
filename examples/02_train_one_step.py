"""
02_train_one_step.py — 训练一步示例
"""
import torch
from nova_llm.model import NovaConfig, NovaTransformer


def make_dummy_batch(vocab_size, batch, seq):
    return torch.randint(0, vocab_size, (batch, seq))


def main():
    torch.manual_seed(0)
    cfg = NovaConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=8,
        num_key_value_heads=2, intermediate_size=512, vocab_size=1000,
        max_position_embeddings=128,
    )
    model = NovaTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    for step in range(5):
        x = make_dummy_batch(cfg.vocab_size, batch=4, seq=32)
        output = model(x, labels=x)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"[train] step {step+1}: loss = {loss.item():.4f}")
    print("[train] 一步训练流程演示完成")


if __name__ == "__main__":
    main()
