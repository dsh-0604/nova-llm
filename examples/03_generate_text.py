"""
03_generate_text.py — 文本生成示例
"""
import argparse
from nova_llm.model import NovaConfig, NovaTransformer
from nova_llm.inference import TextGenerator
from nova_llm.tokenizer import NovaTokenizer, BPETrainer


def _make_demo_tokenizer(vocab_size=300):
    corpus = [
        "你好世界", "这是一个大模型", "人工智能正在改变世界",
        "深度学习神经网络", "自然语言处理", "机器学习算法",
        "hello world", "machine learning", "deep learning",
        "transformer architecture", "attention is all you need",
    ] * 50
    trainer = BPETrainer(vocab_size=vocab_size).train(corpus, verbose=False)
    return NovaTokenizer(merges=trainer.merges, vocab=trainer.vocab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default="你好，请介绍一下你自己。")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.model:
        cfg = NovaConfig.from_json(f"{args.model}/config.json")
        model = NovaTransformer(cfg)
        import torch
        from safetensors.torch import load_file
        model.load_state_dict(load_file(f"{args.model}/model.safetensors"), strict=False)
        tokenizer = NovaTokenizer(args.model)
    else:
        print("[generate] 未提供 --model，使用随机权重小模型演示接口")
        cfg = NovaConfig(hidden_size=256, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2,
                         intermediate_size=512, vocab_size=1000,
                         max_position_embeddings=128)
        model = NovaTransformer(cfg)
        tokenizer = _make_demo_tokenizer(vocab_size=cfg.vocab_size)
    model.eval()
    gen = TextGenerator(model, tokenizer, device=args.device)
    print(f"[generate] prompt: {args.prompt}")
    print("[generate] streaming:")
    for piece in gen.generate_stream(args.prompt, max_new_tokens=args.max_new_tokens):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
