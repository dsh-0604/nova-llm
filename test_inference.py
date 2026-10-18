"""端到端验证：随机权重的迷你 Transformer + 假 tokenizer。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from nova_llm.inference import InferenceEngine, TextGenerator, Sampler, PagedKVCache, quantize_model
from nova_llm.inference.kv_cache import SimpleKVCache
from nova_llm.inference.quantize import dequantize

class MockConfig:
    def __init__(self):
        self.hidden_size = 64; self.num_hidden_layers = 2; self.num_attention_heads = 4
        self.num_key_value_heads = 2; self.head_dim = 16; self.vocab_size = 256; self.max_position_embeddings = 512

class MockOut:
    def __init__(self, logits, past_key_values): self.logits = logits; self.past_key_values = past_key_values

class MockLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, kh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.head_dim; self.n_heads, self.n_kv = h, kh
        self.q = nn.Linear(d, h*self.head_dim, bias=False); self.k = nn.Linear(d, kh*self.head_dim, bias=False)
        self.v = nn.Linear(d, kh*self.head_dim, bias=False); self.o = nn.Linear(h*self.head_dim, d, bias=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, x, past_k, past_v):
        B, S, _ = x.shape
        q = self.q(x).view(B,S,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k(x).view(B,S,self.n_kv,self.head_dim).transpose(1,2)
        v = self.v(x).view(B,S,self.n_kv,self.head_dim).transpose(1,2)
        if past_k is not None: k = torch.cat([past_k,k],dim=2); v = torch.cat([past_v,v],dim=2)
        rep = self.n_heads // self.n_kv
        krep = k.repeat_interleave(rep,dim=1); vrep = v.repeat_interleave(rep,dim=1)
        attn = F.softmax((q @ krep.transpose(-1,-2))/(self.head_dim**0.5), dim=-1)
        ctx = (attn @ vrep).transpose(1,2).reshape(B,S,-1)
        return self.norm(x+self.o(ctx)), k, v

class MockTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([MockLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    def forward(self, input_ids, past_key_values=None, use_cache=True):
        x = self.embed(input_ids); new_past = []
        for i, layer in enumerate(self.layers):
            pk = past_key_values[i][0] if past_key_values is not None else None
            pv = past_key_values[i][1] if past_key_values is not None else None
            x, k, v = layer(x, pk, pv); new_past.append((k,v))
        return MockOut(self.lm_head(x), new_past)

class MockTokenizer:
    eos_token_id = 0
    def encode(self, text): return [max(1,ord(c)%255) for c in text][:64] or [1]
    def decode(self, ids): return "".join(chr(32+(i%90)) for i in ids)

def test_sampler():
    print("\n[1] Sampler ..."); torch.manual_seed(0)
    logits = torch.randn(256); s = Sampler(temperature=0.8,top_k=50,top_p=0.9,repetition_penalty=1.1)
    toks = {s(logits, generated_tokens=list(range(0,5))) for _ in range(20)}
    assert 1 <= len(toks) <= 20; assert s.greedy_search(logits) == int(torch.argmax(logits))
    print(f"  采样 20 次得到 {len(toks)} 种 token, greedy={s.greedy_search(logits)}, OK")

def test_paged_kv():
    print("\n[2] PagedKVCache ..."); torch.manual_seed(0)
    cache = PagedKVCache(num_layers=2,num_kv_heads=2,head_dim=16,block_size=4,num_blocks=8,dtype=torch.float32,device="cpu")
    bt = cache.allocate(2); k_all = torch.randn(8,2,16); v_all = torch.randn(8,2,16)
    for layer in range(2): cache.append(layer, k_all, v_all, bt, 0)
    k_read, v_read = cache.get_kv(0, bt, 8)
    assert torch.allclose(k_read.permute(1,0,2), k_all, atol=1e-5)
    assert torch.allclose(v_read.permute(1,0,2), v_all, atol=1e-5)
    for bid in bt: cache.share(bid)
    assert cache._refcount[bt[0]] == 2
    bt2 = list(bt); cache.append(0, torch.randn(1,2,16), torch.randn(1,2,16), bt2, 0)
    assert cache._refcount[bt[0]] == 1; assert bt2[0] != bt[0]
    freed = cache.free(bt); cache.free(bt2)
    print(f"  roundtrip 一致, COW 触发正常, free 回收块数={freed}, OK")

def test_generator_stream():
    print("\n[3] TextGenerator 流式生成 ..."); torch.manual_seed(0)
    cfg = MockConfig(); model = MockTransformer(cfg); tok = MockTokenizer()
    gen = TextGenerator(model, tok, device="cpu", dtype=torch.float32, sampler=Sampler(temperature=1.0,top_k=0,top_p=1.0,repetition_penalty=1.0))
    chunks = list(gen.generate_stream("hello world", max_new_tokens=10))
    assert len(chunks) == 10; text = gen.generate("test", max_new_tokens=10); assert len(text) == 10
    batch = gen.generate_batch(["a","b"], max_new_tokens=5); assert len(batch) == 2
    print(f"  stream chunks={len(chunks)}, 文本='{text}', batch={len(batch)} OK")

def test_quantize():
    print("\n[4] 量化 int8 / int4 ..."); torch.manual_seed(0)
    cfg = MockConfig(); model = MockTransformer(cfg); x = torch.tensor([[1,2,3,4]])
    m8 = quantize_model(model, method="int8"); out8 = m8(x, past_key_values=None)
    assert out8.logits.shape == (1,4,cfg.vocab_size)
    from nova_llm.inference.quantize import QuantizedLinear
    assert any(isinstance(m, QuantizedLinear) for m in m8.modules())
    assert not isinstance(m8.embed, QuantizedLinear)
    m4 = quantize_model(MockTransformer(cfg), method="int4", group_size=32); out4 = m4(x, past_key_values=None)
    assert out4.logits.shape == (1,4,cfg.vocab_size)
    dequantize(m8); out8b = m8(x, past_key_values=None); assert out8b.logits.shape == (1,4,cfg.vocab_size)
    print(f"  int8 shape={tuple(out8.logits.shape)}, int4 shape={tuple(out4.logits.shape)}, OK")

def test_engine():
    print("\n[5] InferenceEngine 连续批处理 ..."); torch.manual_seed(0)
    cfg = MockConfig(); model = MockTransformer(cfg); tok = MockTokenizer()
    engine = InferenceEngine(model, tok, max_batch_size=4, max_seq_len=64, block_size=8, device="cpu")
    f1 = engine.generate("r1", "hello", max_new_tokens=6, priority=1)
    f2 = engine.generate("r2", "world", max_new_tokens=6, priority=0)
    results = engine.run_until_done(Sampler(temperature=1.0,top_k=0,top_p=1.0,repetition_penalty=1.0))
    assert set(results.keys()) == {"r1","r2"}; assert len(results["r1"]) == 6 and len(results["r2"]) == 6
    assert f1.done() and f2.done(); st = engine.stats(); print(f"  results keys={list(results.keys())}, 每请求 token=6"); print(f"  stats={st}"); assert st["tokens"] > 0; print("  OK")

if __name__ == "__main__":
    test_sampler(); test_paged_kv(); test_generator_stream(); test_quantize(); test_engine()
    print("\n全部测试通过 ✅")
