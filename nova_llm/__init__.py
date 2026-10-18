"""Nova LLM — 从零实现的大语言模型训练与推理系统"""
__version__ = "0.1.0"
__author__ = "Nova AI"
from nova_llm.model.config import NovaConfig, MODEL_CONFIGS
try:
    from nova_llm.model.transformer import NovaTransformer
except ImportError:
    NovaTransformer = None
try:
    from nova_llm.tokenizer.tokenizer import NovaTokenizer
except ImportError:
    NovaTokenizer = None
__all__ = ["NovaConfig", "MODEL_CONFIGS", "NovaTransformer", "NovaTokenizer"]
