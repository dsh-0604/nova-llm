"""
04_function_calling.py — 工具调用示例
"""
import json
from nova_llm.tools import ToolRegistry, FunctionCallParser, FunctionCallingLoop
from nova_llm.tools.tool_registry import build_default_registry


class ScriptedGenerator:
    def __init__(self, script):
        self._script = list(script)
        self._i = 0
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, **kw):
        out = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return out


def main():
    registry = build_default_registry()
    script = [
        '我先查一下当前时间。\n```json\n{"name": "get_current_time", "arguments": {}}\n```',
        "根据查询结果，现在的时间就是您看到的系统时间。",
    ]
    gen = ScriptedGenerator(script)
    loop = FunctionCallingLoop(gen, registry)
    result = loop.run("现在几点了？", max_iterations=3)
    print("=== 工具调用历史 ===")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"[{i}] {tc['name']}({tc['arguments']})")
        print(f"    -> {tc['result']}")
    print("\n=== 最终回复 ===")
    print(result["reply"])


if __name__ == "__main__":
    main()
