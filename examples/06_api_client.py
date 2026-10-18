"""
06_api_client.py — API 客户端示例
"""
from openai import OpenAI


def main():
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    models = client.models.list()
    print("=== models ===")
    for m in models.data:
        print(" -", m.id)
    print("\n=== chat (non-stream) ===")
    resp = client.chat.completions.create(
        model="nova",
        messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
        max_tokens=64, temperature=0.7,
    )
    print(resp.choices[0].message.content)
    print("\n=== chat (stream) ===")
    stream = client.chat.completions.create(
        model="nova",
        messages=[{"role": "user", "content": "讲一个短笑话"}],
        max_tokens=64, stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    print("\n=== embeddings ===")
    emb = client.embeddings.create(model="nova", input=["hello", "world"])
    print("dim:", len(emb.data[0].embedding))


if __name__ == "__main__":
    main()
