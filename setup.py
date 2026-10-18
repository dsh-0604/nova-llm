from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="nova-llm",
    version="0.1.0",
    author="Nova AI",
    description="从零实现的大语言模型训练与推理系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples", "scripts"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "server": ["fastapi", "uvicorn", "pydantic"],
        "flash": ["flash-attn>=2.0"],
        "quantize": ["bitsandbytes"],
    },
    entry_points={
        "console_scripts": [
            "nova-train=nova_llm.train.pretrain:main",
            "nova-generate=nova_llm.inference.generator:main",
            "nova-serve=nova_llm.server.api_server:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
