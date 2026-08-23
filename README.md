# BERT Word Similarity

基于经典 BERT 模型预训练好的 input embedding table，判断一个新单词与两个候选单词中的哪一个更接近。

## 实现思路

1. 使用 `bert-base-uncased` 的 WordPiece tokenizer 将单词转换为 token。
2. 直接查询 BERT 预训练的 input embedding table，不运行 Transformer 编码层。
3. 如果单词被拆成多个 WordPiece，取这些 token embedding 的平均值作为词向量。
4. 计算新单词与两个候选词向量的余弦相似度，输出分数更高的候选词。

## 项目结构

```text
.
├── app.py                 # 交互式命令行入口
├── word_similarity.py     # BERT embedding 与相似度逻辑
├── requirements.txt       # Python 依赖
└── README.md              # 使用说明
```

## 安装与运行

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

首次运行会从 Hugging Face 下载 `bert-base-uncased`，之后会使用本地缓存。

如果服务器无法直接访问 Hugging Face，可在首次运行时使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python app.py
```

运行示例：

```text
候选单词 1: cat
候选单词 2: car
新单词: dog
最接近的候选单词: cat
相似度: cat=..., car=...
```

在任意输入处输入 `q` 即可退出。

## 服务器部署

```bash
git clone <GitHub 仓库地址>
cd BERT-Word-Similarity
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

本项目当前部署目录为：

```text
/root/autodl-tmp/BERT-Word-Similarity
```
