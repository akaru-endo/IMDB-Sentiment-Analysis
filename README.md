# Transformer Sentiment Classifier

PyTorchを用いてTransformer (Encoder) をスクラッチ実装し、映画レビューデータセット（IMDB）に対する感情分析（Positive / Negative 分類）を行うプロジェクトです。

## 概要

ライブラリの組み込みモデル（`nn.TransformerEncoder` など）を使わずにMulti-Head Attention や Positional Encoding などの基本構造を PyTorch でイチから実装しています。
テキストの前処理およびトークナイズには Hugging Face の `bert-base-uncased` を使用しています。

## モデル構造

- Tokenizer: `bert-base-uncased`
- Embedding / Position Representation: Positional Encoding (Sinusoidal)
- Encoder: Multi-Head Attention + Feed Forward Network (Layer Normalization, Residual Connection)
- Classification Head: Mean Pooling + Linear Layer

## 評価結果

IMDB データセット（Trainデータ）での学習および評価結果：

- Training Accuracy: 99.40%

## ディレクトリ構成

.
├── transformer-learning3.ipynb # pythonコードのすべてのスクリプト
├── requirements.txt  # 依存ライブラリ一覧
└── README.md         # 本ファイル

## 使い方

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
