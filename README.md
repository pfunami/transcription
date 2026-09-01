# transcription

Whisper (MLX) を使って音声ファイルを文字起こしするスクリプト。
Apple Silicon の GPU で動作する [mlx-whisper](https://pypi.org/project/mlx-whisper/)（モデルは OpenAI 公式の whisper-large-v3-turbo）を使用。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install mlx-whisper
```

ffmpeg が必要（`brew install ffmpeg`）。

## ディレクトリ構成

```
audio/                  # 入力音声を置く (m4a, mp3, wav など)
output/
  <音声ファイル名>/     # 音声ファイルごとに結果を出力
    <音声ファイル名>.txt
    <音声ファイル名>.srt
    <音声ファイル名>.tsv
```

## 使い方

音声ファイルを `audio/` に置いてから:

```bash
# 特定のファイルを処理
.venv/bin/python transcribe.py 議論_20260901.m4a -l ja

# audio/ 内の全音声を処理
.venv/bin/python transcribe.py -l ja
```

- `-l` — 言語コード（省略時は自動判定）
- `-m` — 使用モデル（default: `mlx-community/whisper-large-v3-turbo`）

出力は `output/<音声ファイル名>/` に 3 形式で書き出される:

- `.txt` — 本文のみ（セグメントごとに改行）
- `.srt` — タイムスタンプ付き字幕形式
- `.tsv` — 開始秒・終了秒・テキストの表形式

`audio/` と `output/` の中身は議論の内容を含むため `.gitignore` で git 管理外にしている。
