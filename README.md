# transcription

Whisper (MLX) を使って音声ファイルを文字起こしするスクリプト。
Apple Silicon の GPU で動作する [mlx-whisper](https://pypi.org/project/mlx-whisper/)（モデルは OpenAI 公式の whisper-large-v3-turbo）を使用。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install mlx-whisper
```

ffmpeg が必要（`brew install ffmpeg`）。

## 使い方

```bash
.venv/bin/python transcribe.py 音声ファイル.m4a -l ja
```

- `-l` — 言語コード（省略時は自動判定）
- `-m` — 使用モデル（default: `mlx-community/whisper-large-v3-turbo`）
- `-o` — 出力ディレクトリ（default: 音声ファイルと同じ場所）

音声ファイルと同名の以下の 3 ファイルを出力する:

- `.txt` — 本文のみ（セグメントごとに改行）
- `.srt` — タイムスタンプ付き字幕形式
- `.tsv` — 開始秒・終了秒・テキストの表形式

音声・文字起こし結果は `.gitignore` で git 管理外にしている。
