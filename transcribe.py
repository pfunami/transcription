#!/usr/bin/env python3
"""Whisper (MLX) で音声ファイルを文字起こしするスクリプト。

audio/ ディレクトリから音声を読み込み、output/<音声ファイル名>/ に結果を書き出す。

使い方:
    .venv/bin/python transcribe.py [音声ファイル名...] [-m モデル] [-l 言語]

例:
    .venv/bin/python transcribe.py 議論_20260901.m4a -l ja   # audio/議論_20260901.m4a を処理
    .venv/bin/python transcribe.py -l ja                      # audio/ 内の全音声を処理

出力: output/<音声ファイル名>/ に .txt(本文のみ), .srt(字幕形式),
      .tsv(開始秒・終了秒・テキスト) を書き出す。
"""

import argparse
import sys
from pathlib import Path

import mlx_whisper

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".mp4", ".webm"}


def format_srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def resolve_audio(name: str) -> Path:
    """引数をそのままのパス → audio/ 内の順で探す。"""
    path = Path(name)
    if path.exists():
        return path
    candidate = AUDIO_DIR / name
    if candidate.exists():
        return candidate
    sys.exit(f"エラー: ファイルが見つかりません: {name} (audio/ 内も確認しました)")


def transcribe_one(audio: Path, model: str, language: str | None) -> None:
    outdir = OUTPUT_DIR / audio.stem
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"文字起こし中: {audio.name} ...", flush=True)

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model,
        language=language,
        verbose=False,
    )

    stem = outdir / audio.stem

    # .txt: 本文のみ(セグメントごとに改行)
    txt_path = stem.with_suffix(".txt")
    txt_path.write_text(
        "\n".join(seg["text"].strip() for seg in result["segments"]) + "\n",
        encoding="utf-8",
    )

    # .srt: 字幕形式
    srt_lines = []
    for i, seg in enumerate(result["segments"], start=1):
        srt_lines.append(
            f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n"
            f"{seg['text'].strip()}\n"
        )
    srt_path = stem.with_suffix(".srt")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # .tsv: 開始秒・終了秒・テキスト
    tsv_path = stem.with_suffix(".tsv")
    tsv_path.write_text(
        "start\tend\ttext\n"
        + "".join(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{seg['text'].strip()}\n"
                  for seg in result["segments"]),
        encoding="utf-8",
    )

    print(f"検出言語: {result.get('language', '不明')}")
    print("出力:")
    for p in (txt_path, srt_path, tsv_path):
        print(f"  {p.relative_to(BASE_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper (MLX) による文字起こし")
    parser.add_argument("audio", nargs="*",
                        help="音声ファイル名 (audio/ 内のファイル名またはパス)。"
                             "省略時は audio/ 内の全音声を処理")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"使用するモデル (default: {DEFAULT_MODEL})")
    parser.add_argument("-l", "--language", default=None,
                        help="言語コード (例: ja)。省略時は自動判定")
    args = parser.parse_args()

    if args.audio:
        targets = [resolve_audio(name) for name in args.audio]
    else:
        targets = sorted(
            p for p in AUDIO_DIR.iterdir()
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not targets:
            sys.exit(f"エラー: {AUDIO_DIR} に音声ファイルがありません")

    print(f"モデル: {args.model}")
    for audio in targets:
        transcribe_one(audio, args.model, args.language)


if __name__ == "__main__":
    main()
