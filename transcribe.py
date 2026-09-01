#!/usr/bin/env python3
"""Whisper (MLX) で音声ファイルを文字起こしするスクリプト。

使い方:
    .venv/bin/python transcribe.py 音声ファイル [-m モデル] [-l 言語] [-o 出力ディレクトリ]

例:
    .venv/bin/python transcribe.py 議論_20260901.m4a -l ja

出力: 音声ファイルと同名の .txt(本文のみ), .srt(字幕形式),
      .tsv(開始秒・終了秒・テキスト) を出力ディレクトリに書き出す。
"""

import argparse
import sys
from pathlib import Path

import mlx_whisper

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def format_srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper (MLX) による文字起こし")
    parser.add_argument("audio", help="音声ファイルのパス (m4a, mp3, wav など)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"使用するモデル (default: {DEFAULT_MODEL})")
    parser.add_argument("-l", "--language", default=None,
                        help="言語コード (例: ja)。省略時は自動判定")
    parser.add_argument("-o", "--outdir", default=None,
                        help="出力ディレクトリ (default: 音声ファイルと同じ場所)")
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        sys.exit(f"エラー: ファイルが見つかりません: {audio}")

    outdir = Path(args.outdir) if args.outdir else audio.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"モデル: {args.model}")
    print(f"文字起こし中: {audio.name} ...", flush=True)

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=args.model,
        language=args.language,
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
        print(f"  {p}")


if __name__ == "__main__":
    main()
