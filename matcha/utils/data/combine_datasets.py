#!/usr/bin/env python
"""
Script to combine LJSpeech and VCTK datasets into a single multi-speaker dataset.

LJSpeech (single-speaker) format: audio_path|text
VCTK (multi-speaker) format: audio_path|speaker_id|text

This script converts LJSpeech entries to multi-speaker format by assigning
speaker ID 109, and combines with VCTK (speakers 0-108) for a total of 110 speakers.
"""
import argparse
from pathlib import Path


def parse_ljspeech_filelist(filepath: Path) -> list[tuple[str, str]]:
    """Parse LJSpeech format filelist (audio_path|text)."""
    entries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                audio_path = parts[0]
                text = parts[1]
                entries.append((audio_path, text))
    return entries


def parse_vctk_filelist(filepath: Path) -> list[tuple[str, int, str]]:
    """Parse VCTK format filelist (audio_path|speaker_id|text)."""
    entries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                audio_path = parts[0]
                speaker_id = int(parts[1])
                text = parts[2]
                entries.append((audio_path, speaker_id, text))
    return entries


def convert_ljspeech_to_multispeaker(
    entries: list[tuple[str, str]], speaker_id: int = 109
) -> list[tuple[str, int, str]]:
    """Convert LJSpeech entries to multi-speaker format with given speaker ID."""
    return [(audio_path, speaker_id, text) for audio_path, text in entries]


def write_combined_filelist(
    filepath: Path, entries: list[tuple[str, int, str]]
) -> None:
    """Write combined filelist in multi-speaker format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for audio_path, speaker_id, text in entries:
            f.write(f"{audio_path}|{speaker_id}|{text}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Combine LJSpeech and VCTK datasets into a single multi-speaker dataset"
    )

    parser.add_argument(
        "--ljspeech-train",
        type=str,
        default="data/LJSpeech-1.1/train.txt",
        help="Path to LJSpeech training filelist",
    )
    parser.add_argument(
        "--ljspeech-val",
        type=str,
        default="data/LJSpeech-1.1/val.txt",
        help="Path to LJSpeech validation filelist",
    )
    parser.add_argument(
        "--vctk-train",
        type=str,
        default="data/filelists/vctk_audio_sid_text_train_filelist.txt",
        help="Path to VCTK training filelist",
    )
    parser.add_argument(
        "--vctk-val",
        type=str,
        default="data/filelists/vctk_audio_sid_text_val_filelist.txt",
        help="Path to VCTK validation filelist",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/filelists",
        help="Output directory for combined filelists",
    )
    parser.add_argument(
        "--ljspeech-speaker-id",
        type=int,
        default=109,
        help="Speaker ID to assign to LJSpeech (default: 109)",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ljspeech_train_path = Path(args.ljspeech_train)
    ljspeech_val_path = Path(args.ljspeech_val)
    vctk_train_path = Path(args.vctk_train)
    vctk_val_path = Path(args.vctk_val)
    output_dir = Path(args.output_dir)

    # Parse LJSpeech filelists
    print(f"Reading LJSpeech train filelist: {ljspeech_train_path}")
    ljspeech_train = parse_ljspeech_filelist(ljspeech_train_path)
    print(f"  Found {len(ljspeech_train)} entries")

    print(f"Reading LJSpeech val filelist: {ljspeech_val_path}")
    ljspeech_val = parse_ljspeech_filelist(ljspeech_val_path)
    print(f"  Found {len(ljspeech_val)} entries")

    # Parse VCTK filelists
    print(f"Reading VCTK train filelist: {vctk_train_path}")
    vctk_train = parse_vctk_filelist(vctk_train_path)
    print(f"  Found {len(vctk_train)} entries")

    print(f"Reading VCTK val filelist: {vctk_val_path}")
    vctk_val = parse_vctk_filelist(vctk_val_path)
    print(f"  Found {len(vctk_val)} entries")

    # Convert LJSpeech to multi-speaker format
    print(f"Assigning LJSpeech speaker ID: {args.ljspeech_speaker_id}")
    ljspeech_train_ms = convert_ljspeech_to_multispeaker(
        ljspeech_train, args.ljspeech_speaker_id
    )
    ljspeech_val_ms = convert_ljspeech_to_multispeaker(
        ljspeech_val, args.ljspeech_speaker_id
    )

    # Combine datasets
    combined_train = vctk_train + ljspeech_train_ms
    combined_val = vctk_val + ljspeech_val_ms

    print(f"\nCombined train set: {len(combined_train)} entries")
    print(f"Combined val set: {len(combined_val)} entries")

    # Write combined filelists
    train_output = output_dir / "combined_train.txt"
    val_output = output_dir / "combined_val.txt"

    print(f"\nWriting combined train filelist: {train_output}")
    write_combined_filelist(train_output, combined_train)

    print(f"Writing combined val filelist: {val_output}")
    write_combined_filelist(val_output, combined_val)

    # Print summary
    vctk_speakers = set(entry[1] for entry in vctk_train + vctk_val)
    max_speaker_id = max(max(vctk_speakers), args.ljspeech_speaker_id)
    n_spks = max_speaker_id + 1

    print(f"\n=== Summary ===")
    print(f"VCTK speakers: {len(vctk_speakers)} (IDs: {min(vctk_speakers)}-{max(vctk_speakers)})")
    print(f"LJSpeech speaker ID: {args.ljspeech_speaker_id}")
    print(f"Total unique speakers: {len(vctk_speakers) + 1}")
    print(f"\nDon't forget to:")
    print(f"  1. Update n_spks in configs/data/combined.yaml to {n_spks}")
    print(f"  2. Run: matcha-data-stats -i combined.yaml")
    print(f"  3. Update data_statistics in configs/data/combined.yaml")


if __name__ == "__main__":
    main()

