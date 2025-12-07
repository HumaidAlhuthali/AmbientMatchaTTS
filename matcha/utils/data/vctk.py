#!/usr/bin/env python
"""
Script to download and prepare the VCTK dataset for Matcha-TTS training.

Downloads VCTK-Corpus-0.92, resamples audio from 48kHz to 22050Hz,
and creates multi-speaker format filelists (audio_path|speaker_id|text).
"""
import argparse
import random
import tempfile
import zipfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file
from tqdm import tqdm

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

INFO_PAGE = "https://datashare.ed.ac.uk/handle/10283/3443"

LICENCE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

CITATION = """
@inproceedings{yamagishi2019vctk,
  author    = {Yamagishi, Junichi and Veaux, Christophe and MacDonald, Kirsten},
  title     = {{CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)}},
  year      = {2019},
  publisher = {University of Edinburgh. The Centre for Speech Technology Research (CSTR)},
  doi       = {10.7488/ds/2645},
}
"""

TARGET_SAMPLE_RATE = 22050


def decision(val_ratio: float = 0.02):
    """Return True for training, False for validation."""
    return random.random() > val_ratio


def get_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare VCTK dataset for Matcha-TTS"
    )

    parser.add_argument(
        "-s", "--save-dir",
        type=str,
        default=None,
        help="Directory to store the downloaded zip file (default: use temp file)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data",
        help="Directory to store the processed data (default: data)"
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip resampling (use if you want to keep original 48kHz audio)"
    )
    parser.add_argument(
        "--filelist-only",
        action="store_true",
        help="Only generate filelists from existing data (skip download)"
    )

    return parser.parse_args()


def extract_zip(zip_path: str, output_dir: Path):
    """Extract zip file to output directory."""
    print(f"Extracting {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    print("Extraction complete.")


def get_speaker_id_mapping(speakers: list[str]) -> dict[str, int]:
    """Create a mapping from speaker names (p225, p226, ...) to integer IDs."""
    # Sort speakers to ensure consistent mapping
    sorted_speakers = sorted(speakers)
    return {spk: idx for idx, spk in enumerate(sorted_speakers)}


def resample_audio(input_path: Path, output_path: Path, target_sr: int = TARGET_SAMPLE_RATE):
    """Resample audio file to target sample rate."""
    audio, sr = torchaudio.load(input_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, target_sr)


def process_vctk(vctk_path: Path, resample: bool = True):
    """Process VCTK dataset: resample audio and create filelists."""
    # Find VCTK root (handle nested extraction)
    if (vctk_path / "VCTK-Corpus-0.92").exists():
        vctk_root = vctk_path / "VCTK-Corpus-0.92"
    elif (vctk_path / "wav48_silence_trimmed").exists():
        vctk_root = vctk_path
    else:
        raise FileNotFoundError(
            f"Could not find VCTK data in {vctk_path}. "
            "Expected 'wav48_silence_trimmed' directory."
        )

    wav48_dir = vctk_root / "wav48_silence_trimmed"
    txt_dir = vctk_root / "txt"
    
    if resample:
        wav_out_dir = vctk_root / "wavs_22050"
    else:
        wav_out_dir = wav48_dir

    # Get all speakers
    speakers = [d.name for d in wav48_dir.iterdir() if d.is_dir()]
    speaker_mapping = get_speaker_id_mapping(speakers)
    
    print(f"Found {len(speakers)} speakers")
    print(f"Speaker ID mapping: {min(speaker_mapping.values())}-{max(speaker_mapping.values())}")

    # Process each speaker
    train_entries = []
    val_entries = []
    
    for speaker in tqdm(sorted(speakers), desc="Processing speakers"):
        speaker_id = speaker_mapping[speaker]
        speaker_wav_dir = wav48_dir / speaker
        speaker_txt_dir = txt_dir / speaker
        
        if resample:
            speaker_out_dir = wav_out_dir / speaker
            speaker_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all audio files for this speaker
        audio_files = list(speaker_wav_dir.glob("*.flac")) + list(speaker_wav_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Get corresponding text file
            # VCTK uses format like p225_001_mic1.flac, text is p225_001.txt
            base_name = audio_file.stem
            # Remove mic suffix if present (e.g., _mic1, _mic2)
            if "_mic" in base_name:
                text_base = base_name.rsplit("_mic", 1)[0]
            else:
                text_base = base_name
            
            txt_file = speaker_txt_dir / f"{text_base}.txt"
            
            if not txt_file.exists():
                continue
            
            # Read transcription
            with open(txt_file, encoding="utf-8") as f:
                text = f.read().strip()
            
            if not text:
                continue
            
            # Resample if needed
            if resample:
                out_audio = speaker_out_dir / f"{base_name}.wav"
                if not out_audio.exists():
                    try:
                        resample_audio(audio_file, out_audio)
                    except Exception as e:
                        print(f"Warning: Failed to process {audio_file}: {e}")
                        continue
                audio_path = out_audio
            else:
                audio_path = audio_file
            
            # Create filelist entry
            entry = (str(audio_path), speaker_id, text)
            
            if decision():
                train_entries.append(entry)
            else:
                val_entries.append(entry)
    
    # Write filelists
    filelist_dir = vctk_path / "filelists"
    filelist_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = filelist_dir / "vctk_audio_sid_text_train_filelist.txt"
    val_file = filelist_dir / "vctk_audio_sid_text_val_filelist.txt"
    
    print(f"\nWriting train filelist: {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for audio_path, speaker_id, text in train_entries:
            f.write(f"{audio_path}|{speaker_id}|{text}\n")
    
    print(f"Writing val filelist: {val_file}")
    with open(val_file, "w", encoding="utf-8") as f:
        for audio_path, speaker_id, text in val_entries:
            f.write(f"{audio_path}|{speaker_id}|{text}\n")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total speakers: {len(speakers)}")
    print(f"Training samples: {len(train_entries)}")
    print(f"Validation samples: {len(val_entries)}")
    print(f"\nFilelists written to: {filelist_dir}")
    print(f"\nDon't forget to update configs/data/vctk.yaml:")
    print(f"  train_filelist_path: {train_file}")
    print(f"  valid_filelist_path: {val_file}")
    print(f"  n_spks: {len(speakers)}")


def main():
    args = get_args()
    
    outpath = Path(args.output_dir)
    if not outpath.is_dir():
        outpath.mkdir(parents=True)
    
    if args.filelist_only:
        print("Generating filelists from existing data...")
        process_vctk(outpath, resample=not args.no_resample)
        return
    
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)
    
    if save_dir:
        zip_name = URL.rsplit("/", maxsplit=1)[-1]
        zip_file = save_dir / zip_name
        if not zip_file.exists():
            print(f"Downloading VCTK dataset to {zip_file}...")
            print("(This is about 11GB, may take a while)")
            download_url_to_file(URL, str(zip_file), progress=True)
        extract_zip(str(zip_file), outpath)
        process_vctk(outpath, resample=not args.no_resample)
    else:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as zf:
            print(f"Downloading VCTK dataset...")
            print("(This is about 11GB, may take a while)")
            download_url_to_file(URL, zf.name, progress=True)
            extract_zip(zf.name, outpath)
            process_vctk(outpath, resample=not args.no_resample)


if __name__ == "__main__":
    main()

