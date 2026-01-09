#!/usr/bin/env python3
"""
Verify decoder calibration dataset format and integrity.
"""

import numpy as np
import argparse
from pathlib import Path
import sys


def check_file_exists(path, description):
    """Check if file exists and print result."""
    if path.exists():
        print(f"  ✅ {description}: {path.name}")
        return True
    else:
        print(f"  ❌ {description} not found: {path}")
        return False


def verify_npy_file(path, expected_shape, expected_dtype, description):
    """Verify NPY file format and content."""
    try:
        data = np.load(path)
        
        # Check shape
        if data.shape != expected_shape:
            print(f"  ❌ {description} shape mismatch:")
            print(f"     Expected: {expected_shape}")
            print(f"     Got:      {data.shape}")
            return False
        
        # Check dtype
        if data.dtype != expected_dtype:
            print(f"  ❌ {description} dtype mismatch:")
            print(f"     Expected: {expected_dtype}")
            print(f"     Got:      {data.dtype}")
            return False
        
        # Check for NaN/Inf
        if np.issubdtype(data.dtype, np.floating):
            if np.isnan(data).any():
                print(f"  ⚠️  {description} contains NaN values")
                return False
            if np.isinf(data).any():
                print(f"  ⚠️  {description} contains Inf values")
                return False
        
        print(f"  ✅ {description}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.4f}, {data.max():.4f}]")
        return True
        
    except Exception as e:
        print(f"  ❌ {description} load failed: {e}")
        return False


def verify_dataset_txt(dataset_path, output_dir):
    """Verify dataset.txt format."""
    if not dataset_path.exists():
        print(f"  ❌ dataset.txt not found: {dataset_path}")
        return False
    
    print(f"\n📄 Verifying dataset.txt: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("  ❌ dataset.txt is empty")
        return False
    
    print(f"  ℹ️  Found {len(lines)} entries")
    
    all_valid = True
    checked_samples = min(len(lines), 5)  # Check first 5 samples
    
    for i, line in enumerate(lines[:checked_samples]):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 2:
            print(f"  ❌ Line {i+1}: Invalid format (expected 2 paths)")
            all_valid = False
            continue
        
        tokens_path = Path(parts[0])
        audio_path = Path(parts[1])
        
        # Check if files exist
        if not tokens_path.exists():
            print(f"  ❌ Line {i+1}: tokens file not found: {tokens_path}")
            all_valid = False
        
        if not audio_path.exists():
            print(f"  ❌ Line {i+1}: audio file not found: {audio_path}")
            all_valid = False
        
        if i == 0:  # Print first entry as example
            print(f"  ℹ️  Sample entry:")
            print(f"     tokens: {tokens_path.name}")
            print(f"     audio:  {audio_path.name}")
    
    if checked_samples < len(lines):
        print(f"  ℹ️  (Checked first {checked_samples} of {len(lines)} entries)")
    
    if all_valid:
        print("  ✅ dataset.txt format valid")
    
    return all_valid


def verify_dataset(output_dir, seq_len=1000, hidden_dim=512):
    """Main verification function."""
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Decoder Calibration Dataset Verification")
    print("=" * 60)
    print(f"\nDataset directory: {output_dir.absolute()}")
    print(f"Expected shapes:")
    print(f"  tokens: (1, 12) INT64")
    print(f"  audio:  (1, {seq_len}, {hidden_dim}) FP16")
    print()
    
    # Check if directory exists
    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        return False
    
    # Find files
    tokens_files = sorted(output_dir.glob("tokens_*.npy"))
    audio_files = sorted(output_dir.glob("audio_*.npy"))
    dataset_file = output_dir / "dataset.txt"
    
    print("📁 File inventory:")
    print(f"  tokens_*.npy: {len(tokens_files)} files")
    print(f"  audio_*.npy:  {len(audio_files)} files")
    print(f"  dataset.txt:  {'✅ exists' if dataset_file.exists() else '❌ missing'}")
    print()
    
    if not tokens_files or not audio_files:
        print("❌ No data files found")
        return False
    
    if len(tokens_files) != len(audio_files):
        print(f"⚠️  Warning: Mismatch in file counts")
        print(f"   tokens: {len(tokens_files)}, audio: {len(audio_files)}")
    
    # Verify a few samples
    print("🔍 Verifying sample files:")
    samples_to_check = min(3, len(tokens_files))
    
    all_valid = True
    for i in range(samples_to_check):
        print(f"\n  Sample {i}:")
        
        # Verify tokens
        valid_tokens = verify_npy_file(
            tokens_files[i],
            (1, 12),
            np.int64,
            "tokens"
        )
        
        # Verify audio
        valid_audio = verify_npy_file(
            audio_files[i],
            (1, seq_len, hidden_dim),
            np.float16,
            "audio"
        )
        
        if not (valid_tokens and valid_audio):
            all_valid = False
    
    if samples_to_check < len(tokens_files):
        print(f"\n  ℹ️  (Checked {samples_to_check} of {len(tokens_files)} samples)")
    
    # Verify dataset.txt
    valid_dataset = verify_dataset_txt(dataset_file, output_dir)
    all_valid = all_valid and valid_dataset
    
    # Print summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ Dataset verification PASSED")
        print("\nYou can now use this dataset for quantization:")
        print(f"  python convert.py \\")
        print(f"      whisper_decoder.onnx \\")
        print(f"      rk3588 \\")
        print(f"      i8 \\")
        print(f"      whisper_decoder_int8.rknn")
        print(f"\n  Remember to update dataset path in convert.py:")
        print(f"    dataset='{dataset_file.absolute()}'")
    else:
        print("❌ Dataset verification FAILED")
        print("\nPlease fix the issues above before quantization.")
    print("=" * 60)
    
    return all_valid


def verify_single_pair(tokens_path, audio_path, seq_len=1000, hidden_dim=512):
    """Verify a single tokens/audio pair."""
    tokens_path = Path(tokens_path)
    audio_path = Path(audio_path)
    
    print(f"Verifying pair:")
    print(f"  tokens: {tokens_path}")
    print(f"  audio:  {audio_path}")
    print()
    
    valid_tokens = verify_npy_file(tokens_path, (1, 12), np.int64, "tokens")
    valid_audio = verify_npy_file(audio_path, (1, seq_len, hidden_dim), np.float16, "audio")
    
    if valid_tokens and valid_audio:
        print("\n✅ Pair is valid")
        
        # Show sample content
        tokens = np.load(tokens_path)
        audio = np.load(audio_path)
        
        print(f"\nSample content:")
        print(f"  tokens[0]: {tokens[0]}")
        print(f"  audio stats:")
        print(f"    mean:  {audio.mean():.6f}")
        print(f"    std:   {audio.std():.6f}")
        print(f"    min:   {audio.min():.6f}")
        print(f"    max:   {audio.max():.6f}")
        return True
    else:
        print("\n❌ Pair is invalid")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify decoder calibration dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify entire dataset
  python verify_decoder_dataset.py --output_dir ./decoder_calib
  
  # Verify with custom dimensions
  python verify_decoder_dataset.py --output_dir ./decoder_calib --seq_len 1500 --hidden_dim 512
  
  # Verify single pair
  python verify_decoder_dataset.py --tokens tokens_000000.npy --audio audio_000000.npy
        """
    )
    
    parser.add_argument('--output_dir', type=str,
                        help='Decoder calibration dataset directory')
    parser.add_argument('--tokens', type=str,
                        help='Single tokens.npy file to verify')
    parser.add_argument('--audio', type=str,
                        help='Single audio.npy file to verify')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='Expected sequence length (1000 for 20s, 1500 for 30s)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Expected hidden dimension (512 for tiny, 768 for base)')
    
    args = parser.parse_args()
    
    # Verify mode: dataset or single pair
    if args.output_dir:
        success = verify_dataset(args.output_dir, args.seq_len, args.hidden_dim)
    elif args.tokens and args.audio:
        success = verify_single_pair(args.tokens, args.audio, args.seq_len, args.hidden_dim)
    else:
        print("Error: Must specify either --output_dir or both --tokens and --audio")
        parser.print_help()
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
