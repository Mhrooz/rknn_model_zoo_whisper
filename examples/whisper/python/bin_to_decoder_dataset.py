#!/usr/bin/env python3
"""
Convert binary encoder outputs from C++ inference to decoder calibration dataset.

This script reads the .bin files dumped by the modified whisper inference code
and converts them to .npy format suitable for decoder quantization.

The C++ code dumps:
  - enc_XXXXXX.bin: encoder output in FP32 [1, 1000, 512]

This script generates:
  - tokens_XXXXXX.npy: fixed tokens [1, 12] INT64
  - audio_XXXXXX.npy: encoder output [1, 1000, 512] FP16
  - dataset.txt: list of token/audio pairs for RKNN quantization
"""

import numpy as np
import argparse
from pathlib import Path
import sys


def float32_to_float16_via_numpy(data_f32):
    """Convert float32 numpy array to float16."""
    return data_f32.astype(np.float16)


def read_encoder_bin(bin_path, seq_len=1000, hidden_dim=512):
    """
    Read encoder output from binary file.
    Expected format: FP32 array of shape [seq_len * hidden_dim]
    
    Args:
        bin_path: Path to .bin file
        seq_len: Sequence length (default 1000 for 20s audio, 1500 for 30s)
        hidden_dim: Hidden dimension (512 for whisper-tiny)
    
    Returns:
        numpy array of shape [1, seq_len, hidden_dim] in FP32
    """
    data = np.fromfile(bin_path, dtype=np.float32)
    expected_size = seq_len * hidden_dim
    
    if data.size != expected_size:
        print(f"Warning: {bin_path} has size {data.size}, expected {expected_size}")
        return None
    
    # Reshape to [1, seq_len, hidden_dim]
    data = data.reshape(1, seq_len, hidden_dim)
    return data


def create_fixed_tokens():
    """
    Create fixed decoder input tokens.
    
    Whisper decoder expects a prompt sequence. For calibration, we use
    a typical start sequence: [sot, task, notimestamps, ...]
    
    Returns:
        numpy array of shape [1, 12] in INT64
    """
    # Typical whisper start tokens (adjust based on your tokenizer)
    # 50258: sot (start of transcript)
    # 50259: en (English task code, can also be 50260 for Chinese)
    # 50359: notimestamps
    # Rest are padding or continuation tokens
    tokens = np.array([
        [50258, 50259, 50359, 50363,
         50364, 50365, 50366, 50367,
         50368, 50369, 50370, 50371]
    ], dtype=np.int64)
    
    return tokens


def process_bin_files(bin_dir, output_dir, seq_len=1000, hidden_dim=512, max_samples=None):
    """
    Process all enc_*.bin files in bin_dir and generate decoder calibration dataset.
    
    Args:
        bin_dir: Directory containing enc_XXXXXX.bin files
        output_dir: Directory to save output files
        seq_len: Encoder output sequence length
        hidden_dim: Hidden dimension
        max_samples: Maximum number of samples to process (None = all)
    """
    bin_dir = Path(bin_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all encoder output bin files
    enc_bins = sorted(bin_dir.glob("enc_*.bin"))
    
    if not enc_bins:
        print(f"Error: No enc_*.bin files found in {bin_dir}")
        return
    
    print(f"Found {len(enc_bins)} encoder output files")
    
    if max_samples is not None and len(enc_bins) > max_samples:
        enc_bins = enc_bins[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # Open dataset.txt for writing
    dataset_file = output_dir / "dataset.txt"
    with open(dataset_file, 'w') as f_dataset:
        kept = 0
        
        for i, enc_bin in enumerate(enc_bins):
            # Read encoder output
            encoder_output = read_encoder_bin(enc_bin, seq_len, hidden_dim)
            
            if encoder_output is None:
                print(f"Skipping {enc_bin.name} (invalid size)")
                continue
            
            # Convert to FP16
            encoder_output_fp16 = float32_to_float16_via_numpy(encoder_output)
            
            # Create tokens
            tokens = create_fixed_tokens()
            
            # Save tokens.npy
            tokens_path = output_dir / f"tokens_{kept:06d}.npy"
            np.save(tokens_path, tokens)
            
            # Save audio.npy (encoder output in FP16)
            audio_path = output_dir / f"audio_{kept:06d}.npy"
            np.save(audio_path, encoder_output_fp16)
            
            # Write to dataset.txt (absolute paths)
            f_dataset.write(f"{tokens_path.absolute()} {audio_path.absolute()}\n")
            
            kept += 1
            
            if kept % 50 == 0:
                print(f"Processed {kept} samples...")
    
    print(f"\nDone! Generated {kept} decoder calibration samples")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Dataset file: {dataset_file.absolute()}")
    print(f"\nGenerated files:")
    print(f"  - tokens_*.npy: shape [1, 12], dtype int64")
    print(f"  - audio_*.npy:  shape [1, {seq_len}, {hidden_dim}], dtype float16")
    print(f"  - dataset.txt:  list of 'tokens.npy audio.npy' pairs")


def verify_dataset(output_dir):
    """Verify the generated dataset by loading a sample."""
    output_dir = Path(output_dir)
    
    # Check if files exist
    tokens_files = list(output_dir.glob("tokens_*.npy"))
    audio_files = list(output_dir.glob("audio_*.npy"))
    dataset_file = output_dir / "dataset.txt"
    
    if not tokens_files or not audio_files or not dataset_file.exists():
        print("Error: Dataset files not found")
        return False
    
    # Load first sample
    tokens = np.load(tokens_files[0])
    audio = np.load(audio_files[0])
    
    print("\n=== Dataset Verification ===")
    print(f"tokens shape: {tokens.shape}, dtype: {tokens.dtype}")
    print(f"audio shape:  {audio.shape}, dtype: {audio.dtype}")
    print(f"tokens sample: {tokens[0, :4]}...")
    print(f"audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    # Check dataset.txt format
    with open(dataset_file, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) == 2:
            print(f"\ndataset.txt format: OK")
            print(f"  Example line: {Path(parts[0]).name} {Path(parts[1]).name}")
        else:
            print(f"\nWarning: dataset.txt format may be incorrect")
            print(f"  Expected: 'tokens.npy audio.npy'")
            print(f"  Got: {first_line}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert encoder output bins to decoder calibration dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all enc_*.bin files
  python bin_to_decoder_dataset.py --bin_dir /path/to/dumps --output_dir ./decoder_calib
  
  # Process with custom sequence length (for 30s audio)
  python bin_to_decoder_dataset.py --bin_dir ./dumps --output_dir ./decoder_calib --seq_len 1500
  
  # Process only first 100 samples
  python bin_to_decoder_dataset.py --bin_dir ./dumps --output_dir ./decoder_calib --max_samples 100
        """
    )
    
    parser.add_argument('--bin_dir', type=str, required=True,
                        help='Directory containing enc_*.bin files from C++ inference')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for decoder calibration dataset')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='Encoder output sequence length (1000 for 20s, 1500 for 30s)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension (512 for tiny, 768 for base, 1024 for small)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated dataset after processing')
    
    args = parser.parse_args()
    
    # Process files
    process_bin_files(
        args.bin_dir,
        args.output_dir,
        args.seq_len,
        args.hidden_dim,
        args.max_samples
    )
    
    # Verify if requested
    if args.verify:
        verify_dataset(args.output_dir)


if __name__ == '__main__':
    main()
