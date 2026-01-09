#!/bin/bash
# Batch process audio files to generate encoder outputs for decoder quantization

set -e

# Configuration
LIBRISPEECH_DIR="${1:-../../datasets/Librispeech/dev-clean}"
DUMP_DIR="${2:-/tmp/whisper_encoder_dumps}"
ENCODER_MODEL="${3:-./model/whisper_encoder.rknn}"
DECODER_MODEL="${4:-./model/whisper_decoder.rknn}"
TASK="${5:-en}"
MAX_FILES="${6:-500}"

# Check if whisper demo exists
if [ ! -f "./build/rknn_whisper_demo" ]; then
    echo "Error: rknn_whisper_demo not found. Please build first:"
    echo "  cd build && cmake .. && make"
    exit 1
fi

# Check if models exist
if [ ! -f "$ENCODER_MODEL" ]; then
    echo "Error: Encoder model not found: $ENCODER_MODEL"
    exit 1
fi

if [ ! -f "$DECODER_MODEL" ]; then
    echo "Error: Decoder model not found: $DECODER_MODEL"
    exit 1
fi

# Create dump directory
mkdir -p "$DUMP_DIR"

echo "============================================"
echo "Whisper Encoder Output Batch Generation"
echo "============================================"
echo "Librispeech directory: $LIBRISPEECH_DIR"
echo "Output directory:      $DUMP_DIR"
echo "Encoder model:         $ENCODER_MODEL"
echo "Decoder model:         $DECODER_MODEL"
echo "Task:                  $TASK"
echo "Max files:             $MAX_FILES"
echo "============================================"
echo ""

# Important: Make sure whisper.cc has the correct dump_dir
echo "⚠️  IMPORTANT: Make sure you have modified dump_dir in whisper.cc to:"
echo "    const std::string dump_dir = \"$DUMP_DIR\";"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Find all FLAC files
echo "Searching for FLAC files..."
flac_files=($(find "$LIBRISPEECH_DIR" -name "*.flac" | head -n "$MAX_FILES"))
total_files=${#flac_files[@]}

if [ $total_files -eq 0 ]; then
    echo "Error: No FLAC files found in $LIBRISPEECH_DIR"
    exit 1
fi

echo "Found $total_files FLAC files"
echo ""

# Process each file
count=0
success=0
failed=0

for audio_file in "${flac_files[@]}"; do
    count=$((count + 1))
    printf "[%3d/%3d] Processing: %s\n" "$count" "$total_files" "$(basename "$audio_file")"
    
    if ./build/rknn_whisper_demo "$ENCODER_MODEL" "$DECODER_MODEL" "$TASK" "$audio_file" > /dev/null 2>&1; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
        echo "  ⚠️  Failed to process file"
    fi
    
    # Show progress every 10 files
    if [ $((count % 10)) -eq 0 ]; then
        echo "  Progress: $count/$total_files files processed ($success success, $failed failed)"
    fi
done

echo ""
echo "============================================"
echo "Batch processing complete!"
echo "============================================"
echo "Total files:     $total_files"
echo "Successful:      $success"
echo "Failed:          $failed"
echo "Output directory: $DUMP_DIR"
echo ""

# Check generated files
enc_files=$(ls -1 "$DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l)
echo "Generated encoder output files: $enc_files"

if [ $enc_files -gt 0 ]; then
    echo ""
    echo "✅ Next step: Convert bin files to decoder calibration dataset"
    echo ""
    echo "Run:"
    echo "  cd ../python"
    echo "  python bin_to_decoder_dataset.py \\"
    echo "      --bin_dir $DUMP_DIR \\"
    echo "      --output_dir ./decoder_calib \\"
    echo "      --seq_len 1000 \\"
    echo "      --hidden_dim 512 \\"
    echo "      --max_samples 500 \\"
    echo "      --verify"
else
    echo ""
    echo "⚠️  Warning: No encoder output files generated!"
    echo "   Please check if dump_dir in whisper.cc is set correctly."
fi
