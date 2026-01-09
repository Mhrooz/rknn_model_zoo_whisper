#!/bin/bash
# End-to-end script for generating decoder calibration dataset and quantizing decoder

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Default configuration
LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-$PROJECT_ROOT/datasets/Librispeech/dev-clean}"
DUMP_DIR="${DUMP_DIR:-/tmp/whisper_encoder_dumps}"
CALIB_DIR="${CALIB_DIR:-$SCRIPT_DIR/decoder_calib}"
ENCODER_MODEL="${ENCODER_MODEL:-$SCRIPT_DIR/model/whisper_encoder.rknn}"
DECODER_ONNX="${DECODER_ONNX:-$SCRIPT_DIR/model/whisper_decoder.onnx}"
DECODER_MODEL_OUT="${DECODER_MODEL_OUT:-$SCRIPT_DIR/model/whisper_decoder_int8.rknn}"
TASK="${TASK:-en}"
PLATFORM="${PLATFORM:-rk3588}"
SEQ_LEN="${SEQ_LEN:-1000}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

function print_banner() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

function print_config() {
    print_banner "Configuration"
    echo "Librispeech directory:  $LIBRISPEECH_DIR"
    echo "Encoder dumps:          $DUMP_DIR"
    echo "Calibration dataset:    $CALIB_DIR"
    echo "Encoder model:          $ENCODER_MODEL"
    echo "Decoder ONNX:           $DECODER_ONNX"
    echo "Decoder RKNN output:    $DECODER_MODEL_OUT"
    echo "Task:                   $TASK"
    echo "Platform:               $PLATFORM"
    echo "Sequence length:        $SEQ_LEN"
    echo "Hidden dimension:       $HIDDEN_DIM"
    echo "Max samples:            $MAX_SAMPLES"
    echo ""
}

function check_requirements() {
    print_banner "Checking Requirements"
    
    # Check if build exists
    if [ ! -f "$SCRIPT_DIR/build/rknn_whisper_demo" ]; then
        echo "❌ rknn_whisper_demo not found"
        echo "   Building now..."
        mkdir -p "$SCRIPT_DIR/build"
        cd "$SCRIPT_DIR/build"
        cmake .. && make
        cd "$SCRIPT_DIR"
    else
        echo "✅ rknn_whisper_demo found"
    fi
    
    # Check if encoder model exists
    if [ ! -f "$ENCODER_MODEL" ]; then
        echo "❌ Encoder model not found: $ENCODER_MODEL"
        exit 1
    else
        echo "✅ Encoder model found"
    fi
    
    # Check if decoder ONNX exists
    if [ ! -f "$DECODER_ONNX" ]; then
        echo "❌ Decoder ONNX not found: $DECODER_ONNX"
        exit 1
    else
        echo "✅ Decoder ONNX found"
    fi
    
    # Check if librispeech exists
    if [ ! -d "$LIBRISPEECH_DIR" ]; then
        echo "❌ Librispeech directory not found: $LIBRISPEECH_DIR"
        exit 1
    else
        echo "✅ Librispeech directory found"
    fi
    
    # Check Python requirements
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo "❌ Python numpy not found"
        exit 1
    else
        echo "✅ Python numpy found"
    fi
    
    echo ""
}

function step1_check_dump_dir() {
    print_banner "Step 1: Check dump_dir in whisper.cc"
    
    if grep -q "const std::string dump_dir = \"$DUMP_DIR\"" "$SCRIPT_DIR/rknpu2/whisper.cc" 2>/dev/null; then
        echo "✅ dump_dir is correctly set to: $DUMP_DIR"
    else
        echo "⚠️  Warning: dump_dir in whisper.cc may not match"
        echo "   Expected: const std::string dump_dir = \"$DUMP_DIR\";"
        echo ""
        echo "   Please manually edit examples/whisper/cpp/rknpu2/whisper.cc"
        echo "   and set dump_dir to the correct path, then rebuild."
        echo ""
        read -p "   Have you set the correct dump_dir? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please fix dump_dir first."
            exit 1
        fi
    fi
}

function step2_generate_encoder_outputs() {
    print_banner "Step 2: Generate Encoder Outputs"
    
    mkdir -p "$DUMP_DIR"
    
    echo "Finding FLAC files..."
    flac_files=($(find "$LIBRISPEECH_DIR" -name "*.flac" | head -n "$MAX_SAMPLES"))
    total_files=${#flac_files[@]}
    
    if [ $total_files -eq 0 ]; then
        echo "❌ No FLAC files found"
        exit 1
    fi
    
    echo "Found $total_files FLAC files"
    echo "Processing..."
    
    count=0
    success=0
    
    # Use a temporary decoder model path (we only need encoder)
    TEMP_DECODER="${DECODER_MODEL_OUT%.rknn}_temp.rknn"
    if [ ! -f "$TEMP_DECODER" ]; then
        # If no decoder exists yet, use encoder as placeholder
        TEMP_DECODER="$ENCODER_MODEL"
    fi
    
    for audio_file in "${flac_files[@]}"; do
        count=$((count + 1))
        
        if [ $((count % 50)) -eq 0 ]; then
            echo "  Progress: $count/$total_files files"
        fi
        
        if "$SCRIPT_DIR/build/rknn_whisper_demo" \
            "$ENCODER_MODEL" "$TEMP_DECODER" "$TASK" "$audio_file" \
            > /dev/null 2>&1; then
            success=$((success + 1))
        fi
    done
    
    enc_files=$(ls -1 "$DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l)
    
    echo ""
    echo "Processed: $count files"
    echo "Generated: $enc_files encoder output files"
    
    if [ $enc_files -eq 0 ]; then
        echo "❌ No encoder outputs generated. Check dump_dir in whisper.cc"
        exit 1
    fi
    
    echo "✅ Encoder outputs generated successfully"
}

function step3_convert_to_calibration_dataset() {
    print_banner "Step 3: Convert to Calibration Dataset"
    
    cd "$PROJECT_ROOT/examples/whisper/python"
    
    python3 bin_to_decoder_dataset.py \
        --bin_dir "$DUMP_DIR" \
        --output_dir "$CALIB_DIR" \
        --seq_len "$SEQ_LEN" \
        --hidden_dim "$HIDDEN_DIM" \
        --max_samples "$MAX_SAMPLES" \
        --verify
    
    cd "$SCRIPT_DIR"
    
    if [ ! -f "$CALIB_DIR/dataset.txt" ]; then
        echo "❌ Failed to generate calibration dataset"
        exit 1
    fi
    
    echo "✅ Calibration dataset generated"
}

function step4_quantize_decoder() {
    print_banner "Step 4: Quantize Decoder Model"
    
    cd "$PROJECT_ROOT/examples/whisper/python"
    
    # Create a temporary convert script with correct dataset path
    TEMP_CONVERT=$(mktemp)
    cat > "$TEMP_CONVERT" << EOF
import sys
from rknn.api import RKNN

model_path = "$DECODER_ONNX"
platform = "$PLATFORM"
output_path = "$DECODER_MODEL_OUT"
dataset_path = "$CALIB_DIR/dataset.txt"

print(f"Model: {model_path}")
print(f"Platform: {platform}")
print(f"Dataset: {dataset_path}")
print(f"Output: {output_path}")

# Create RKNN object
rknn = RKNN(verbose=True)

# Config
print('\\n--> Config model')
rknn.config(target_platform=platform)
print('done')

# Load model
print('\\n--> Loading model')
ret = rknn.load_onnx(model=model_path)
if ret != 0:
    print('Load model failed!')
    sys.exit(ret)
print('done')

# Build with quantization
print('\\n--> Building model with INT8 quantization')
print(f'    Using dataset: {dataset_path}')
ret = rknn.build(do_quantization=True, dataset=dataset_path)
if ret != 0:
    print('Build model failed!')
    sys.exit(ret)
print('done')

# Export
print('\\n--> Export rknn model')
ret = rknn.export_rknn(output_path)
if ret != 0:
    print('Export rknn model failed!')
    sys.exit(ret)
print('done')

# Release
rknn.release()

print(f'\\n✅ Decoder model quantized successfully: {output_path}')
EOF
    
    python3 "$TEMP_CONVERT"
    rm "$TEMP_CONVERT"
    
    cd "$SCRIPT_DIR"
    
    if [ ! -f "$DECODER_MODEL_OUT" ]; then
        echo "❌ Failed to generate quantized decoder model"
        exit 1
    fi
    
    echo "✅ Decoder model quantized successfully"
}

function main() {
    print_banner "Whisper Decoder INT8 Quantization Pipeline"
    print_config
    
    read -p "Continue with these settings? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    check_requirements
    step1_check_dump_dir
    step2_generate_encoder_outputs
    step3_convert_to_calibration_dataset
    step4_quantize_decoder
    
    print_banner "✅ All Steps Completed!"
    echo "Quantized decoder model: $DECODER_MODEL_OUT"
    echo ""
    echo "You can now test the quantized model with:"
    echo "  cd $SCRIPT_DIR/build"
    echo "  ./rknn_whisper_demo \\"
    echo "      $ENCODER_MODEL \\"
    echo "      $DECODER_MODEL_OUT \\"
    echo "      $TASK \\"
    echo "      /path/to/test_audio.flac"
    echo ""
}

# Run main function
main
