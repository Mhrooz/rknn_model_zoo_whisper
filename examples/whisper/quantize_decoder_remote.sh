#!/bin/bash
# End-to-end decoder quantization with remote board execution support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Execution mode: local or remote
EXEC_MODE="${EXEC_MODE:-remote}"  # remote by default for RK3588

# Remote board configuration (for remote mode)
BOARD_IP="${BOARD_IP:-10.204.62.95}"
BOARD_USER="${BOARD_USER:-hanzhang}"
BOARD_SSH_KEY="${BOARD_SSH_KEY:-~/.ssh/id_rsa}"
BOARD_WORK_DIR="${BOARD_WORK_DIR:-/mnt/playground/hanzhang/RTT/whisper_work}"

# Dataset configuration
LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-$PROJECT_ROOT/datasets/LibriSpeech/dev-clean}"
DUMP_DIR="${DUMP_DIR:-$SCRIPT_DIR/encoder_dumps}"
CALIB_DIR="${CALIB_DIR:-$SCRIPT_DIR/decoder_calib}"

# Model configuration
ENCODER_MODEL="${ENCODER_MODEL:-$SCRIPT_DIR/model/whisper_encoder_base_20s_i8_2.rknn}"
DECODER_ONNX="${DECODER_ONNX:-$SCRIPT_DIR/model/whisper_decoder.onnx}"
DECODER_MODEL_OUT="${DECODER_MODEL_OUT:-$SCRIPT_DIR/model/whisper_decoder_base_20s_i8_3.rknn}"

# Processing parameters
TASK="${TASK:-en}"
PLATFORM="${PLATFORM:-rk3588}"
SEQ_LEN="${SEQ_LEN:-1000}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

SSH_CMD="ssh -i $BOARD_SSH_KEY $BOARD_USER@$BOARD_IP"
SCP_CMD="scp -i $BOARD_SSH_KEY"

function print_banner() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

function print_config() {
    print_banner "Configuration"
    echo "Execution Mode:         $EXEC_MODE"
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "Board IP:               $BOARD_IP"
        echo "Board User:             $BOARD_USER"
        echo "Board Work Dir:         $BOARD_WORK_DIR"
    fi
    echo ""
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

function step1_generate_encoder_outputs() {
    print_banner "Step 1: Generate Encoder Outputs on Board"
    
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "Using remote execution on RK3588 board..."
        echo ""
        
        # Export environment variables for remote script
        export BOARD_IP="$BOARD_IP"
        export BOARD_USER="$BOARD_USER"
        export BOARD_SSH_KEY="$BOARD_SSH_KEY"
        export BOARD_WORK_DIR="$BOARD_WORK_DIR"
        export LOCAL_LIBRISPEECH="$LIBRISPEECH_DIR"
        export LOCAL_DUMP_DIR="$DUMP_DIR"
        export ENCODER_MODEL="$ENCODER_MODEL"
        export DECODER_MODEL="$ENCODER_MODEL"  # Use encoder as placeholder
        export TASK="$TASK"
        export MAX_FILES="$MAX_SAMPLES"
        
        # Run remote generation script
        "$SCRIPT_DIR/generate_encoder_outputs_remote.sh"
    else
        echo "❌ Local execution mode not fully supported"
        echo "   RK3588 NPU requires running on the actual board"
        echo "   Please use EXEC_MODE=remote"
        exit 1
    fi
    
    # Verify outputs
    local enc_count=$(ls -1 "$DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l)
    if [ $enc_count -eq 0 ]; then
        echo "❌ No encoder outputs generated"
        exit 1
    fi
    
    echo "✅ Generated $enc_count encoder outputs"
}

function step2_convert_to_calibration_dataset() {
    print_banner "Step 2: Convert to Calibration Dataset"
    
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

function step3_quantize_decoder() {
    print_banner "Step 3: Quantize Decoder Model"
    
    cd "$PROJECT_ROOT/examples/whisper/python"
    
    # Create temporary convert script with correct dataset path
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

function step4_upload_quantized_model() {
    print_banner "Step 4: Upload Quantized Model to Board"
    
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "Uploading quantized decoder to board..."
        
        $SCP_CMD "$DECODER_MODEL_OUT" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/whisper_decoder_int8.rknn
        
        echo "✅ Quantized model uploaded to board"
        echo "   Location: $BOARD_WORK_DIR/model/whisper_decoder_int8.rknn"
    else
        echo "Skipping upload (local mode)"
    fi
}

function show_test_instructions() {
    print_banner "Testing Instructions"
    
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "To test the quantized model on the board:"
        echo ""
        echo "1. SSH to the board:"
        echo "   ssh -i $BOARD_SSH_KEY $BOARD_USER@$BOARD_IP"
        echo ""
        echo "2. Run inference:"
        echo "   cd $BOARD_WORK_DIR"
        echo "   ./rknn_whisper_demo \\"
        echo "       model/whisper_encoder.rknn \\"
        echo "       model/whisper_decoder_int8.rknn \\"
        echo "       $TASK \\"
        echo "       /path/to/test_audio.flac"
        echo ""
        echo "Or upload a test audio file:"
        echo "   scp -i $BOARD_SSH_KEY test.flac $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/"
    fi
}

function main() {
    print_banner "Whisper Decoder INT8 Quantization Pipeline (Remote Execution)"
    print_config
    
    # Warnings for remote mode
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "⚠️  IMPORTANT NOTES:"
        echo "   1. Before running, make sure you have compiled rknn_whisper_demo with:"
        echo "      const std::string dump_dir = \"$BOARD_WORK_DIR/dumps\";"
        echo "   2. The board must be on and accessible at $BOARD_IP"
        echo "   3. SSH key authentication must be configured"
        echo ""
    fi
    
    read -p "Continue with these settings? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    step1_generate_encoder_outputs
    step2_convert_to_calibration_dataset
    step3_quantize_decoder
    step4_upload_quantized_model
    show_test_instructions
    
    print_banner "✅ All Steps Completed!"
    echo "Quantized decoder model: $DECODER_MODEL_OUT"
    if [ "$EXEC_MODE" = "remote" ]; then
        echo "Also available on board at: $BOARD_WORK_DIR/model/whisper_decoder_int8.rknn"
    fi
    echo ""
}

# Run main function
main
