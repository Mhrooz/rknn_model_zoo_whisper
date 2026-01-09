#!/bin/bash
# Remote execution script for generating encoder outputs on RK3588 board

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remote board configuration (remote = on-board paths)
BOARD_IP="${BOARD_IP:-10.204.62.95}"
BOARD_USER="${BOARD_USER:-hanzhang}"
# Expand ~ if used in ssh key path
BOARD_SSH_KEY="${BOARD_SSH_KEY:-~/.ssh/id_rsa}"
BOARD_SSH_KEY="${BOARD_SSH_KEY/#\~/$HOME}"
# Default work dir on the board. Usually on RKNN boards we use /mnt/playground/...
BOARD_WORK_DIR="${BOARD_WORK_DIR:-/mnt/playground/hanzhang/RTT/whisper_work}"

# Local configuration (paths on your host)
LOCAL_LIBRISPEECH="${LOCAL_LIBRISPEECH:-../../datasets/LibriSpeech/dev-clean}"
LOCAL_DUMP_DIR="${LOCAL_DUMP_DIR:-./encoder_dumps}"
# encoder model on host (will be uploaded to the board)
ENCODER_MODEL="${ENCODER_MODEL:-./model/whisper_encoder_base_20s.rknn}"
# decoder model on host (will be uploaded too). Keep this as a host path.
DECODER_MODEL="${DECODER_MODEL:-./model/whisper_decoder_base_20s.rknn}"
TASK="${TASK:-en}"
MAX_FILES="${MAX_FILES:-500}"

# Set NONINTERACTIVE=1 in env to skip interactive prompts (useful for CI)
NONINTERACTIVE="${NONINTERACTIVE:-0}"

SSH_CMD="ssh -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no $BOARD_USER@$BOARD_IP"
SCP_CMD="scp -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no"

function print_banner() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

function print_config() {
    print_banner "Remote Execution Configuration"
    echo "Board IP:            $BOARD_IP"
    echo "Board User:          $BOARD_USER"
    echo "Board SSH Key:       $BOARD_SSH_KEY"
    echo "Board Work Dir:      $BOARD_WORK_DIR"
    echo ""
    echo "Local Librispeech:   $LOCAL_LIBRISPEECH  (HOST)"
    echo "Local Dump Dir:      $LOCAL_DUMP_DIR      (HOST)"
    echo "Encoder Model:       $ENCODER_MODEL      (HOST -> will be uploaded to board)"
    echo "Decoder Model:       $DECODER_MODEL      (HOST -> will be uploaded to board)"
    echo "Task:                $TASK"
    echo "Max Files:           $MAX_FILES"
    echo ""
}

function check_ssh_connection() {
    print_banner "Step 1: Checking SSH Connection"
    
    echo "Testing connection to $BOARD_USER@$BOARD_IP..."
    if $SSH_CMD "echo 'SSH connection successful'" > /dev/null 2>&1; then
        echo "✅ SSH connection OK"
    else
        echo "❌ SSH connection failed"
        echo "   Please check:"
        echo "   - Board IP: $BOARD_IP"
        echo "   - SSH key: $BOARD_SSH_KEY"
        echo "   - Network connectivity"
        exit 1
    fi
}

function prepare_board() {
    print_banner "Step 2: Preparing Board Environment"
    
    echo "Creating work directory on board..."
    $SSH_CMD "mkdir -p $BOARD_WORK_DIR/model $BOARD_WORK_DIR/dumps $BOARD_WORK_DIR/audio"
    
    echo "✅ Board directories created"
}

function upload_models() {
    print_banner "Step 3: Uploading Models to Board"
    
    if [ ! -f "$ENCODER_MODEL" ]; then
        echo "❌ Encoder model not found: $ENCODER_MODEL"
        exit 1
    fi
    
    if [ ! -f "$DECODER_MODEL" ]; then
        echo "❌ Decoder model not found: $DECODER_MODEL"
        exit 1
    fi
    
    echo "Uploading encoder model to board: $BOARD_WORK_DIR/model/whisper_encoder.rknn"
    if ! $SCP_CMD "$ENCODER_MODEL" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/whisper_encoder.rknn; then
        echo "❌ Failed to upload encoder model"
        exit 1
    fi
    
    echo "Uploading decoder model to board: $BOARD_WORK_DIR/model/whisper_decoder.rknn"
    if ! $SCP_CMD "$DECODER_MODEL" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/whisper_decoder.rknn; then
        echo "❌ Failed to upload decoder model"
        exit 1
    fi
    
    echo "✅ Models uploaded"
}

function upload_executable() {
    print_banner "Step 4: Uploading Executable"
    
    local exe_path="$SCRIPT_DIR/cpp/build/rknn_whisper_demo"
    
    if [ ! -f "$exe_path" ]; then
        echo "⚠️  Executable not found, building..."
        cd "$SCRIPT_DIR/cpp/build"
        cmake .. && make
        cd "$SCRIPT_DIR"
    fi
    
    if [ ! -f "$exe_path" ]; then
        echo "❌ Failed to build executable"
        exit 1
    fi
    
    echo "Uploading executable to board: $BOARD_WORK_DIR/rknn_whisper_demo"
    if ! $SCP_CMD "$exe_path" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/rknn_whisper_demo; then
        echo "❌ Failed to upload executable"
        exit 1
    fi
    
    # Upload mel filters
    local mel_filters="$SCRIPT_DIR/cpp/model/mel_80_filters.txt"
    if [ -f "$mel_filters" ]; then
        echo "Uploading mel filters..."
        $SSH_CMD "mkdir -p $BOARD_WORK_DIR/model"
        if ! $SCP_CMD "$mel_filters" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/; then
            echo "⚠️  Warning: failed to upload mel filters"
        fi
    fi
    
    # Upload vocab files if exist
    for vocab in "$SCRIPT_DIR/cpp/model/vocab_en.txt" "$SCRIPT_DIR/cpp/model/vocab_zh.txt"; do
        if [ -f "$vocab" ]; then
            echo "Uploading $(basename $vocab)..."
            $SCP_CMD "$vocab" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/
        fi
    done
    
    echo "✅ Executable and dependencies uploaded"
}

function upload_audio_batch() {
    local batch_num=$1
    local batch_files=("${@:2}")
    
    echo "  Uploading batch $batch_num (${#batch_files[@]} files)..."
    
    # Upload audio files
    for audio_file in "${batch_files[@]}"; do
        local filename=$(basename "$audio_file")
        $SCP_CMD "$audio_file" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/audio/ > /dev/null 2>&1
    done
}

function process_audio_on_board() {
    local audio_files=("$@")
    
    echo "  Processing ${#audio_files[@]} audio files on board..."
    
    # Create processing script on board
    $SSH_CMD "cat > $BOARD_WORK_DIR/process_batch.sh << 'EOF'
#!/bin/bash
cd $BOARD_WORK_DIR
count=0
for audio in audio/*.flac; do
    if [ -f \"\$audio\" ]; then
        ./rknn_whisper_demo \\
            model/whisper_encoder.rknn \\
            model/whisper_decoder.rknn \\
            $TASK \\
            \"\$audio\" > /dev/null 2>&1
        ((count++))
    fi
done
echo \"Processed \$count files\"
EOF
chmod +x $BOARD_WORK_DIR/process_batch.sh"
    
    # Execute processing
    $SSH_CMD "$BOARD_WORK_DIR/process_batch.sh"
}

function download_results() {
    local batch_num=$1
    
    echo "  Downloading results from board..."
    
    # Download all enc_*.bin files
    $SSH_CMD "ls $BOARD_WORK_DIR/dumps/enc_*.bin 2>/dev/null" | while read remote_file; do
        local filename=$(basename "$remote_file")
        $SCP_CMD $BOARD_USER@$BOARD_IP:$remote_file "$LOCAL_DUMP_DIR/" > /dev/null 2>&1
    done
}

function cleanup_board_batch() {
    echo "  Cleaning up board batch files..."
    $SSH_CMD "rm -f $BOARD_WORK_DIR/audio/*.flac $BOARD_WORK_DIR/dumps/enc_*.bin"
}

function generate_encoder_outputs() {
    print_banner "Step 5: Generating Encoder Outputs"
    
    # Check if librispeech directory exists
    if [ ! -d "$LOCAL_LIBRISPEECH" ]; then
        echo "❌ Librispeech directory not found: $LOCAL_LIBRISPEECH"
        exit 1
    fi
    
    # Create local dump directory
    mkdir -p "$LOCAL_DUMP_DIR"
    
    # Find all FLAC files
    echo "Finding FLAC files..."
    mapfile -t flac_files < <(find "$LOCAL_LIBRISPEECH" -name "*.flac" | head -n "$MAX_FILES")
    total_files=${#flac_files[@]}
    
    if [ $total_files -eq 0 ]; then
        echo "❌ No FLAC files found"
        exit 1
    fi
    
    echo "Found $total_files FLAC files"
    echo ""
    
    # Configure dump_dir on board (create a config file)
    $SSH_CMD "cat > $BOARD_WORK_DIR/set_dump_dir.sh << 'EOF'
#!/bin/bash
# This script reminds you to set dump_dir in whisper.cc
echo \"Make sure whisper.cc has: const std::string dump_dir = \\\"$BOARD_WORK_DIR/dumps\\\";\"
EOF
chmod +x $BOARD_WORK_DIR/set_dump_dir.sh"
    
    echo "⚠️  IMPORTANT: Make sure whisper.cc on your BUILD MACHINE has:"
    echo "    const std::string dump_dir = \"$BOARD_WORK_DIR/dumps\";"
    echo "  (This is a REMOTE board path where the executable will write enc_*.bin)"
    echo ""
    if [ "$NONINTERACTIVE" != "1" ]; then
        read -p "Press Enter to continue or Ctrl+C to cancel..."
    else
        echo "NONINTERACTIVE=1 set, proceeding without prompt"
    fi
    
    # Process in batches to avoid overwhelming the board
    local batch_size=10
    local processed=0
    local batch_num=0
    
    for ((i=0; i<total_files; i+=batch_size)); do
        batch_num=$((batch_num + 1))
        batch_files=("${flac_files[@]:i:batch_size}")
        
        echo ""
        echo "Processing batch $batch_num (files $((i+1))-$((i+${#batch_files[@]})) of $total_files)"
        
        # Upload batch
        upload_audio_batch $batch_num "${batch_files[@]}"
        
        # Process on board
        if ! process_audio_on_board "${batch_files[@]}"; then
            echo "⚠️  Warning: processing batch $batch_num failed on board"
        fi
        
        # Download results
        download_results $batch_num
        
        # Cleanup
        cleanup_board_batch
        
        processed=$((processed + ${#batch_files[@]}))
        
        echo "  Progress: $processed/$total_files files processed"
    done
    
    # Count results
    local result_count=$(ls -1 "$LOCAL_DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l || true)
    
    echo ""
    echo "✅ Generated $result_count encoder output files (if zero, check board logs and dump_dir settings)"
    echo "   Output directory: $LOCAL_DUMP_DIR"
}

function verify_outputs() {
    print_banner "Step 6: Verifying Outputs"
    
    local enc_count=$(ls -1 "$LOCAL_DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l)
    
    if [ $enc_count -eq 0 ]; then
        echo "❌ No encoder outputs found"
        echo "   Please check:"
        echo "   1. dump_dir is set correctly in whisper.cc"
        echo "   2. Executable was recompiled after changing dump_dir"
        echo "   3. Board has write permission to $BOARD_WORK_DIR/dumps"
        return 1
    fi
    
    echo "Found $enc_count encoder output files"
    
    # Check first file size
    local first_file=$(ls -1 "$LOCAL_DUMP_DIR"/enc_*.bin 2>/dev/null | head -1)
    if [ -f "$first_file" ]; then
        local file_size=$(stat -f%z "$first_file" 2>/dev/null || stat -c%s "$first_file" 2>/dev/null)
        local expected_size=2048000  # 1000 * 512 * 4 bytes for 20s audio
        
        echo "First file: $(basename $first_file)"
        echo "Size: $file_size bytes (expected: $expected_size bytes)"
        
        if [ $file_size -eq $expected_size ]; then
            echo "✅ File size correct"
        else
            echo "⚠️  File size mismatch (may be OK if using different audio length)"
        fi
    fi
    
    echo ""
    echo "✅ Verification complete"
}

function cleanup_board() {
    print_banner "Cleanup (Optional)"
    
    echo "Do you want to clean up files on the board?"
    read -p "This will remove models and dumps (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning up board..."
        $SSH_CMD "rm -rf $BOARD_WORK_DIR/dumps/* $BOARD_WORK_DIR/audio/*"
        echo "✅ Board cleaned"
        
        echo ""
        echo "Keep models on board for future use?"
        read -p "(y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            $SSH_CMD "rm -rf $BOARD_WORK_DIR/model/*"
            echo "Models removed from board"
        fi
    else
        echo "Skipping cleanup. Files remain on board at: $BOARD_WORK_DIR"
    fi
}

function main() {
    print_banner "Whisper Encoder Remote Execution Script"
    print_config
    
    read -p "Continue with these settings? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    check_ssh_connection
    prepare_board
    upload_models
    upload_executable
    generate_encoder_outputs
    verify_outputs
    cleanup_board
    
    print_banner "✅ Remote Execution Complete!"
    echo "Encoder outputs saved to: $LOCAL_DUMP_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Convert to decoder calibration dataset:"
    echo "     cd python"
    echo "     python bin_to_decoder_dataset.py \\"
    echo "         --bin_dir $LOCAL_DUMP_DIR \\"
    echo "         --output_dir ./decoder_calib \\"
    echo "         --seq_len 1000 \\"
    echo "         --hidden_dim 512 \\"
    echo "         --verify"
    echo ""
    echo "  2. Quantize decoder:"
    echo "     python convert.py \\"
    echo "         whisper_decoder.onnx \\"
    echo "         rk3588 \\"
    echo "         i8 \\"
    echo "         whisper_decoder_int8.rknn"
}

# Run main function
main
