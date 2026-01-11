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
LOCAL_LIBRISPEECH="${LOCAL_LIBRISPEECH:-../../datasets/Librispeech/dev-clean}"
LOCAL_DUMP_DIR="${LOCAL_DUMP_DIR:-./encoder_dumps}"
# encoder model on host (will be uploaded to the board)
ENCODER_MODEL="${ENCODER_MODEL:-./model/whisper_encoder_base_i8_2.rknn}"
# decoder model on host (will be uploaded too). Keep this as a host path.
DECODER_MODEL="${DECODER_MODEL:-./model/whisper_decoder_base_i8.rknn}"
TASK="${TASK:-en}"
MAX_FILES="${MAX_FILES:-500}"

# Set NONINTERACTIVE=1 in env to skip interactive prompts (useful for CI)
NONINTERACTIVE="${NONINTERACTIVE:-0}"

# Skip steps for faster debugging (set to 1 to skip):
# SKIP_UPLOAD_MODELS=1    - Skip uploading encoder/decoder models
# SKIP_UPLOAD_EXE=1       - Skip uploading executable
# SKIP_UPLOAD_DEPS=1      - Skip uploading mel_filters and vocab files
# SKIP_SETUP_BOARD=1      - Skip all uploads (models, exe, deps)
SKIP_UPLOAD_MODELS="${SKIP_UPLOAD_MODELS:-0}"
SKIP_UPLOAD_EXE="${SKIP_UPLOAD_EXE:-0}"
SKIP_UPLOAD_DEPS="${SKIP_UPLOAD_DEPS:-0}"
SKIP_SETUP_BOARD="${SKIP_SETUP_BOARD:-0}"

SSH_CMD="ssh -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no $BOARD_USER@$BOARD_IP"
SCP_CMD="scp -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no"

function print_banner() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo ""
}

function print_usage() {
    cat << 'EOF'
Usage: ./generate_encoder_outputs_remote.sh

Environment variables for skipping steps (useful for debugging):
  SKIP_SETUP_BOARD=1     Skip all uploads (models, executable, deps)
  SKIP_UPLOAD_MODELS=1   Skip uploading encoder/decoder models
  SKIP_UPLOAD_EXE=1      Skip uploading executable
  SKIP_UPLOAD_DEPS=1     Skip uploading mel_filters and vocab files
  NONINTERACTIVE=1       Skip interactive prompts

Examples:
  # First run - upload everything
  ./generate_encoder_outputs_remote.sh
  
  # Quick rerun - skip model uploads (models already on board)
  SKIP_UPLOAD_MODELS=1 ./generate_encoder_outputs_remote.sh
  
  # After recompiling - only upload new executable
  SKIP_UPLOAD_MODELS=1 SKIP_UPLOAD_DEPS=1 ./generate_encoder_outputs_remote.sh
  
  # Skip all uploads - just process audio (board already set up)
  SKIP_SETUP_BOARD=1 NONINTERACTIVE=1 ./generate_encoder_outputs_remote.sh

EOF
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
    echo "Encoder Model:       $ENCODER_MODEL      (HOST -> will be uploaded to board)")
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
    
    if [ "$SKIP_SETUP_BOARD" = "1" ] || [ "$SKIP_UPLOAD_MODELS" = "1" ]; then
        echo "⏭️  Skipping model upload (SKIP_UPLOAD_MODELS=1)"
        return 0
    fi
    
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
    
    if [ "$SKIP_SETUP_BOARD" = "1" ] || [ "$SKIP_UPLOAD_EXE" = "1" ]; then
        echo "⏭️  Skipping executable upload (SKIP_UPLOAD_EXE=1)"
        return 0
    fi
    
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
    
    if [ "$SKIP_UPLOAD_DEPS" = "1" ]; then
        echo "⏭️  Skipping dependency files upload (SKIP_UPLOAD_DEPS=1)"
        echo "✅ Executable uploaded"
        return 0
    fi
    
    # Upload mel filters
    local mel_filters="$SCRIPT_DIR/model/mel_80_filters.txt"
    if [ -f "$mel_filters" ]; then
        echo "Uploading mel filters..."
        $SSH_CMD "mkdir -p $BOARD_WORK_DIR/model"
        if ! $SCP_CMD "$mel_filters" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/model/; then
            echo "⚠️  Warning: failed to upload mel filters"
        fi
    else
        echo "⚠️  Warning: mel_80_filters.txt not found at $mel_filters"
    fi
    
    # Upload vocab files if exist
    for vocab in "$SCRIPT_DIR/model/vocab_en.txt" "$SCRIPT_DIR/model/vocab_zh.txt"; do
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
    echo "  [DEBUG] Creating temp directory..."
    
    # Create temporary directory for converted audio
    local temp_dir=$(mktemp -d)
    echo "  [DEBUG] Temp dir: $temp_dir"
    
    # Convert FLAC to WAV (board's libsndfile may not support FLAC)
    echo "  Converting FLAC to WAV format for board compatibility..."
    
    echo "  [DEBUG] Checking for conversion tools..."
    local conversion_tool=""
    if command -v ffmpeg >/dev/null 2>&1; then
        conversion_tool="ffmpeg"
        echo "  Using: ffmpeg"
    elif command -v sox >/dev/null 2>&1; then
        conversion_tool="sox"
        echo "  Using: sox"
    else
        echo "  ❌ ERROR: Neither ffmpeg nor sox found. Please install one:"
        echo "     macOS: brew install ffmpeg"
        echo "     Linux: sudo apt-get install ffmpeg"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    echo "  [DEBUG] Starting conversion of ${#batch_files[@]} files..."
    echo "  [DEBUG] First file: ${batch_files[0]}"
    if [ ! -f "${batch_files[0]}" ]; then
        echo "  ❌ ERROR: First audio file not found: ${batch_files[0]}"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    local converted=0
    local total=${#batch_files[@]}
    
    echo "  [DEBUG] Entering for loop..."
    for audio_file in "${batch_files[@]}"; do
        echo "  [DEBUG] Loop iteration: audio_file=$audio_file"
        local filename=$(basename "$audio_file" .flac)
        local wav_file="$temp_dir/${filename}.wav"
        
        converted=$((converted + 1))  # Safe for set -e
        printf "    [$converted/$total] %-40s" "$(basename "$audio_file")..."
        
        # Convert using available tool (disable set -e temporarily)
        set +e
        if [ "$conversion_tool" = "ffmpeg" ]; then
            ffmpeg -i "$audio_file" -ar 16000 -ac 1 -sample_fmt s16 "$wav_file" -y -loglevel warning 2>/dev/null
            local ret=$?
        else
            sox "$audio_file" -r 16000 -c 1 -b 16 "$wav_file" 2>/dev/null
            local ret=$?
        fi
        set -e
        
        if [ $ret -eq 0 ] && [ -f "$wav_file" ]; then
            echo " ✅"
        else
            echo " ❌ FAILED"
            echo "  ❌ ERROR: Failed to convert $audio_file"
            rm -rf "$temp_dir"
            exit 1
        fi
    done
    
    echo "  Conversion complete: $converted files processed"
    
    # Upload ONLY converted WAV files (not FLAC)
    echo "  Uploading WAV files to board..."
    local uploaded=0
    local total_wav=$(ls -1 "$temp_dir"/*.wav 2>/dev/null | wc -l)
    total_wav=$(echo "$total_wav" | tr -d ' ')
    
    for wav_file in "$temp_dir"/*.wav; do
        if [ -f "$wav_file" ]; then
            uploaded=$((uploaded + 1))  # Safe for set -e
            printf "    [$uploaded/$total_wav] %-40s" "$(basename $wav_file)..."
            
            # Disable set -e temporarily for upload
            set +e
            $SCP_CMD "$wav_file" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/audio/ >/dev/null 2>&1
            local ret=$?
            set -e
            
            if [ $ret -eq 0 ]; then
                echo " ✅"
            else
                echo " ❌ FAILED"
                echo "  ❌ ERROR: Failed to upload $(basename $wav_file)"
                rm -rf "$temp_dir"
                exit 1
            fi
        fi
    done
    
    echo "  Upload complete: $uploaded/$total_wav files uploaded"
    
    if [ $uploaded -eq 0 ]; then
        echo "  ❌ ERROR: No WAV files uploaded"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
}

function process_audio_on_board() {
    local audio_files=("$@")
    local batch_num=$1
    
    echo "  Processing ${#audio_files[@]} audio files on board..."
    
    # Create processing script on board with detailed logging
    $SSH_CMD "cat > $BOARD_WORK_DIR/process_batch.sh << 'EOF'
#!/bin/bash
cd $BOARD_WORK_DIR

# Log file for this batch
LOG_FILE=\"batch_\$(date +%Y%m%d_%H%M%S).log\"
echo \"=== Batch processing started at \$(date) ===\" > \"\$LOG_FILE\"
echo \"Working directory: \$(pwd)\" >> \"\$LOG_FILE\"
echo \"\" >> \"\$LOG_FILE\"

# Check if executable exists and is executable
if [ ! -f \"./rknn_whisper_demo\" ]; then
    echo \"ERROR: rknn_whisper_demo not found\" | tee -a \"\$LOG_FILE\"
    exit 1
fi

if [ ! -x \"./rknn_whisper_demo\" ]; then
    echo \"WARNING: rknn_whisper_demo not executable, fixing...\" | tee -a \"\$LOG_FILE\"
    chmod +x ./rknn_whisper_demo
fi

# Check if models exist
for model in model/whisper_encoder.rknn model/whisper_decoder.rknn; do
    if [ ! -f \"\$model\" ]; then
        echo \"ERROR: Model not found: \$model\" | tee -a \"\$LOG_FILE\"
        exit 1
    fi
done

# Check dumps directory
if [ ! -d \"dumps\" ]; then
    echo \"Creating dumps directory...\" | tee -a \"\$LOG_FILE\"
    mkdir -p dumps
fi

# Test write permission
if ! touch dumps/test_write 2>/dev/null; then
    echo \"ERROR: Cannot write to dumps directory\" | tee -a \"\$LOG_FILE\"
    exit 1
else
    rm -f dumps/test_write
    echo \"dumps directory writable: OK\" | tee -a \"\$LOG_FILE\"
fi

# Process audio files
count=0
success=0
failed=0

# Only process WAV files (not FLAC - those are not uploaded)
for audio in audio/*.wav; do
    if [ -f \"\$audio\" ]; then
        echo \"\" >> \"\$LOG_FILE\"
        echo \"Processing: \$audio\" | tee -a \"\$LOG_FILE\"
        
        # Run demo and capture output
        if ./rknn_whisper_demo \\
            model/whisper_encoder.rknn \\
            model/whisper_decoder.rknn \\
            $TASK \\
            \"\$audio\" >> \"\$LOG_FILE\" 2>&1; then
            echo \"  -> SUCCESS\" | tee -a \"\$LOG_FILE\"
            ((success++))
        else
            echo \"  -> FAILED (exit code: \$?)\" | tee -a \"\$LOG_FILE\"
            ((failed++))
        fi
        ((count++))
    fi
done

# Summary
echo \"\" | tee -a \"\$LOG_FILE\"
echo \"=== Processing complete at \$(date) ===\" | tee -a \"\$LOG_FILE\"
echo \"Total: \$count, Success: \$success, Failed: \$failed\" | tee -a \"\$LOG_FILE\"

# Check dumps directory
enc_count=\$(ls -1 dumps/enc_*.bin 2>/dev/null | wc -l)
echo \"Generated \$enc_count encoder output files\" | tee -a \"\$LOG_FILE\"

if [ \$enc_count -eq 0 ]; then
    echo \"WARNING: No enc_*.bin files generated!\" | tee -a \"\$LOG_FILE\"
    echo \"Checking dumps directory:\" | tee -a \"\$LOG_FILE\"
    ls -lah dumps/ >> \"\$LOG_FILE\" 2>&1
fi

# Keep log file for debugging
echo \"Log file: \$LOG_FILE\"
EOF
chmod +x $BOARD_WORK_DIR/process_batch.sh"
    
    # Execute processing and capture output
    echo "  Executing on board (this may take a while)..."
    if ! $SSH_CMD "$BOARD_WORK_DIR/process_batch.sh" 2>&1 | tee /tmp/board_output_$$.log; then
        echo "  ⚠️  Remote execution returned non-zero exit code"
    fi
    
    # Download and display the log file
    local log_file=$($SSH_CMD "ls -t $BOARD_WORK_DIR/batch_*.log 2>/dev/null | head -1" || echo "")
    if [ -n "$log_file" ]; then
        echo ""
        echo "  📄 Fetching detailed log from board..."
        $SSH_CMD "cat $log_file" | tee /tmp/board_detailed_$$.log
        echo ""
        echo "  💾 Board log saved locally: /tmp/board_detailed_$$.log"
    fi
}

function download_results() {
    local batch_num=$1
    
    echo "  Downloading results from board..."
    
    # Check how many files exist on board first
    local remote_count=$($SSH_CMD "ls -1 $BOARD_WORK_DIR/dumps/enc_*.bin 2>/dev/null | wc -l" || echo "0")
    remote_count=$(echo "$remote_count" | tr -d ' ')
    
    echo "  Found $remote_count enc_*.bin files on board"
    
    if [ "$remote_count" -eq 0 ]; then
        echo "  ⚠️  WARNING: No encoder output files found on board!"
        echo "  Checking board dumps directory..."
        $SSH_CMD "ls -lah $BOARD_WORK_DIR/dumps/ 2>&1" || true
        echo ""
        echo "  Checking if program wrote to a different location..."
        $SSH_CMD "find $BOARD_WORK_DIR -name 'enc_*.bin' 2>/dev/null" || true
        return 1
    fi
    
    # Download all enc_*.bin files
    mkdir -p "$LOCAL_DUMP_DIR"
    
    # Use rsync if available for better progress, otherwise use scp
    if command -v rsync >/dev/null 2>&1; then
        rsync -avz -e "ssh -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no" \
            $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/dumps/enc_*.bin "$LOCAL_DUMP_DIR/" 2>&1 || {
            echo "  ⚠️  rsync failed, trying scp..."
            $SCP_CMD $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/dumps/enc_\*.bin "$LOCAL_DUMP_DIR/" 2>&1
        }
    else
        # scp with explicit file list
        $SSH_CMD "ls $BOARD_WORK_DIR/dumps/enc_*.bin 2>/dev/null" | while read remote_file; do
            local filename=$(basename "$remote_file")
            echo "    Downloading $filename..."
            if ! $SCP_CMD $BOARD_USER@$BOARD_IP:$remote_file "$LOCAL_DUMP_DIR/" 2>&1; then
                echo "    ⚠️  Failed to download $filename"
            fi
        done
    fi
    
    # Verify downloads
    local local_count=$(ls -1 "$LOCAL_DUMP_DIR"/enc_*.bin 2>/dev/null | wc -l)
    local_count=$(echo "$local_count" | tr -d ' ')
    echo "  Downloaded $local_count of $remote_count files to: $LOCAL_DUMP_DIR"
    
    if [ "$local_count" -lt "$remote_count" ]; then
        echo "  ⚠️  WARNING: Not all files were downloaded!"
    fi
}

function cleanup_board_batch() {
    echo "  Cleaning up board batch files..."
    # Only clean WAV files (we don't upload FLAC anymore)
    $SSH_CMD "rm -f $BOARD_WORK_DIR/audio/*.wav $BOARD_WORK_DIR/dumps/enc_*.bin"
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
    # Check for help flag
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        print_usage
        exit 0
    fi
    
    print_banner "Whisper Encoder Remote Execution Script"
    
    # Show skip status if any skips are enabled
    if [ "$SKIP_SETUP_BOARD" = "1" ] || [ "$SKIP_UPLOAD_MODELS" = "1" ] || [ "$SKIP_UPLOAD_EXE" = "1" ] || [ "$SKIP_UPLOAD_DEPS" = "1" ]; then
        echo "🚀 Quick Run Mode - Skipping:"
        [ "$SKIP_SETUP_BOARD" = "1" ] && echo "   ⏭️  All uploads (SKIP_SETUP_BOARD=1)"
        [ "$SKIP_UPLOAD_MODELS" = "1" ] && echo "   ⏭️  Model uploads (SKIP_UPLOAD_MODELS=1)"
        [ "$SKIP_UPLOAD_EXE" = "1" ] && echo "   ⏭️  Executable upload (SKIP_UPLOAD_EXE=1)"
        [ "$SKIP_UPLOAD_DEPS" = "1" ] && echo "   ⏭️  Dependency uploads (SKIP_UPLOAD_DEPS=1)"
        echo ""
    fi
    
    print_config
    
    if [ "$NONINTERACTIVE" != "1" ]; then
        read -p "Continue with these settings? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
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

# Run main function with all arguments
main "$@"
