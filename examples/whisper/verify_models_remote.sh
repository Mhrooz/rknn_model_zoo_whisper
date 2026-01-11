#!/bin/bash
# 在开发板上运行模型验证

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 板子配置
BOARD_IP="${BOARD_IP:-10.204.62.95}"
BOARD_USER="${BOARD_USER:-hanzhang}"
BOARD_SSH_KEY="${BOARD_SSH_KEY:-~/.ssh/id_rsa}"
BOARD_SSH_KEY="${BOARD_SSH_KEY/#\~/$HOME}"
BOARD_WORK_DIR="${BOARD_WORK_DIR:-/mnt/playground/hanzhang/RTT/whisper_work}"

SSH_CMD="ssh -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no $BOARD_USER@$BOARD_IP"
SCP_CMD="scp -i $BOARD_SSH_KEY -o StrictHostKeyChecking=no"

echo "=================================================="
echo "在开发板上验证 RKNN 模型"
echo "=================================================="
echo ""
echo "开发板: $BOARD_USER@$BOARD_IP"
echo "工作目录: $BOARD_WORK_DIR"
echo ""

# 检查连接
echo "检查 SSH 连接..."
if ! $SSH_CMD "echo 'OK'" > /dev/null 2>&1; then
    echo "❌ 无法连接到开发板"
    exit 1
fi
echo "✅ SSH 连接正常"
echo ""

# 上传验证脚本
echo "上传验证脚本..."
if ! $SCP_CMD "$SCRIPT_DIR/python/verify_models.py" $BOARD_USER@$BOARD_IP:$BOARD_WORK_DIR/; then
    echo "❌ 上传失败"
    exit 1
fi
echo "✅ 脚本已上传"
echo ""

# 在板上运行
echo "在开发板上执行验证..."
echo "=================================================="
$SSH_CMD "cd $BOARD_WORK_DIR && python3 verify_models.py \
  --encoder model/whisper_encoder.rknn \
  --decoder model/whisper_decoder.rknn"

RESULT=$?
echo ""
echo "=================================================="
if [ $RESULT -eq 0 ]; then
    echo "✅ 验证完成"
else
    echo "❌ 验证失败 (退出码: $RESULT)"
fi
echo ""
echo "提示: 如果看到错误，请查看上面的详细输出"
echo "      特别注意 Encoder/Decoder 的统计信息"

exit $RESULT
