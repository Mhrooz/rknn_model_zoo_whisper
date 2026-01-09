# Whisper Decoder 量化 - 快速开始

## 最简单的方法（推荐）

### 🚀 远程执行模式（RK3588开发板）

**推荐！** 因为 RK3588 的 NPU 只能在开发板上运行，使用远程脚本自动处理上传、执行、下载：

```bash
cd examples/whisper

# 1. 确保开发板配置正确（如果需要修改）
export BOARD_IP=10.204.62.95
export BOARD_USER=hanzhang
export BOARD_SSH_KEY=~/.ssh/id_rsa

# 2. 编辑 cpp/rknpu2/whisper.cc，设置开发板上的 dump_dir
vim cpp/rknpu2/whisper.cc
# 找到: const std::string dump_dir = "...";
# 改为: const std::string dump_dir = "/home/hanzhang/whisper_work/dumps";

# 3. 重新编译（会自动上传到开发板）
cd cpp/build
cmake .. && make
cd ../..

# 4. 运行远程端到端脚本
chmod +x quantize_decoder_remote.sh
./quantize_decoder_remote.sh
```

该脚本会自动完成：
1. ✅ SSH 连接到开发板
2. ✅ 上传模型和可执行文件
3. ✅ 批量处理音频生成 encoder 输出
4. ✅ 下载结果到本地
5. ✅ 转换为校准数据集
6. ✅ 量化 decoder
7. ✅ 上传量化后的模型到开发板

### 本地执行模式（如果有本地 RK3588 环境）

如果你的主机就是 RK3588 或有本地模拟环境：

```bash
cd examples/whisper

# 1. 修改 dump_dir
vim cpp/rknpu2/whisper.cc
# 改为: const std::string dump_dir = "/tmp/whisper_dumps";

# 2. 编译并运行
cd cpp/build && cmake .. && make && cd ../..
./quantize_decoder_e2e.sh
```

## 手动步骤（如果需要更多控制）

### 远程执行模式

#### 步骤 1: 在开发板上生成 encoder 输出

```bash
cd examples/whisper

# 使用远程生成脚本
chmod +x generate_encoder_outputs_remote.sh
./generate_encoder_outputs_remote.sh
```

这个脚本会：
- SSH 到开发板
- 上传模型和可执行文件  
- 批量上传音频文件
- 在开发板上运行推理
- 下载 encoder 输出 (enc_*.bin)
- 自动清理临时文件

#### 步骤 2: 转换为校准数据集

```bash
cd python

python bin_to_decoder_dataset.py \
    --bin_dir ../encoder_dumps \
    --output_dir ./decoder_calib \
    --seq_len 1000 \
    --hidden_dim 512 \
    --max_samples 500 \
    --verify
```

#### 步骤 3: 量化 decoder

```bash
# 修改 convert.py 中的 dataset 路径
python convert.py \
    whisper_decoder.onnx \
    rk3588 \
    i8 \
    whisper_decoder_int8.rknn
```

#### 步骤 4: 上传量化模型到开发板

```bash
scp -i ~/.ssh/id_rsa \
    whisper_decoder_int8.rknn \
    hanzhang@10.204.62.95:/home/hanzhang/whisper_work/model/
```

### 本地执行模式

#### 步骤 1: 修改并编译

```bash
cd examples/whisper/cpp

# 修改 rknpu2/whisper.cc 中的 dump_dir
vim rknpu2/whisper.cc

# 编译
cd build && cmake .. && make && cd ..
```

#### 步骤 2: 生成 encoder 输出

```bash
# 使用批处理脚本
chmod +x batch_generate_encoder_outputs.sh
./batch_generate_encoder_outputs.sh \
    ../../datasets/Librispeech/dev-clean \
    /tmp/whisper_dumps \
    ./model/whisper_encoder.rknn \
    ./model/whisper_decoder.rknn \
    en \
    500
```

### 步骤 3: 转换为校准数据集

```bash
cd ../python

python bin_to_decoder_dataset.py \
    --bin_dir /tmp/whisper_dumps \
    --output_dir ./decoder_calib \
    --seq_len 1000 \
    --hidden_dim 512 \
    --max_samples 500 \
    --verify
```

### 步骤 4: 量化 decoder

```bash
# 修改 convert.py 中的 dataset 路径
# 找到这行：dataset='/home/hanzhang/workspace/RTT/rknn_model_zoo/datasets/test_decode/dataset.txt'
# 改成：   dataset='./decoder_calib/dataset.txt'

python convert.py \
    whisper_decoder.onnx \
    rk3588 \
    i8 \
    whisper_decoder_int8.rknn
```

## 参数说明

### 音频长度
- **20秒音频**: `--seq_len 1000`
- **30秒音频**: `--seq_len 1500`

### 模型大小
- **Whisper-tiny**: `--hidden_dim 512`
- **Whisper-base**: `--hidden_dim 768`
- **Whisper-small**: `--hidden_dim 1024`

### 校准样本数量
- 推荐: 200-500 个样本
- 最少: 100 个样本
- 更多样本 = 更好的量化效果，但需要更长时间

## 验证结果

```bash
# 测试量化后的模型
cd examples/whisper/cpp/build

./rknn_whisper_demo \
    ../model/whisper_encoder.rknn \
    ../model/whisper_decoder_int8.rknn \
    en \
    /path/to/test.flac
```

## 文件结构

生成的文件：
```
/tmp/whisper_dumps/          # encoder 输出 (bin 格式)
├── enc_000000.bin
├── enc_000001.bin
└── ...

decoder_calib/               # decoder 校准数据集
├── tokens_000000.npy        # [1, 12] INT64
├── audio_000000.npy         # [1, 1000, 512] FP16
├── tokens_000001.npy
├── audio_000001.npy
├── ...
└── dataset.txt              # 文件列表

model/
├── whisper_encoder.rknn     # 原始 encoder
├── whisper_decoder.onnx     # decoder ONNX
└── whisper_decoder_int8.rknn # 量化后的 decoder
```

## 故障排除

### 问题: 没有生成 enc_*.bin 文件

**解决方案**:
1. 检查 `whisper.cc` 中的 `dump_dir` 路径是否正确
2. 确保目录有写权限
3. 重新编译: `cd build && cmake .. && make`

### 问题: bin 文件大小不对

**检查**:
```bash
# 20s 音频: 应该是 2048000 字节 (1000 * 512 * 4)
ls -lh /tmp/whisper_dumps/enc_*.bin | head

# 如果不对，检查 ENCODER_OUTPUT_SIZE
grep ENCODER_OUTPUT_SIZE cpp/process.h
```

### 问题: 量化失败

**解决方案**:
1. 确保 `dataset.txt` 中的路径是绝对路径
2. 确保所有 .npy 文件都存在
3. 检查 RKNN toolkit 是否安装正确

### 问题: 量化后精度下降严重

**解决方案**:
1. 增加校准样本数量 (500-1000)
2. 确保使用真实 encoder 输出，而非随机数据
3. 使用更多样化的音频数据

## 相关脚本

- `quantize_decoder_e2e.sh`: 端到端自动化脚本
- `batch_generate_encoder_outputs.sh`: 批量生成 encoder 输出
- `bin_to_decoder_dataset.py`: 转换 bin 为 npy
- `dump_decoder_calib.cpp`: 独立生成工具（使用随机数据）

## 需要帮助？

查看详细文档：
```bash
cat DECODER_QUANTIZATION_GUIDE.md
```
