#!/usr/bin/env python3
"""
验证 Whisper Encoder 和 Decoder 模型是否正常工作

使用方法:
  PC 上:   python verify_models.py --check-only    (只检查文件，不推理)
  板上:    python verify_models.py                  (实际推理测试)
"""

import numpy as np
import sys
import os
import argparse
import platform

# 检查是否在 ARM 板上
def is_arm_board():
    machine = platform.machine().lower()
    return 'arm' in machine or 'aarch64' in machine

# 只依赖 rknn-toolkit-lite (RKNNLite) 在板上做推理验证
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN_LITE = True
    print("✅ 找到 rknn-toolkit-lite (RKNNLite) — 板上推理可用")
except ImportError:
    HAS_RKNN_LITE = False
    print("⚠️ 未找到 rknn-toolkit-lite (RKNNLite)。如果要在开发板上运行推理，请安装 rknnlite。")


def check_model_file(model_path, model_name):
    """只检查模型文件（不推理）"""
    print(f"\n检查 {model_name} 模型文件...")
    
    if not os.path.exists(model_path):
        print(f"  ❌ 文件不存在: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"  📁 文件路径: {model_path}")
    print(f"  📊 文件大小: {file_size_mb:.2f} MB")
    
    # 检查文件大小是否合理
    if file_size < 1024 * 1024:  # < 1MB
        print(f"  ⚠️  警告: 文件太小 ({file_size_mb:.2f} MB)，可能损坏")
        return False
    
    # 读取文件头检查是否是 RKNN 格式
    try:
        with open(model_path, 'rb') as f:
            header = f.read(4)
            if header[:4] == b'RKNN':
                print(f"  ✅ 文件格式正确 (RKNN)")
            else:
                print(f"  ⚠️  警告: 文件头不是 RKNN 格式")
                return False
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return False
    
    print(f"  ✅ {model_name} 文件检查通过")
    return True


def test_encoder(model_path, check_only=False):
    """测试 Encoder 模型"""
    print("\n" + "="*60)
    print("测试 Encoder 模型")
    print("="*60)
    
    if check_only:
        return check_model_file(model_path, "Encoder")

    if not HAS_RKNN_LITE:
        print("❌ 未安装 rknn-toolkit-lite (RKNNLite)。无法在板上做推理验证。")
        print("   请在开发板的 Python 环境中安装: pip install rknnlite")
        print("   或使用: python verify_models.py --check-only 在 PC 上只检查文件")
        return False

    # 仅在 ARM 开发板上使用 RKNNLite 做推理验证
    use_lite = HAS_RKNN_LITE and is_arm_board()
    print(f"使用 {'RKNNLite' if use_lite else 'RKNN'} 进行推理测试")
    
    if not use_lite and not is_arm_board():
        print("⚠️  检测到非 ARM 环境 — 跳过推理或请在开发板上运行（使用 --check-only 在 PC 上仅检查文件）")

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    print(f"加载模型: {model_path}")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=True, verbose_file='./rknn_lite_debug_encoder.log')

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"❌ 加载模型失败: {ret}")
        return False
    
    ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_AUTO)
   
    if ret != 0:
        print(f"❌ 初始化失败: {ret}")
        rknn.release()
        return False
 
    print("使用 RKNNLite (板上 NPU 推理)")
    
    # 查询模型输入输出信息
    try:
        from rknnlite.api import RKNNLite as RKNN_QUERY
        input_attrs = rknn.query(RKNN_QUERY.INPUT_ATTR)
        output_attrs = rknn.query(RKNN_QUERY.OUTPUT_ATTR)
        
        print("\n📋 模型输入信息:")
        if input_attrs and len(input_attrs) > 0:
            for i, attr in enumerate(input_attrs):
                print(f"  输入 {i}: shape={attr['dims']}, dtype={attr.get('dtype', 'unknown')}, "
                      f"format={attr.get('fmt', 'unknown')}")
        
        print("\n📋 模型输出信息:")
        if output_attrs and len(output_attrs) > 0:
            for i, attr in enumerate(output_attrs):
                print(f"  输出 {i}: shape={attr['dims']}, dtype={attr.get('dtype', 'unknown')}, "
                      f"format={attr.get('fmt', 'unknown')}")
    except Exception as e:
        print(f"\n⚠️  无法查询模型信息: {e}")
   
    # 创建模拟输入 (mel features: 1 x 80 x 3000)
    print("\n创建测试输入 (1, 80, 3000) - 模拟 30 秒音频的 mel 特征")
    mel_input = np.random.randn(1, 80, 3000).astype(np.float32)
    print(f"  输入形状: {mel_input.shape}")
    print(f"  输入范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
    print(f"  输入均值: {mel_input.mean():.6f}")
    
    print("\n执行推理...")
    outputs = rknn.inference(inputs=[mel_input])
    
    if outputs is None or len(outputs) == 0:
        print("❌ 推理失败: 没有输出")
        rknn.release()
        return False
    
    encoder_out = outputs[0]
    
    print("\n📊 Encoder 输出分析:")
    print(f"  输出形状: {encoder_out.shape}")
    expected_shape = (1, 1500, 512)  # 30秒音频
    if encoder_out.shape != expected_shape:
        print(f"  ⚠️  警告: 预期形状 {expected_shape}, 实际 {encoder_out.shape}")
    
    print(f"  数据类型: {encoder_out.dtype}")
    print(f"  值范围: [{encoder_out.min():.6f}, {encoder_out.max():.6f}]")
    print(f"  均值: {encoder_out.mean():.6f}")
    print(f"  标准差: {encoder_out.std():.6f}")
    
    # 检查前 10 个值
    flat = encoder_out.flatten()
    print(f"  前 10 个值: {flat[:10]}")
    
    # 检查是否异常
    issues = []
    
    if np.all(encoder_out == 0):
        issues.append("❌ 所有输出值都是 0")
    
    if np.isnan(encoder_out).any():
        issues.append(f"❌ 包含 NaN 值: {np.isnan(encoder_out).sum()} 个")
    
    if np.isinf(encoder_out).any():
        issues.append(f"❌ 包含 Inf 值: {np.isinf(encoder_out).sum()} 个")
    
    if abs(encoder_out.mean()) > 100:
        issues.append(f"⚠️  均值过大: {encoder_out.mean():.3f}")
    
    if encoder_out.std() < 0.01:
        issues.append(f"⚠️  标准差过小 (可能量化失败): {encoder_out.std():.6f}")
    
    if issues:
        print("\n问题检测:")
        for issue in issues:
            print(f"  {issue}")
        result = False
    else:
        print("\n✅ Encoder 输出看起来正常")
        result = True
    
    rknn.release()
    return result


def test_decoder(model_path, check_only=False):
    """测试 Decoder 模型"""
    print("\n" + "="*60)
    print("测试 Decoder 模型")
    print("="*60)
    
    if check_only:
        return check_model_file(model_path, "Decoder")

    if not HAS_RKNN_LITE:
        print("❌ 未安装 rknn-toolkit-lite (RKNNLite)。无法在板上做推理验证。")
        print("   请在开发板的 Python 环境中安装: pip install rknnlite")
        print("   或使用: python verify_models.py --check-only 在 PC 上只检查文件")
        return False

    # 仅在 ARM 开发板上使用 RKNNLite 做推理验证
    use_lite = HAS_RKNN_LITE and is_arm_board()

    if not use_lite and not is_arm_board():
        print("⚠️  检测到非 ARM 环境 — 跳过推理或请在开发板上运行（使用 --check-only 在 PC 上仅检查文件）")

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    print(f"加载模型: {model_path}")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=True, verbose_file='./rknn_lite_debug_decoder.log')

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"❌ 加载模型失败: {ret}")
        return False
    

    ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_AUTO)

    if ret != 0:
        print(f"❌ 初始化失败: {ret}")
        rknn.release()
        return False
 
    print("使用 RKNNLite (板上 NPU 推理)")
    
    # 查询模型输入输出信息
    try:
        from rknnlite.api import RKNNLite as RKNN_QUERY
        input_attrs = rknn.query(RKNN_QUERY.INPUT_ATTR)
        output_attrs = rknn.query(RKNN_QUERY.OUTPUT_ATTR)
        
        print("\n📋 模型输入信息:")
        if input_attrs and len(input_attrs) > 0:
            for i, attr in enumerate(input_attrs):
                print(f"  输入 {i}: shape={attr['dims']}, dtype={attr.get('dtype', 'unknown')}, "
                      f"format={attr.get('fmt', 'unknown')}")
        
        print("\n📋 模型输出信息:")
        if output_attrs and len(output_attrs) > 0:
            for i, attr in enumerate(output_attrs):
                print(f"  输出 {i}: shape={attr['dims']}, dtype={attr.get('dtype', 'unknown')}, "
                      f"format={attr.get('fmt', 'unknown')}")
    except Exception as e:
        print(f"\n⚠️  无法查询模型信息: {e}")
       
   
    # 创建模拟输入
    print("\n创建测试输入:")
    print("  Encoder 输出: (1, 1000, 512)")
    encoder_out = np.random.randn(1, 1000, 512).astype(np.float32)
    
    print("  Token 序列: (1, 4) - [50258, 50259, 50360, 1220]")
    # 50258: <|startoftranscript|>
    # 50259: <|en|>
    # 50360: <|transcribe|>
    # 1220: 随机 token
    tokens = np.array([[50258, 50259, 50360, 1220]], dtype=np.int32)
    
    print("\n执行推理...")
    try:
        outputs = rknn.inference(inputs=[encoder_out, tokens])
    except Exception as e:
        print(f"❌ 推理时发生异常: {e}")
        print(f"   异常类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        rknn.release()
        return False
    
    if outputs is None or len(outputs) == 0:
        print("❌ 推理失败: 没有输出")
        rknn.release()
        return False
    
    logits = outputs[0]
    
    print("\n📊 Decoder 输出分析:")
    print(f"  输出形状: {logits.shape}")
    expected_shape = (1, 4, 51865)  # vocab size
    if logits.shape != expected_shape:
        print(f"  ⚠️  警告: 预期形状 {expected_shape}, 实际 {logits.shape}")
    
    print(f"  数据类型: {logits.dtype}")
    print(f"  Logits 范围: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # 分析最后一个 token 的预测
    last_logits = logits[0, -1, :]
    top5_indices = np.argsort(last_logits)[-5:][::-1]
    top5_values = last_logits[top5_indices]
    
    print(f"\n  最后一个 token 位置的 Top-5 预测:")
    for i, (idx, val) in enumerate(zip(top5_indices, top5_values), 1):
        print(f"    {i}. Token {idx}: logit={val:.3f}")
    
    # 检查是否异常
    issues = []
    
    if np.all(logits == 0):
        issues.append("❌ 所有 logits 都是 0")
    
    if np.isnan(logits).any():
        issues.append(f"❌ 包含 NaN 值: {np.isnan(logits).sum()} 个")
    
    if np.isinf(logits).any():
        issues.append(f"❌ 包含 Inf 值: {np.isinf(logits).sum()} 个")
    
    # 检查是否总是预测 EOS
    EOS_TOKEN = 50257
    if top5_indices[0] == EOS_TOKEN:
        issues.append(f"⚠️  最高概率是 EOS token ({EOS_TOKEN}) - 可能导致空输出")
    
    # 检查概率分布是否太平坦
    if logits.std() < 0.1:
        issues.append(f"⚠️  Logits 标准差过小 (分布太平坦): {logits.std():.6f}")
    
    if issues:
        print("\n问题检测:")
        for issue in issues:
            print(f"  {issue}")
        result = False
    else:
        print("\n✅ Decoder 输出看起来正常")
        result = True
    
    rknn.release()
    return result


def main():
    parser = argparse.ArgumentParser(
        description='验证 Whisper RKNN 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # PC 上只检查文件（推荐）
  python verify_models.py --check-only
  
  # 开发板上实际推理测试
  python verify_models.py
  
  # 指定模型路径
  python verify_models.py --encoder model/encoder.rknn --decoder model/decoder.rknn --check-only
        """
    )
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查模型文件，不执行推理（PC 上使用）')
    parser.add_argument('--encoder', default='model/whisper_encoder_base_20s_i8_2.rknn',
                        help='Encoder 模型路径')
    parser.add_argument('--decoder', default='model/whisper_decoder_base_20s_i8.rknn',
                        help='Decoder 模型路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Whisper 模型验证工具")
    print("="*60)
    
    if args.check_only:
        print("\n模式: 仅检查文件（不推理）")
    else:
        print("\n模式: 完整测试（推理验证）")
        if not is_arm_board():
            print("\n⚠️  检测到非 ARM 环境")
            print("RKNN 模型只能在开发板上推理")
            print("建议使用: python verify_models.py --check-only")
            response = input("\n继续尝试推理? (y/N): ")
            if response.lower() != 'y':
                print("已取消")
                return 0
    
    if not HAS_RKNN_LITE and not args.check_only:
        print("\n❌ 错误: 未安装 rknn-toolkit-lite (RKNNLite)")
        print("请在开发板的 Python 环境中安装: pip install rknnlite")
        print("或在 PC 上使用: python verify_models.py --check-only 来仅检查模型文件")
        return 1
    
    # 测试 Encoder
    encoder_ok = test_encoder(args.encoder, args.check_only)
    
    # 测试 Decoder
    decoder_ok = test_decoder(args.decoder, args.check_only)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"Encoder: {'✅ 通过' if encoder_ok else '❌ 失败'}")
    print(f"Decoder: {'✅ 通过' if decoder_ok else '❌ 失败'}")
    
    if not encoder_ok:
        print("\n🔍 Encoder 问题可能原因:")
        print("  1. 量化配置不当（mean_values, std_values 错误）")
        print("  2. 输入数据预处理问题")
        print("  3. 模型转换时出错")
        print("\n建议:")
        print("  - 检查量化时的 mean_values 和 std_values")
        print("  - 尝试使用 FP16 或混合量化")
        print("  - 增加校准数据的多样性")
    
    if not decoder_ok:
        print("\n🔍 Decoder 问题可能原因:")
        print("  1. 校准数据不正确（encoder 输出质量差）")
        print("  2. Token embedding 量化失败")
        print("  3. 词汇表或 token 配置错误")
        print("\n建议:")
        print("  - 确认 encoder_dumps/*.bin 文件正常")
        print("  - 检查 bin_to_decoder_dataset.py 转换是否正确")
        print("  - 验证 decoder 量化配置")
    
    if encoder_ok and decoder_ok:
        print("\n✅ 两个模型都通过了基本验证")
        print("\n如果实际推理仍输出为空，检查:")
        print("  1. 音频预处理（mel 特征提取）")
        print("  2. Token 解码逻辑")
        print("  3. 词汇表加载")
        print("  4. 后处理步骤")
    
    print("\n" + "="*60)
    
    return 0 if (encoder_ok and decoder_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
