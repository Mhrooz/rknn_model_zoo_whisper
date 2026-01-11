#!/usr/bin/env python3
"""
验证 Whisper Encoder 和 Decoder 模型是否正常工作
"""

import numpy as np
import sys
import os

# 检查是否有 rknn-toolkit2
try:
    from rknn.api import RKNN
    print("✅ rknn-toolkit2 已安装")
except ImportError:
    print("❌ 错误: 需要安装 rknn-toolkit2")
    print("   pip install rknn-toolkit2")
    sys.exit(1)


def test_encoder(model_path):
    """测试 Encoder 模型"""
    print("\n" + "="*60)
    print("测试 Encoder 模型")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"加载模型: {model_path}")
    rknn = RKNN(verbose=False)
    
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"❌ 加载模型失败: {ret}")
        return False
    
    print("初始化运行时环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"❌ 初始化失败: {ret}")
        rknn.release()
        return False
    
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


def test_decoder(model_path):
    """测试 Decoder 模型"""
    print("\n" + "="*60)
    print("测试 Decoder 模型")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"加载模型: {model_path}")
    rknn = RKNN(verbose=False)
    
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"❌ 加载模型失败: {ret}")
        return False
    
    print("初始化运行时环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"❌ 初始化失败: {ret}")
        rknn.release()
        return False
    
    # 创建模拟输入
    print("\n创建测试输入:")
    print("  Encoder 输出: (1, 1500, 512)")
    encoder_out = np.random.randn(1, 1500, 512).astype(np.float32)
    
    print("  Token 序列: (1, 4) - [50258, 50259, 50360, 1220]")
    # 50258: <|startoftranscript|>
    # 50259: <|en|>
    # 50360: <|transcribe|>
    # 1220: 随机 token
    tokens = np.array([[50258, 50259, 50360, 1220]], dtype=np.int32)
    
    print("\n执行推理...")
    outputs = rknn.inference(inputs=[encoder_out, tokens])
    
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
    print("="*60)
    print("Whisper 模型验证工具")
    print("="*60)
    
    # 模型路径
    encoder_path = "model/whisper_encoder_base_i8_2.rknn"
    decoder_path = "model/whisper_decoder_base_i8.rknn"
    
    # 测试 Encoder
    encoder_ok = test_encoder(encoder_path)
    
    # 测试 Decoder
    decoder_ok = test_decoder(decoder_path)
    
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
