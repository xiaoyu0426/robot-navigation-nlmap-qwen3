#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-4B 本地部署测试脚本
验证模型是否能够成功加载和运行
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def test_qwen3_deployment():
    """
    测试Qwen3-4B模型的本地部署
    """
    print("=== Qwen3-4B 本地部署测试 ===")
    
    # 检查CUDA可用性
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
    
    # 模型路径 (从环境变量或默认路径获取)
    model_path = os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-models")
    
    print(f"\n加载模型路径: {model_path}")
    
    try:
        # 加载tokenizer
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
        
        # 加载模型
        print("正在加载模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print(f"✓ 模型加载成功，运行设备: {device}")
        
        # 测试推理
        print("\n开始测试推理...")
        test_prompt = "你好，请介绍一下你自己。"
        
        # 编码输入
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n测试输入: {test_prompt}")
        print(f"模型回复: {response[len(test_prompt):].strip()}")
        
        print("\n✓ Qwen3-4B 本地部署测试成功！")
        return True
        
    except Exception as e:
        print(f"\n✗ 模型部署测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_qwen3_deployment()
    if success:
        print("\n🎉 Qwen3-4B 已成功部署并可以正常使用！")
    else:
        print("\n❌ Qwen3-4B 部署存在问题，请检查模型文件和依赖。")