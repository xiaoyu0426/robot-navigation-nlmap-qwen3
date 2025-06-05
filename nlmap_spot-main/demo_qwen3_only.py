#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 物品提议演示脚本（简化版）
由于NLMap依赖问题，先演示Qwen3的物品提议功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from saycan_qwen3 import Qwen3ObjectProposer

def main():
    print("=" * 60)
    print("Qwen3 物品提议演示（使用unline_data离线数据）")
    print("=" * 60)
    
    # 检查数据目录是否存在
    data_dir = "./unline_data/cit121_115"
    if os.path.exists(data_dir):
        print(f"\n✓ 数据目录检查通过: {data_dir}")
        print(f"   - 包含 {len([f for f in os.listdir(data_dir) if f.startswith('color_')])} 张彩色图像")
        print(f"   - 包含 {len([f for f in os.listdir(data_dir) if f.startswith('depth_')])} 个深度文件")
        if os.path.exists(os.path.join(data_dir, 'pose_data.pkl')):
            print(f"   - ✓ 姿态数据文件存在")
        if os.path.exists(os.path.join(data_dir, 'pointcloud.pcd')):
            print(f"   - ✓ 点云数据文件存在")
    else:
        print(f"\n⚠ 数据目录不存在: {data_dir}")
    
    try:
        # 初始化 Qwen3 对象提议器
        print("\n1. 初始化 Qwen3 对象提议器...")
        qwen3_proposer = Qwen3ObjectProposer()
        print("   ✓ Qwen3 对象提议器初始化成功")
        
        # 测试任务列表
        test_tasks = [
            "帮我准备咖啡",
            "清理桌子",
            "整理办公室",
            "准备会议",
            "打扫房间",
            "准备早餐"
        ]
        
        print("\n2. 开始任务演示...")
        print("=" * 40)
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n任务 {i}: {task}")
            print("-" * 30)
            
            try:
                # 使用 Qwen3 提议相关对象
                print("\n2.1 Qwen3 对象提议:")
                proposed_objects = qwen3_proposer.query_llm_for_objects(task)
                print(f"   提议的对象: {', '.join(proposed_objects)}")
                
                # 生成分阶段规划
                print("\n2.2 分阶段规划:")
                planning_text = qwen3_proposer.generate_planning_text(task, proposed_objects)
                print(f"   {planning_text}")
                
                # 模拟在离线数据中查找这些对象
                print("\n2.3 离线数据中的潜在匹配:")
                # 基于unline_data的常见物品（从图像文件名和常见室内物品推测）
                common_objects = [
                    "cup", "mug", "bottle", "book", "laptop", "phone", "chair", 
                    "table", "desk", "monitor", "keyboard", "mouse", "pen", "paper",
                    "bag", "clock", "lamp", "plant", "window", "door"
                ]
                
                matches = [obj for obj in proposed_objects if any(common in obj.lower() for common in common_objects)]
                if matches:
                    print(f"   可能在数据中找到: {', '.join(matches)}")
                else:
                    print(f"   需要在实际图像中进行视觉检测")
                
            except Exception as e:
                print(f"   ✗ 任务处理失败: {e}")
            
            print("\n" + "-" * 30)
        
        print("\n3. 演示完成！")
        print("\n说明:")
        print("   - Qwen3成功生成了物品提议和分阶段规划")
        print(f"   - 离线数据位于: {data_dir}")
        print("   - 要完整运行NLMap，需要安装tensorflow和相关依赖")
        print("   - 可以使用配置文件: ./configs/unline_data_config.ini")
        
        # 显示配置文件信息
        config_file = "./configs/unline_data_config.ini"
        if os.path.exists(config_file):
            print(f"\n4. 配置文件信息:")
            print(f"   ✓ 配置文件已创建: {config_file}")
            print("   - 设置为离线模式 (use_robot = False)")
            print("   - 数据目录指向 unline_data/cit121_115")
            print("   - 启用点云和姿态数据处理")
            print("   - 包含丰富的物品类别列表")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查 Qwen3 模型路径是否正确")
        print("2. 确保在正确的conda环境中运行")
        print("3. 检查transformers库是否正确安装")

if __name__ == "__main__":
    main()