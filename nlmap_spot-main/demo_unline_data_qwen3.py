#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLMap + Qwen3 离线数据演示脚本
使用 unline_data 中的数据进行复现，结合 Qwen3 进行物品提议
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlmap import NLMap
from saycan_qwen3 import Qwen3ObjectProposer

def main():
    print("=" * 60)
    print("NLMap + Qwen3 离线数据演示")
    print("=" * 60)
    
    # 配置文件路径
    config_path = "./configs/unline_data_config.ini"
    
    print(f"\n1. 加载配置文件: {config_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在！")
        return
    
    # 检查数据目录是否存在
    data_dir = "./unline_data/cit121_115"
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 不存在！")
        return
    
    print(f"\n2. 数据目录检查通过: {data_dir}")
    print(f"   - 包含 {len([f for f in os.listdir(data_dir) if f.startswith('color_')])} 张彩色图像")
    print(f"   - 包含 {len([f for f in os.listdir(data_dir) if f.startswith('depth_')])} 个深度文件")
    
    try:
        # 初始化 NLMap
        print("\n3. 初始化 NLMap...")
        nlmap = NLMap(config_path=config_path)
        print("   ✓ NLMap 初始化成功")
        
        # 初始化 Qwen3 对象提议器
        print("\n4. 初始化 Qwen3 对象提议器...")
        qwen3_proposer = Qwen3ObjectProposer()
        print("   ✓ Qwen3 对象提议器初始化成功")
        
        # 测试任务列表
        test_tasks = [
            "帮我准备咖啡",
            "清理桌子",
            "整理办公室",
            "准备会议"
        ]
        
        print("\n5. 开始任务演示...")
        print("=" * 40)
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n任务 {i}: {task}")
            print("-" * 30)
            
            # 使用 Qwen3 提议相关对象
            print("\n5.1 Qwen3 对象提议:")
            try:
                proposed_objects = qwen3_proposer.query_llm_for_objects(task)
                print(f"   提议的对象: {', '.join(proposed_objects)}")
                
                # 生成分阶段规划
                planning_text = qwen3_proposer.generate_planning_text(task, proposed_objects)
                print(f"\n5.2 分阶段规划:")
                print(f"   {planning_text}")
                
            except Exception as e:
                print(f"   Qwen3 处理出错: {e}")
                proposed_objects = ["cup", "coffee machine", "water"]
                print(f"   使用默认对象: {', '.join(proposed_objects)}")
            
            # 在 NLMap 中查询前3个对象
            print("\n5.3 NLMap 物品查询:")
            for obj in proposed_objects[:3]:  # 只查询前3个对象
                try:
                    print(f"   查询对象: {obj}")
                    # 可视化查询结果
                    nlmap.viz_top_k(obj, viz_2d=True, viz_pointcloud=False)
                    print(f"   ✓ {obj} 查询完成")
                except Exception as e:
                    print(f"   ✗ {obj} 查询失败: {e}")
            
            print("\n" + "-" * 30)
        
        print("\n6. 演示完成！")
        print("\n说明:")
        print("   - 图像结果保存在 ./figs 目录")
        print("   - 缓存文件保存在 ./cache 目录")
        print("   - 可以查看生成的可视化结果")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查 CLIP 和 ViLD 模型是否正确安装")
        print("2. 检查数据文件是否完整")
        print("3. 检查 Qwen3 模型路径是否正确")
        print("4. 查看详细错误信息进行调试")

if __name__ == "__main__":
    main()