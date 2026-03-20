#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线NLMap + Qwen3物品提议演示
从离线数据集中获取物品清单与实际位置，传递给交互界面由Qwen3提议
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import configparser
import json
from typing import List, Dict, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from saycan_qwen3 import Qwen3ObjectProposer

class OfflineNLMapDataExtractor:
    """从离线数据集中提取物品清单和位置信息"""
    
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # 获取数据目录
        # 从配置文件顶部读取data值（在DEFAULT section中）
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data_name = None
        for line in lines:
            if line.strip().startswith('data ='):
                data_name = line.split('=')[1].strip()
                break
        
        if not data_name:
            data_name = 'cit121_115'  # 默认值
        
        data_dir_root = self.config.get('paths', 'data_dir_root')
        self.data_dir = os.path.join(data_dir_root, data_name)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 获取物品类别列表
        category_string = self.config.get('text', 'category_name_string')
        self.categories = [cat.strip() for cat in category_string.split(';')]
        
        print(f"数据目录: {self.data_dir}")
        print(f"物品类别: {self.categories}")
    
    def load_pose_data(self) -> Dict:
        """加载姿态数据"""
        pose_file = os.path.join(self.data_dir, 'pose_data.pkl')
        if os.path.exists(pose_file):
            with open(pose_file, 'rb') as f:
                pose_data = pickle.load(f)
            print(f"✓ 加载姿态数据: {len(pose_data)} 个姿态")
            return pose_data
        else:
            print("⚠ 姿态数据文件不存在")
            return {}
    
    def scan_image_files(self) -> List[str]:
        """扫描图像文件"""
        color_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith('color_') and file.endswith('.jpg'):
                color_files.append(file)
        color_files.sort()
        print(f"✓ 发现 {len(color_files)} 个彩色图像文件")
        return color_files
    
    def extract_object_inventory(self) -> Dict[str, Dict]:
        """提取物品清单和模拟位置信息"""
        pose_data = self.load_pose_data()
        image_files = self.scan_image_files()
        
        # 模拟物品检测结果（实际应该通过VILD/CLIP进行检测）
        object_inventory = {}
        
        # 为每个类别生成模拟的检测结果
        for i, category in enumerate(self.categories):
            if i < len(image_files):  # 确保有对应的图像
                image_file = image_files[i % len(image_files)]
                image_id = image_file.replace('color_', '').replace('.jpg', '')
                
                # 模拟3D位置（基于姿态数据或随机生成）
                if image_id in pose_data:
                    # 使用真实姿态数据
                    pose_info = pose_data[image_id]
                    position = pose_info.get('position', [0, 0, 0])
                    # 添加一些随机偏移来模拟物体相对于相机的位置
                    object_position = [
                        position[0] + np.random.uniform(-0.5, 0.5),
                        position[1] + np.random.uniform(-0.5, 0.5),
                        position[2] + np.random.uniform(-0.2, 0.2)
                    ]
                else:
                    # 生成模拟位置
                    object_position = [
                        np.random.uniform(-2, 2),  # x
                        np.random.uniform(-2, 2),  # y
                        np.random.uniform(0, 1.5)  # z
                    ]
                
                object_inventory[category] = {
                    'position': object_position,
                    'image_file': image_file,
                    'image_id': image_id,
                    'confidence': np.random.uniform(0.7, 0.95),
                    'bounding_box': {
                        'x': np.random.randint(50, 300),
                        'y': np.random.randint(50, 300),
                        'width': np.random.randint(80, 200),
                        'height': np.random.randint(80, 200)
                    }
                }
        
        print(f"✓ 生成物品清单: {len(object_inventory)} 个物品")
        return object_inventory
    
    def save_inventory_to_json(self, inventory: Dict, output_file: str):
        """保存物品清单到JSON文件"""
        # 转换numpy类型为Python原生类型
        json_inventory = {}
        for obj_name, obj_data in inventory.items():
            json_inventory[obj_name] = {
                'position': [float(x) for x in obj_data['position']],
                'image_file': obj_data['image_file'],
                'image_id': obj_data['image_id'],
                'confidence': float(obj_data['confidence']),
                'bounding_box': {
                    'x': int(obj_data['bounding_box']['x']),
                    'y': int(obj_data['bounding_box']['y']),
                    'width': int(obj_data['bounding_box']['width']),
                    'height': int(obj_data['bounding_box']['height'])
                }
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_inventory, f, ensure_ascii=False, indent=2)
        print(f"✓ 物品清单已保存到: {output_file}")

class InteractiveQwen3Interface:
    """交互式Qwen3物品提议界面"""
    
    def __init__(self, inventory_file: str):
        self.qwen3_proposer = Qwen3ObjectProposer()
        
        # 加载物品清单
        with open(inventory_file, 'r', encoding='utf-8') as f:
            self.object_inventory = json.load(f)
        
        self.available_objects = list(self.object_inventory.keys())
        print(f"✓ 加载物品清单: {len(self.available_objects)} 个物品")
        print(f"可用物品: {', '.join(self.available_objects)}")
    
    def find_matching_objects(self, proposed_objects: List[str]) -> Dict[str, Dict]:
        """在物品清单中查找匹配的物品"""
        matches = {}
        
        for proposed in proposed_objects:
            # 精确匹配
            if proposed in self.object_inventory:
                matches[proposed] = self.object_inventory[proposed]
                continue
            
            # 模糊匹配
            for available in self.available_objects:
                if (proposed.lower() in available.lower() or 
                    available.lower() in proposed.lower()):
                    matches[proposed] = self.object_inventory[available]
                    matches[proposed]['matched_name'] = available
                    break
        
        return matches
    
    def display_object_details(self, obj_name: str, obj_data: Dict):
        """显示物品详细信息"""
        print(f"\n  📍 {obj_name}:")
        print(f"     位置: ({obj_data['position'][0]:.2f}, {obj_data['position'][1]:.2f}, {obj_data['position'][2]:.2f})")
        print(f"     置信度: {obj_data['confidence']:.2f}")
        print(f"     图像文件: {obj_data['image_file']}")
        if 'matched_name' in obj_data:
            print(f"     匹配名称: {obj_data['matched_name']}")
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n" + "=" * 60)
        print("🤖 NLMap + Qwen3 交互式物品提议系统")
        print("=" * 60)
        
        while True:
            print("\n请输入任务描述 (输入 'quit' 退出, 'list' 查看可用物品):")
            task = input("> ").strip()
            
            if task.lower() == 'quit':
                print("👋 再见！")
                break
            
            if task.lower() == 'list':
                print(f"\n📦 可用物品清单 ({len(self.available_objects)} 个):")
                for i, obj in enumerate(self.available_objects, 1):
                    obj_data = self.object_inventory[obj]
                    print(f"  {i:2d}. {obj} (位置: {obj_data['position'][0]:.1f}, {obj_data['position'][1]:.1f}, {obj_data['position'][2]:.1f})")
                continue
            
            if not task:
                continue
            
            try:
                print(f"\n🔍 处理任务: {task}")
                print("-" * 40)
                
                # Qwen3物品提议
                print("\n1. Qwen3 物品提议:")
                proposed_objects = self.qwen3_proposer.query_llm_for_objects(task)
                print(f"   提议的物品: {', '.join(proposed_objects)}")
                
                # 在离线数据中查找匹配
                print("\n2. 在离线数据中查找匹配:")
                matches = self.find_matching_objects(proposed_objects)
                
                if matches:
                    print(f"   ✓ 找到 {len(matches)} 个匹配物品:")
                    for obj_name, obj_data in matches.items():
                        self.display_object_details(obj_name, obj_data)
                else:
                    print("   ⚠ 未找到匹配的物品")
                
                # 生成分阶段规划
                print("\n3. 分阶段规划:")
                available_objects = list(matches.keys()) if matches else []
                planning_text = self.qwen3_proposer.generate_planning_text(task, available_objects)
                print(f"   {planning_text}")
                
                # 显示可执行的操作
                if matches:
                    print("\n4. 可执行的操作:")
                    for obj_name, obj_data in matches.items():
                        pos = obj_data['position']
                        print(f"   • 导航到 {obj_name}: 目标位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                        print(f"   • 抓取 {obj_name}: 在图像 {obj_data['image_file']} 中检测到")
                
            except Exception as e:
                print(f"   ✗ 处理失败: {e}")
            
            print("\n" + "-" * 40)

def main():
    print("🚀 启动离线NLMap + Qwen3演示系统")
    print("=" * 50)
    
    # 配置文件路径
    config_file = "./configs/unline_data_config.ini"
    inventory_file = "./offline_object_inventory.json"
    
    try:
        # 1. 从离线数据集提取物品清单
        print("\n📊 步骤1: 从离线数据集提取物品清单")
        extractor = OfflineNLMapDataExtractor(config_file)
        inventory = extractor.extract_object_inventory()
        extractor.save_inventory_to_json(inventory, inventory_file)
        
        # 2. 启动交互界面
        print("\n🎮 步骤2: 启动交互式Qwen3界面")
        interface = InteractiveQwen3Interface(inventory_file)
        interface.interactive_demo()
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("\n💡 解决方案:")
        print("1. 确保配置文件存在: ./configs/unline_data_config.ini")
        print("2. 确保数据目录存在: ./unline_data/cit121_115")
        print("3. 检查数据目录中是否包含必要的文件")
    
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查Qwen3模型是否正确加载")
        print("2. 确保在正确的conda环境中运行")
        print("3. 检查所有依赖是否正确安装")

if __name__ == "__main__":
    main()