#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLMap + Qwen3 交互式演示界面
基于 Streamlit 的简易初版实现
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime

# 添加项目路径
sys.path.append('./nlmap_spot-main')

try:
    from saycan_qwen3 import Qwen3ObjectProposer
except ImportError:
    st.error("无法导入 Qwen3ObjectProposer，请确保模型已正确配置")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="NLMap + Qwen3 智能机器人助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 预定义场景配置
SCENE_CONFIGS = {
    "厨房环境": {
        "description": "家庭厨房场景，包含各种烹饪和用餐相关物品",
        "objects": [
            "coffee machine", "cup", "mug", "table", "chair", "apple", "banana",
            "water bottle", "microwave", "fridge", "sink", "sponge", "towel",
            "bread", "butter", "knife", "plate", "bowl", "stove", "pot", "pan",
            "cutting board", "spoon", "fork", "glass", "milk", "sugar", "coffee beans"
        ],
        "sample_tasks": [
            "帮我准备咖啡",
            "清理厨房桌子",
            "准备简单早餐",
            "洗碗收拾餐具",
            "制作水果沙拉"
        ]
    },
    "客厅环境": {
        "description": "家庭客厅场景，包含娱乐和休闲相关物品",
        "objects": [
            "sofa", "tv", "remote control", "coffee table", "lamp", "book",
            "magazine", "cushion", "blanket", "plant", "vase", "picture frame",
            "speaker", "game controller", "laptop", "phone", "charger", "tissue box",
            "candle", "decorative item", "carpet", "curtain"
        ],
        "sample_tasks": [
            "整理客厅桌子",
            "打开电视看新闻",
            "调节客厅灯光",
            "收拾沙发上的物品",
            "给植物浇水"
        ]
    },
    "办公室环境": {
        "description": "办公室工作场景，包含办公和学习相关物品",
        "objects": [
            "desk", "chair", "computer", "keyboard", "mouse", "monitor", "printer",
            "paper", "pen", "pencil", "notebook", "folder", "stapler", "calculator",
            "phone", "lamp", "trash can", "water bottle", "coffee cup", "calendar",
            "whiteboard", "marker", "eraser", "filing cabinet"
        ],
        "sample_tasks": [
            "整理办公桌",
            "打印重要文件",
            "准备会议材料",
            "清空垃圾桶",
            "整理文件夹"
        ]
    },
    "卧室环境": {
        "description": "卧室休息场景，包含睡眠和个人护理相关物品",
        "objects": [
            "bed", "pillow", "blanket", "nightstand", "lamp", "alarm clock",
            "wardrobe", "clothes", "shoes", "mirror", "brush", "towel",
            "book", "phone", "charger", "tissue", "water glass", "curtain",
            "laundry basket", "hangers", "jewelry box", "perfume"
        ],
        "sample_tasks": [
            "整理床铺",
            "收拾衣物",
            "设置闹钟",
            "关闭窗帘",
            "准备睡前用品"
        ]
    }
}

# 初始化会话状态
if 'proposer' not in st.session_state:
    st.session_state.proposer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

def load_model():
    """加载 Qwen3 模型"""
    try:
        with st.spinner('🔄 正在加载 Qwen3 模型，请稍候...'):
            proposer = Qwen3ObjectProposer()
            st.session_state.proposer = proposer
            st.session_state.model_loaded = True
            st.success('✅ Qwen3 模型加载成功！')
            return True
    except Exception as e:
        st.error(f'❌ 模型加载失败: {str(e)}')
        return False

def process_task(task, scene_objects):
    """处理用户任务"""
    if not st.session_state.model_loaded:
        st.error('❌ 模型未加载，请先加载模型')
        return None, None
    
    try:
        proposer = st.session_state.proposer
        
        # 对象提议
        with st.spinner('🤔 正在分析任务并提议相关对象...'):
            start_time = time.time()
            proposed_objects = proposer.query_llm_for_objects(task)
            proposal_time = time.time() - start_time
        
        # 场景匹配
        available_objects = []
        for proposed in proposed_objects:
            for scene_obj in scene_objects:
                if (proposed.lower() in scene_obj.lower() or 
                    scene_obj.lower() in proposed.lower() or
                    any(word in scene_obj.lower() for word in proposed.lower().split())):
                    if scene_obj not in available_objects:
                        available_objects.append(scene_obj)
        
        if not available_objects:
            available_objects = scene_objects[:5]  # 使用前5个对象作为备选
        
        # 生成规划
        with st.spinner('📝 正在生成执行规划...'):
            start_time = time.time()
            planning = proposer.generate_planning_text(task, available_objects)
            planning_time = time.time() - start_time
        
        return {
            'task': task,
            'proposed_objects': proposed_objects,
            'available_objects': available_objects,
            'planning': planning,
            'proposal_time': proposal_time,
            'planning_time': planning_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, None
        
    except Exception as e:
        return None, str(e)

# 主界面
st.title('🤖 NLMap + Qwen3 智能机器人助手')
st.markdown('---')

# 侧边栏 - 模型状态和配置
with st.sidebar:
    st.header('🔧 系统配置')
    
    # 模型状态
    st.subheader('📊 模型状态')
    if st.session_state.model_loaded:
        st.success('✅ Qwen3 模型已加载')
    else:
        st.warning('⚠️ 模型未加载')
        if st.button('🚀 加载模型', type='primary'):
            load_model()
    
    st.markdown('---')
    
    # 场景选择
    st.subheader('🏠 场景选择')
    selected_scene = st.selectbox(
        '选择测试场景:',
        list(SCENE_CONFIGS.keys()),
        index=0
    )
    
    scene_config = SCENE_CONFIGS[selected_scene]
    st.info(f"📝 {scene_config['description']}")
    
    # 显示场景物品数量
    st.metric('场景物品数量', len(scene_config['objects']))
    
    st.markdown('---')
    
    # 快速操作
    st.subheader('⚡ 快速操作')
    
    # 重新部署模型按钮
    if st.button('🔄 重新部署模型'):
        st.session_state.model_loaded = False
        st.session_state.proposer = None
        with st.spinner('🔄 正在重新部署 Qwen3 模型...'):
            if load_model():
                st.success('✅ 模型重新部署成功！')
                st.rerun()
            else:
                st.error('❌ 模型重新部署失败')
    
    if st.button('🗑️ 清空历史记录'):
        st.session_state.results_history = []
        st.success('历史记录已清空')
    
    if st.button('📊 显示统计信息'):
        if st.session_state.results_history:
            avg_proposal_time = sum(r['proposal_time'] for r in st.session_state.results_history) / len(st.session_state.results_history)
            avg_planning_time = sum(r['planning_time'] for r in st.session_state.results_history) / len(st.session_state.results_history)
            st.metric('平均对象提议时间', f'{avg_proposal_time:.2f}秒')
            st.metric('平均规划生成时间', f'{avg_planning_time:.2f}秒')
        else:
            st.info('暂无历史数据')

# 主内容区域
col1, col2 = st.columns([1, 1])

with col1:
    st.header('🎯 任务输入与测试')
    
    # 示例任务
    st.subheader('💡 示例任务')
    sample_tasks = scene_config['sample_tasks']
    
    selected_sample = st.selectbox(
        '选择示例任务或自定义输入:',
        ['自定义输入'] + sample_tasks
    )
    
    # 任务输入
    if selected_sample == '自定义输入':
        user_task = st.text_input(
            '请输入您的任务:',
            placeholder='例如: 帮我准备咖啡',
            key='custom_task'
        )
    else:
        user_task = selected_sample
        st.text_input(
            '当前任务:',
            value=user_task,
            disabled=True,
            key='selected_task'
        )
    
    # 处理按钮
    if st.button('🚀 开始处理', type='primary', disabled=not st.session_state.model_loaded or not user_task):
        if user_task:
            st.session_state.processing = True
            result, error = process_task(user_task, scene_config['objects'])
            
            if result:
                st.session_state.results_history.append(result)
                st.success('✅ 任务处理完成！')
            else:
                st.error(f'❌ 处理失败: {error}')
            
            st.session_state.processing = False

with col2:
    st.header('📋 场景物品列表')
    
    # 物品列表展示
    st.subheader(f'🏠 {selected_scene} - 可用物品')
    
    # 分列显示物品
    objects = scene_config['objects']
    cols = st.columns(3)
    
    for i, obj in enumerate(objects):
        with cols[i % 3]:
            st.write(f'• {obj}')
    
    st.info(f'💡 提示: 当前场景共有 {len(objects)} 个可用物品，您可以基于这些物品来思考和设计任务。')

# 结果展示区域
if st.session_state.results_history:
    st.markdown('---')
    st.header('📊 处理结果')
    
    # 显示最新结果
    latest_result = st.session_state.results_history[-1]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric('对象提议时间', f'{latest_result["proposal_time"]:.2f}秒')
    
    with col2:
        st.metric('规划生成时间', f'{latest_result["planning_time"]:.2f}秒')
    
    with col3:
        st.metric('总处理时间', f'{latest_result["proposal_time"] + latest_result["planning_time"]:.2f}秒')
    
    # 详细结果
    st.subheader('🎯 任务分析结果')
    
    # 任务信息
    st.write(f"**任务**: {latest_result['task']}")
    st.write(f"**处理时间**: {latest_result['timestamp']}")
    
    # 对象提议
    st.subheader('📋 AI提议的相关对象')
    if latest_result['proposed_objects']:
        for i, obj in enumerate(latest_result['proposed_objects'][:10], 1):  # 显示前10个
            st.write(f'{i}. {obj}')
    else:
        st.warning('未找到相关对象')
    
    # 场景匹配
    st.subheader('🔍 场景中的可用对象')
    if latest_result['available_objects']:
        # 显示匹配的对象列表
        for obj in latest_result['available_objects']:
            st.write(f'✅ {obj}')
    else:
        st.warning('场景中未找到匹配对象')
    
    # 执行规划
    st.subheader('📝 AI生成的执行规划')
    planning_text = latest_result['planning']
    if planning_text:
        # 直接显示规划文本，保持原始格式
        planning_lines = planning_text.split('\n')
        
        for line in planning_lines:
            line = line.strip()
            if line:  # 只显示非空行
                st.write(line)
    else:
        st.warning('未生成执行规划')

# 历史记录
if len(st.session_state.results_history) > 1:
    st.markdown('---')
    st.header('📚 历史记录')
    
    # 历史记录表格
    history_data = []
    for i, result in enumerate(reversed(st.session_state.results_history[:-1]), 1):
        history_data.append({
            '序号': i,
            '任务': result['task'][:30] + '...' if len(result['task']) > 30 else result['task'],
            '提议对象数': len(result['proposed_objects']),
            '可用对象数': len(result['available_objects']),
            '处理时间': f"{result['proposal_time'] + result['planning_time']:.2f}秒",
            '时间戳': result['timestamp']
        })
    
    if history_data:
        st.dataframe(history_data, use_container_width=True)

# 页脚信息
st.markdown('---')
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 NLMap + Qwen3 智能机器人助手 | 基于本地部署的 Qwen3-4B 模型</p>
        <p>💡 提示: 请确保模型已正确加载后再进行任务测试</p>
    </div>
    """,
    unsafe_allow_html=True
)