#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SayCan with Qwen3-4B Integration
使用本地部署的Qwen3-4B替代闭源LLM进行对象提议
"""

import os
import re
import warnings

# 设置环境变量以避免事件循环冲突
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"导入错误: {e}")
    raise e

class Qwen3ObjectProposer:
    def __init__(self, model_path=None):
        """
        初始化Qwen3对象提议器
        """
        if model_path is None:
            model_path = "c:/Users/91954/Desktop/个人/课程/robot_nav/final_project/Qwen3-main/Qwen3-models"
        
        self.model_path = model_path
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"正在加载Qwen3模型从: {model_path}")
            print(f"使用设备: {self.device}")
            
            # 设置torch线程数以避免冲突
            torch.set_num_threads(1)
            
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False  # 避免快速tokenizer的潜在问题
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 减少内存使用
            )
            
            print("✓ Qwen3模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("尝试使用CPU模式...")
            try:
                self.device = "cpu"
                torch.set_num_threads(1)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True,
                    use_fast=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✓ Qwen3模型在CPU模式下加载成功")
            except Exception as cpu_e:
                print(f"❌ CPU模式下模型加载也失败: {str(cpu_e)}")
                raise cpu_e
    
    def generate_response_from_llm(self, prompt):
        """
        使用Qwen3生成回复
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        new_response = response[len(prompt):].strip()
        return new_response
    
    def parse_response(self, response):
        """
        解析LLM回复，提取对象列表
        """
        # 首先尝试找到第一行的对象列表（通常在冒号后）
        lines = response.split('\n')
        first_line = lines[0].strip()
        
        # 查找冒号后的内容
        if ':' in first_line:
            objects_part = first_line.split(':', 1)[1].strip()
        else:
            objects_part = first_line.strip()
        
        # 移除可能的句号
        objects_part = objects_part.rstrip('.')
        
        # 如果第一行为空或太短，尝试查找其他行
        if not objects_part or len(objects_part) < 3:
            for line in lines[1:3]:  # 只检查前几行
                line = line.strip()
                if line and not line.startswith('The task') and ':' in line:
                    objects_part = line.split(':', 1)[1].strip().rstrip('.')
                    break
        
        # 按逗号分割并清理
        objects = [obj.strip() for obj in objects_part.split(',') if obj.strip()]
        
        # 进一步清理，移除多余的词汇和无效内容
        cleaned_objects = []
        for obj in objects:
            # 跳过包含示例文本的对象
            if 'The task' in obj or 'may involve' in obj or len(obj) > 50:
                continue
                
            # 移除常见的连接词和修饰词
            obj = re.sub(r'^(and|or|the|a|an)\s+', '', obj, flags=re.IGNORECASE)
            obj = obj.strip()
            
            # 确保是有效的对象名称
            if obj and len(obj) > 1 and len(obj) < 30:  # 合理的对象名称长度
                cleaned_objects.append(obj)
        
        return cleaned_objects
    
    def query_llm_for_objects(self, task):
        """
        查询LLM获取任务相关对象
        """
        # 构建上下文学习提示
        in_context_learning = """The task 'hold the snickers' may involve the following objects: snickers.
The task 'wipe the table' may involve the following objects: table, napkin, sponge, towel, cloth.
The task 'put a water bottle and an oatmeal next to the microwave' may involve the following objects: water bottle, oatmeal, microwave.
The task 'place the mug in the cardboard box' may involve the following objects: mug, cardboard box.
The task 'go to the fridge' may involve the following objects: fridge.
The task 'put a grapefruit from the table into the bowl' may involve the following objects: grapefruit, table, bowl.
The task 'can you open the glass jar' may involve the following objects: glass jar.
The task 'heat up the taco and bring it to me' may involve the following objects: taco, human, microwave oven, fridge.
The task 'hold the fancy plate with flower pattern' may involve the following objects: fancy plate with flower pattern.
The task 'put the fruits in the fridge' may involve the following objects: fridge, apple, orange, banana, peach, grape, blueberry.
The task 'get a sponge from the counter and put it in the sink' may involve the following objects: sponge, counter, sink.
The task 'empty the water bottle' may involve the following objects: water bottle, sink.
The task 'i am hungry, give me something to eat' may involve the following objects: human, candy, snickers, chips, apple, banana, orange.
The task 'go to the trash can for bottles' may involve the following objects: trash can for bottles.
The task 'put the apple in the basket and close the door' may involve the following objects: apple, basket, door.
The task 'help me make a cup of coffee' may involve the following objects: cup, coffee, mug, coffee machine.
The task 'check what time is it now' may involve the following objects: clock, watch.
The task 'let go of the banana' may involve the following objects: banana, trash can.
The task 'put the grapes in the bowl and then move the cheese to the table' may involve the following objects: grape, bowl, cheese, table.
The task 'find a coffee machine' may involve the following objects: coffee machine.
The task 'clean the kitchen' may involve the following objects: kitchen, sink, sponge, towel, soap, counter, stove, dishwasher.
The task 'prepare breakfast' may involve the following objects: bread, toaster, butter, jam, plate, knife, milk, cereal, bowl.
The task 'water the plants' may involve the following objects: plants, watering can, water, pot, soil.
The task 'turn on the lights' may involve the following objects: light switch, lamp, bulb.
The task 'set the table for dinner' may involve the following objects: table, plate, fork, knife, spoon, napkin, glass, chair.
The task 'do the laundry' may involve the following objects: washing machine, clothes, detergent, basket, dryer.
The task 'vacuum the living room' may involve the following objects: vacuum cleaner, living room, carpet, sofa, floor, wall, ceiling.
The task 'organize the bookshelf' may involve the following objects: bookshelf, books, shelf, organizer, picture.
The task 'feed the pet' may involve the following objects: pet, food bowl, pet food, water bowl, water.
The task 'charge my phone' may involve the following objects: phone, charger, outlet, cable.
The task 'take out the garbage' may involve the following objects: garbage bag, trash can, garbage bin, bottle, paper.
The task 'wash the dishes' may involve the following objects: dishes, sink, soap, sponge, towel, plate, cup, mug, bowl, water.
The task 'make the bed' may involve the following objects: bed, pillow, blanket, sheet, mattress.
The task 'open the window' may involve the following objects: window, curtain, blinds.
The task 'lock the door' may involve the following objects: door, key, lock.
The task 'turn off the TV' may involve the following objects: TV, remote control, power button, monitor.
The task 'put on my shoes' may involve the following objects: shoes, socks, shoelace, bag.
The task 'brush my teeth' may involve the following objects: toothbrush, toothpaste, sink, mirror, cup, water.
The task 'cook pasta' may involve the following objects: pasta, pot, water, stove, salt, sauce, plate, table.
The task 'read a book' may involve the following objects: book, chair, lamp, glasses, bookmark, table.
The task 'listen to music' may involve the following objects: speaker, phone, headphones, music player.
The task 'study at desk' may involve the following objects: desk, chair, book, laptop, pen, paper, lamp, keyboard, mouse.
The task 'decorate the room' may involve the following objects: picture, plant, lamp, wall, floor, ceiling, window.
The task 'check the time' may involve the following objects: clock, phone, watch, wall.
The task 'seal a package' may involve the following objects: tape, box, paper, scissors.
The task 'take notes' may involve the following objects: pen, paper, notebook, laptop, keyboard, mouse, table.
The task 'water the plants' may involve the following objects: plant, water, bottle.
The task 'adjust room lighting' may involve the following objects: lamp, light switch, ceiling, wall.
The task 'organize workspace' may involve the following objects: desk, chair, laptop, keyboard, mouse, monitor, pen, paper, book.

The task '帮我准备咖啡' may involve the following objects: coffee machine, cup, mug, coffee beans, water, sugar, milk.
The task '清理厨房桌子' may involve the following objects: table, sponge, towel, cloth, soap, cleaning spray, bottle.
The task '准备简单早餐' may involve the following objects: bread, toaster, butter, jam, plate, knife, milk, cereal, bowl, egg, pan, table, cup, mug.
The task '洗碗收拾餐具' may involve the following objects: dishes, sink, soap, sponge, towel, plate, cup, mug, bowl, fork, knife, spoon, water.
The task '制作水果沙拉' may involve the following objects: apple, banana, orange, grape, bowl, knife, cutting board, spoon, table.
The task '整理客厅桌子' may involve the following objects: coffee table, magazine, book, remote control, tissue box, decorative item, lamp, plant, picture.
The task '打开电视看新闻' may involve the following objects: tv, remote control, sofa, chair, monitor.
The task '调节客厅灯光' may involve the following objects: lamp, light switch, remote control, dimmer, ceiling.
The task '收拾沙发上的物品' may involve the following objects: sofa, cushion, blanket, book, magazine, remote control, phone, laptop.
The task '给植物浇水' may involve the following objects: plant, watering can, water, pot, soil, bottle.
The task '整理办公桌' may involve the following objects: desk, paper, pen, pencil, notebook, folder, stapler, calculator, laptop, keyboard, mouse, monitor, phone.
The task '打印重要文件' may involve the following objects: printer, paper, computer, document, folder, laptop, keyboard, mouse.
The task '准备会议材料' may involve the following objects: paper, pen, notebook, folder, whiteboard, marker, projector, laptop, keyboard, mouse, table, chair.
The task '清空垃圾桶' may involve the following objects: trash can, garbage bag, waste, bottle, paper.
The task '整理文件夹' may involve the following objects: folder, paper, document, filing cabinet, desk, pen.
The task '整理床铺' may involve the following objects: bed, pillow, blanket, sheet, mattress, floor.
The task '收拾衣物' may involve the following objects: clothes, wardrobe, hangers, laundry basket, shoes, bag.
The task '设置闹钟' may involve the following objects: alarm clock, phone, nightstand, table.
The task '关闭窗帘' may involve the following objects: curtain, window, blinds, wall.
The task '准备睡前用品' may involve the following objects: pillow, blanket, water glass, book, phone, charger, tissue, table, lamp.
The task '封装包裹' may involve the following objects: tape, box, paper, scissors, table.
The task '记录笔记' may involve the following objects: pen, paper, notebook, laptop, keyboard, mouse, table, chair.
The task '查看时间' may involve the following objects: clock, phone, watch, wall.
The task '调整房间布局' may involve the following objects: chair, table, sofa, lamp, plant, picture, wall, floor, ceiling.
The task '整理书籍' may involve the following objects: book, bookshelf, table, desk, bag, paper.
The task '充电设备' may involve the following objects: phone, laptop, charger, cable, outlet, table.
The task '装饰墙面' may involve the following objects: picture, wall, tape, plant, clock.
The task '清洁地板' may involve the following objects: floor, mop, water, cleaning spray, bottle.
"""
        
        prompt = f"{in_context_learning}The task '{task}' may involve the following objects:"
        
        print(f"\n查询任务: {task}")
        print("正在生成对象提议...")
        
        response = self.generate_response_from_llm(prompt)
        parsed_objects = self.parse_response(response)
        
        print(f"原始回复: {response}")
        print(f"提取的对象: {parsed_objects}")
        
        return parsed_objects
    
    def generate_planning_text(self, task, available_objects):
        """
        基于任务和可用对象生成分阶段规划文本
        """
        objects_str = ", ".join(available_objects) if available_objects else "no specific objects"
        
        planning_prompt = f"""You are a helpful robot assistant. Given a task and available objects, provide a structured plan with two phases: object acquisition and task execution.

Task: {task}
Available objects: {objects_str}

Please provide a structured plan following this format:

**阶段一：获取所需物品**
1. [具体描述需要获取的第一个物品及其位置]
2. [具体描述需要获取的第二个物品及其位置]
...

**阶段二：分步执行规划**
1. [使用已获取物品的第一个执行步骤]
2. [使用已获取物品的第二个执行步骤]
...

Example format:
**阶段一：获取所需物品**
1. 前往厨房台面，拿取咖啡机
2. 前往橱柜，取出咖啡杯
3. 前往储物柜，获取咖啡豆
4. 前往水槽，准备清水

**阶段二：分步执行规划**
1. 将咖啡机放置在合适的工作台面上
2. 检查咖啡机电源连接
3. 在咖啡机中加入适量清水
4. 将咖啡豆放入咖啡机的豆仓中
5. 将咖啡杯放在咖啡机出水口下方
6. 启动咖啡机开始制作咖啡
7. 等待咖啡制作完成
8. 取出制作好的咖啡

Now provide the structured plan for the given task:
"""
        
        print(f"\n生成分阶段规划任务: {task}")
        print(f"可用对象: {objects_str}")
        print("正在生成分阶段规划文本...")
        
        response = self.generate_response_from_llm(planning_prompt)
        
        return response

# 兼容性函数，保持与原始API一致
def generate_response_from_llm(prompt):
    """
    兼容性函数 - 使用全局Qwen3实例
    """
    global qwen3_proposer
    if 'qwen3_proposer' not in globals():
        qwen3_proposer = Qwen3ObjectProposer()
    return qwen3_proposer.generate_response_from_llm(prompt)

def parse_response(response):
    """
    兼容性函数 - 使用全局Qwen3实例
    """
    global qwen3_proposer
    if 'qwen3_proposer' not in globals():
        qwen3_proposer = Qwen3ObjectProposer()
    return qwen3_proposer.parse_response(response)

def query_llm_for_objects(task):
    """
    兼容性函数 - 使用全局Qwen3实例
    """
    global qwen3_proposer
    if 'qwen3_proposer' not in globals():
        qwen3_proposer = Qwen3ObjectProposer()
    return qwen3_proposer.query_llm_for_objects(task)

if __name__ == "__main__":
    # 测试示例
    proposer = Qwen3ObjectProposer()
    
    # 测试对象提议
    test_tasks = [
        "help me prepare coffee",
        "clean the kitchen table", 
        "bring me a snack",
        "put the books on the shelf"
    ]
    
    print("=== Qwen3对象提议测试 ===")
    for task in test_tasks:
        objects = proposer.query_llm_for_objects(task)
        print(f"\n任务: {task}")
        print(f"提议对象: {objects}")
        print("-" * 50)
    
    # 测试分阶段规划生成
    print("\n=== Qwen3分阶段规划生成测试 ===")
    
    planning_test_cases = [
        {
            "task": "help me prepare coffee",
            "objects": ["coffee machine", "cup", "coffee beans", "water", "sugar", "milk"]
        },
        {
            "task": "clean the kitchen table",
            "objects": ["table", "cloth", "cleaning spray", "paper towels", "sponge"]
        },
        {
            "task": "prepare breakfast",
            "objects": ["bread", "toaster", "butter", "jam", "plate", "knife", "eggs", "pan"]
        }
    ]
    
    for test_case in planning_test_cases:
        print(f"\n{'='*60}")
        print(f"测试任务: {test_case['task']}")
        print(f"可用对象: {', '.join(test_case['objects'])}")
        print(f"{'='*60}")
        
        planning = proposer.generate_planning_text(test_case['task'], test_case['objects'])
        print(f"\n分阶段规划结果:\n{planning}")
        print(f"\n{'='*60}")