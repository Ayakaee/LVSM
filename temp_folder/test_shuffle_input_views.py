#!/usr/bin/env python3
"""
测试输入视角随机打乱功能
"""

import torch
import numpy as np
from easydict import EasyDict as edict
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import ProcessData

def create_dummy_data(batch_size=2, num_views=8, num_input_views=4):
    """创建测试用的虚拟数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建虚拟图像数据 [b, v, c, h, w]
    images = torch.arange(batch_size * num_views * 3 * 64 * 64).reshape(batch_size, num_views, 3, 64, 64).float()
    
    # 创建虚拟相机内参 [b, v, 4]
    fxfycxcy = torch.ones(batch_size, num_views, 4) * 100.0
    
    # 创建虚拟相机外参 [b, v, 4, 4]
    c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, 4, 4)
    
    data_batch = {
        "image": images.to(device),
        "fxfycxcy": fxfycxcy.to(device),
        "c2w": c2w.to(device),
        "scene_name": ["scene_1", "scene_2"]
    }
    
    return data_batch

def test_shuffle_functionality():
    """测试打乱功能"""
    print("测试输入视角随机打乱功能...")
    
    # 创建配置
    config = edict({
        'training': {
            'num_input_views': 4,
            'num_target_views': 1,
            'num_views': 8,
            'shuffle_input_views': True,  # 启用打乱
            'dynamic_input_view_num': False,
            'target_has_input': True
        },
        'inference': {
            'if_inference': False
        }
    })
    
    # 创建数据处理器
    process_data = ProcessData(config)
    
    # 创建测试数据
    data_batch = create_dummy_data()
    
    print(f"原始数据形状: {data_batch['image'].shape}")
    print(f"原始输入视角顺序: {data_batch['image'][0, :4, 0, 0, 0].cpu().numpy()}")
    
    # 处理数据
    input_dict, target_dict = process_data(data_batch, has_target_image=True, target_has_input=True, compute_rays=False)
    
    print(f"处理后输入数据形状: {input_dict['image'].shape}")
    print(f"打乱后输入视角顺序: {input_dict['image'][0, :, 0, 0, 0].cpu().numpy()}")
    
    # 多次运行测试打乱的随机性
    print("\n测试多次运行的随机性:")
    for i in range(3):
        input_dict, _ = process_data(data_batch, has_target_image=True, target_has_input=True, compute_rays=False)
        print(f"运行 {i+1}: {input_dict['image'][0, :, 0, 0, 0].cpu().numpy()}")
    
    # 测试推理模式（不应该打乱）
    print("\n测试推理模式（不应该打乱）:")
    config.inference.if_inference = True
    process_data_inference = ProcessData(config)
    input_dict, _ = process_data_inference(data_batch, has_target_image=True, target_has_input=True, compute_rays=False)
    print(f"推理模式输入视角顺序: {input_dict['image'][0, :, 0, 0, 0].cpu().numpy()}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_shuffle_functionality() 