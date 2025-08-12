# 视角屏蔽功能 (View Masking)

## 概述

视角屏蔽功能允许在训练过程中动态屏蔽一些输入视角，通过attention mask来防止模型访问被屏蔽的视角信息。这个功能可以提高模型的鲁棒性，使其在部分视角缺失的情况下仍能正常工作。

## 功能特点

- **动态屏蔽**: 在训练过程中随机或按策略屏蔽部分输入视角
- **多种策略**: 支持随机、前几个、后几个、中间视角等屏蔽策略
- **可配置比例**: 可以设置屏蔽比例从0%到100%
- **训练时启用**: 只在训练时启用，推理时自动禁用

## 配置参数

在配置文件中添加以下参数：

```yaml
training:
  # 启用视角屏蔽功能
  use_view_masking: true
  
  # 屏蔽比例 (0.0-1.0)
  # 0.0: 不屏蔽任何视角
  # 0.3: 屏蔽30%的输入视角
  # 1.0: 屏蔽所有输入视角
  view_mask_ratio: 0.3
  
  # 屏蔽策略
  # - 'random': 随机屏蔽视角
  # - 'first': 屏蔽前几个视角
  # - 'last': 屏蔽后几个视角
  # - 'middle': 屏蔽中间的视角
  view_mask_strategy: 'random'
```

## 屏蔽策略详解

### 1. Random (随机屏蔽)
- 每个batch和每个target view都会随机选择要屏蔽的视角
- 适合模拟视角随机缺失的情况

### 2. First (屏蔽前几个)
- 固定屏蔽前N个视角
- 适合模拟视角按顺序缺失的情况

### 3. Last (屏蔽后几个)
- 固定屏蔽后N个视角
- 适合模拟视角按顺序缺失的情况

### 4. Middle (屏蔽中间)
- 屏蔽中间的N个视角
- 适合模拟视角中间缺失的情况

## 技术实现

### Attention Mask生成

```python
def create_attention_mask(self, v_input, v_target, n_patches, device, batch_size=1, mask_ratio=0.0, mask_strategy='random'):
    """
    创建attention mask来屏蔽一些视角
    
    Args:
        v_input: 输入视角数量
        v_target: 目标视角数量  
        n_patches: 每个视角的patch数量
        device: 设备
        batch_size: batch大小
        mask_ratio: 屏蔽比例 (0.0-1.0)
        mask_strategy: 屏蔽策略
        
    Returns:
        attn_mask: [b, v_target, n_patches, v_input * n_patches] 的attention mask
    """
```

### 在Cross Attention中应用

修改了`QK_Norm_CrossAttention`、`QK_Norm_SelfAttentionBlock`、`QK_Norm_CrossAttentionBlock`和`QK_Norm_SelfCrossAttentionBlock`类，添加了`attn_bias`参数支持。

### 在Forward过程中集成

在`forward`方法中：
1. 检查是否启用视角屏蔽
2. 生成attention mask
3. 将mask传递给`pass_layers`方法
4. 在cross attention中应用mask

## 使用示例

### 基本使用

```python
# 1. 在配置文件中启用功能
config.training.use_view_masking = True
config.training.view_mask_ratio = 0.3
config.training.view_mask_strategy = 'random'

# 2. 正常训练，系统会自动应用mask
model = Images2LatentScene(config, logger)
result = model(data_batch, input, target, train=True)
```

### 测试不同策略

```python
# 测试随机屏蔽
config.training.view_mask_strategy = 'random'
config.training.view_mask_ratio = 0.5

# 测试屏蔽前几个视角
config.training.view_mask_strategy = 'first'
config.training.view_mask_ratio = 0.25

# 测试屏蔽中间视角
config.training.view_mask_strategy = 'middle'
config.training.view_mask_ratio = 0.4
```

## 运行示例

```bash
# 运行演示脚本
python example_view_masking.py

# 使用配置文件训练
python train.py --config config_example_view_masking.yaml
```

## 注意事项

1. **训练时启用**: 视角屏蔽只在训练时启用，推理时自动禁用
2. **性能影响**: 使用attention mask会略微增加计算开销
3. **梯度检查点**: 与梯度检查点功能兼容
4. **REPA兼容**: 与REPA功能兼容
5. **Batch处理**: 支持多batch训练

## 实验建议

### 渐进式训练
```yaml
# 第一阶段：不使用屏蔽
training:
  use_view_masking: false

# 第二阶段：轻微屏蔽
training:
  use_view_masking: true
  view_mask_ratio: 0.1
  view_mask_strategy: 'random'

# 第三阶段：增加屏蔽
training:
  use_view_masking: true
  view_mask_ratio: 0.3
  view_mask_strategy: 'random'
```

### 不同策略对比
建议对比不同屏蔽策略的效果：
- Random vs Fixed策略
- 不同屏蔽比例的影响
- 与数据增强的结合效果

## 故障排除

### 常见问题

1. **Mask形状不匹配**
   - 检查`v_input`、`v_target`、`n_patches`参数
   - 确保batch_size正确

2. **内存不足**
   - 减少batch_size
   - 减少屏蔽比例

3. **训练不稳定**
   - 降低学习率
   - 减少屏蔽比例
   - 使用渐进式训练

### 调试技巧

```python
# 检查mask是否正确生成
mask = model.create_attention_mask(v_input, v_target, n_patches, device)
print(f"Mask shape: {mask.shape}")
print(f"Masked views: {torch.isinf(mask).sum().item() // (n_patches * v_target * batch_size)}")

# 检查attention权重
# 在transformer模块中添加调试代码
```

## 扩展功能

### 自定义屏蔽策略

可以扩展`create_attention_mask`方法添加新的屏蔽策略：

```python
elif mask_strategy == 'custom':
    # 实现自定义屏蔽逻辑
    pass
```

### 动态屏蔽比例

可以实现训练过程中动态调整屏蔽比例：

```python
# 根据训练步数调整屏蔽比例
current_step = get_current_step()
mask_ratio = min(0.5, current_step / 1000 * 0.3)
```

## 参考文献

- Attention Is All You Need
- Vision Transformer
- Multi-view Learning 