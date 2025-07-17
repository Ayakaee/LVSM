# Patch插值功能说明

## 概述

LVSM模型现在支持一个新的选项，允许对预训练编码器（如DINO）的输出进行patch插值，而不是通过调整输入图像分辨率来匹配patch数量。

## 问题背景

在原始的LVSM实现中，为了确保编码器输出的patch数量与模型的patch数量匹配，模型会调整输入图像的分辨率。例如：
- 如果模型使用8×8的patch size，256×256的图像会产生1024个patch
- 但DINO编码器在448×448输入下只产生256个patch（16×16网格）
- 原来的解决方案是将输入图像调整到更高的分辨率以匹配patch数量

## 新的解决方案

通过添加`use_patch_interpolation`选项，现在可以：
1. 使用编码器的标准输入分辨率（如DINO的448×448）
2. 获得编码器的原始输出（如256个patch）
3. 通过双线性插值将编码器输出插值到目标patch数量（如1024个patch）

## 配置选项

在配置文件的`model.image_tokenizer`部分添加：

```yaml
image_tokenizer:
  type: dino  # 或 PE
  image_size: 256
  patch_size: 8
  in_channels: 9
  use_patch_interpolation: true  # 新增选项
```

## 使用方法

### 1. 使用patch插值（推荐）
```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_decoder_only_patch_interp.yaml
```

### 2. 使用原始方法（调整输入分辨率）
```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_ours.yaml
```

## 技术细节

### 插值过程
1. **编码器输入**：使用标准分辨率（如448×448）输入到预训练编码器
2. **特征提取**：获得编码器的原始输出特征（如256个patch）
3. **特征重塑**：将特征重塑为2D网格格式
4. **双线性插值**：使用双线性插值将特征网格调整到目标大小
5. **特征重塑**：将插值后的特征重塑回patch序列格式

### 支持的编码器
- **DINO/DINOv2**：支持patch插值
- **PE-Core/PE-Spatial**：支持patch插值
- **其他编码器**：可以通过类似方式扩展

## 优势

1. **保持编码器性能**：使用编码器的标准输入分辨率，避免因分辨率调整导致的性能下降
2. **更灵活的特征处理**：可以在特征空间而不是像素空间进行插值
3. **更好的特征保持**：双线性插值在特征空间通常比在像素空间调整分辨率更有效
4. **易于扩展**：可以轻松支持其他预训练编码器

## 注意事项

1. **内存使用**：patch插值可能会增加一些内存使用，但通常可以忽略不计
2. **计算开销**：插值操作会带来轻微的计算开销
3. **特征质量**：插值可能会轻微影响特征质量，但通常比调整输入分辨率更好

## 实验建议

建议对比两种方法：
1. 使用`use_patch_interpolation: false`（原始方法）
2. 使用`use_patch_interpolation: true`（新方法）

比较指标包括：
- 训练收敛速度
- 最终模型性能（PSNR、SSIM、LPIPS）
- 推理速度
- 内存使用情况 