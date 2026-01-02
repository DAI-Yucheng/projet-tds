# 第二步：U-Net模型实现

## 概述

实现一个简化但正确的U-Net模型用于声源分离，完全按照TP要求。

## TP要求

### 必须包含的结构

1. **Encoder（编码器）**
   - Conv2D
   - stride = 2（下采样）
   - LeakyReLU激活函数

2. **Decoder（解码器）**
   - ConvTranspose2D（上采样）
   - **Skip connections（重点！）** - 连接encoder和decoder对应层

3. **最后一层**
   - Sigmoid激活函数
   - 输出mask ∈ [0, 1]

4. **Loss函数**
   - L1 Loss（不是MSE）
   - 公式: `L = || mask ⊙ X - Y ||₁`
   - 其中：
     - `mask`: 模型预测的mask
     - `X`: mix的magnitude spectrogram
     - `Y`: 真实的vocals magnitude spectrogram
     - `⊙`: 逐元素相乘

### 可以简化的地方

1. ✅ **不用训练两个模型** - 只做vocals分离（不做instruments）
2. ✅ **不用完全一样的通道数** - 可以使用更少的通道（如32而不是论文的更多）
3. ✅ **可以简化层数** - 4层就够（论文可能更多）

## 模型架构

```
输入: (batch, 513, 128) - mix的magnitude spectrogram
    ↓
添加channel维度: (batch, 1, 513, 128)
    ↓
Encoder (下采样):
  Conv2D + LeakyReLU + stride=2  (1 → 32 channels)
  Conv2D + LeakyReLU + stride=2  (32 → 64 channels)
  Conv2D + LeakyReLU + stride=2  (64 → 128 channels)
  Conv2D + LeakyReLU + stride=2  (128 → 256 channels)
    ↓
Decoder (上采样) + Skip Connections:
  ConvTranspose2D + LeakyReLU  (256 → 128) + skip(128)
  ConvTranspose2D + LeakyReLU  (256 → 64) + skip(64)
  ConvTranspose2D + LeakyReLU  (128 → 32) + skip(32)
  ConvTranspose2D + Sigmoid    (64 → 1)
    ↓
移除channel维度: (batch, 513, 128)
    ↓
输出: mask ∈ [0, 1]
    ↓
estimated_vocals = mask ⊙ mix
```

## 文件说明

### `unet_model.py`
- **UNet类**: 模型定义
- **test_unet()**: 测试函数，验证模型结构

### `train.py`
- **训练脚本**: 完整的训练流程
- **L1Loss类**: 实现论文的L1 loss
- **train()函数**: 主训练函数

### `inference.py`
- **推理脚本**: 使用训练好的模型进行预测
- **可视化功能**: 显示mix、mask和estimated vocals

## 使用方法

### 1. 测试模型结构

```bash
python unet_model.py
```

这会验证：
- 模型输入输出shape
- Mask值域在[0, 1]
- Skip connections正常工作

### 2. 训练模型

```bash
# 基本训练（使用默认参数）
python train.py

# 自定义参数
python train.py --epochs 20 --batch-size 16 --lr 1e-3 --n-songs 10

# 使用CPU（如果没有GPU）
python train.py --cpu
```

**训练参数说明**:
- `--epochs`: 训练轮数（建议10-20，目标是收敛）
- `--batch-size`: Batch大小（建议8-16）
- `--lr`: 学习率（建议1e-3到1e-4）
- `--n-songs`: 使用的歌曲数量（建议5-10首，快速测试）
- `--save-dir`: 模型保存目录（默认: checkpoints）

### 3. 查看训练进度

```bash
# 启动TensorBoard
tensorboard --logdir checkpoints/logs

# 然后在浏览器打开 http://localhost:6006
```

### 4. 使用模型推理

```bash
python inference.py
```

## 训练建议（TP要求）

根据TP的指导：

1. **数据量**: 选择5-10首歌曲（不是全部MUSDB）
2. **训练轮数**: 10-20 epochs
3. **Batch size**: 小一点（8-16）
4. **目标**: **收敛**（不是追求性能）
   - Loss曲线应该下降
   - 不应该发散

## 输出文件

训练后会生成：

```
checkpoints/
├── best_model.pth          # 最佳模型（验证loss最低）
├── final_model.pth         # 最终模型
├── checkpoint_epoch_5.pth   # 每5个epoch的checkpoint
├── checkpoint_epoch_10.pth
└── logs/                   # TensorBoard日志
    └── YYYYMMDD_HHMMSS/
```

## 关键点

### 1. Skip Connections（重点！）

这是U-Net的核心特性，必须实现：

```python
# Encoder保存特征图
skip_connections.append(encoder_output)

# Decoder连接对应层的特征
decoder_input = torch.cat([decoder_output, skip_connection], dim=1)
```

### 2. L1 Loss（不是MSE）

论文明确要求使用L1 loss：

```python
estimated_vocals = mask * mix_spec
loss = L1Loss(estimated_vocals, vocals_spec)
```

### 3. Mask值域

最后一层必须是Sigmoid，确保mask在[0, 1]：

```python
nn.Sigmoid()  # 最后一层
```

## 报告中的表述

可以在报告中这样写：

> "Nous implémentons une version simplifiée du U-Net proposée dans l'article, tout en conservant les principes essentiels (skip connections, masque spectral). Le modèle utilise un encodeur avec des couches Conv2D (stride=2) et LeakyReLU, et un décodeur avec des couches ConvTranspose2D et des connexions de saut. La fonction de perte utilisée est la perte L1: L = || mask ⊙ X - Y ||₁, comme spécifié dans l'article."

## 常见问题

### Q: 训练loss不下降？

A: 
- 检查学习率（尝试1e-4）
- 检查数据是否正确归一化
- 检查模型是否太小（增加n_channels）

### Q: 内存不足？

A:
- 减小batch_size
- 减小n_channels或n_layers

### Q: 如何知道模型收敛了？

A:
- Loss曲线应该下降
- 不应该发散（loss越来越大）
- 验证loss也应该下降

## 下一步

完成这一步后，可以：
1. 验证模型能正常训练和收敛
2. 查看TensorBoard的loss曲线
3. 进入第三步：音频重建

