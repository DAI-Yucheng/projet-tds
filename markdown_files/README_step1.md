# 第一步：数据管道实现

## 概述

这一步实现了从MUSDB数据集生成spectrogram patches的数据管道，完全按照论文参数配置。

## 论文参数

- **采样率**: 8192 Hz
- **STFT窗口**: 1024
- **STFT hop**: 768
- **Patch长度**: 128帧
- **输入shape**: (freq_bins, 128) ≈ (513, 128)

## 关键实现点

### 1. Overlap处理（TP要求回答的问题）

**问题**: "on observe un fort taux de recouvrement entre deux spectrogrammes de 128 trames"

**解决方案**: 使用sliding window with overlap（滑窗重叠）

- **实现方式**: 每32帧取一个patch（而不是每128帧）
- **重叠率**: (128-32)/128 = 75%
- **优势**: 
  - 增加训练样本数量（约4倍）
  - 保持时间连续性
  - 提高模型对边界区域的预测能力

**报告中的表述**:
> "Nous utilisons une fenêtre glissante avec recouvrement afin d'augmenter le nombre d'exemples d'apprentissage tout en conservant la continuité temporelle."

### 2. 数据预处理

- **STFT**: 使用librosa的STFT，参数严格按照论文
- **只取magnitude**: 忽略phase，phase在重建时使用mix的phase
- **归一化**: 
  - 先取log scale（log(x + eps)）
  - 再min-max归一化到[0, 1]

### 3. 生成器设计

- **无限循环**: `while True`，可以持续生成数据
- **随机采样**: 每次随机选择歌曲和chunk位置
- **Batch组织**: 自动收集patches组成batch

## 使用方法

### 安装系统依赖（重要！）

**musdb需要ffmpeg来处理音频文件，必须先安装：**

在Ubuntu/WSL中：
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

验证安装：
```bash
ffmpeg -version
```

### 安装Python依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
python data_generator.py
```

### 在训练中使用

```python
from data_generator import SpectrogramGenerator

# 创建生成器
generator = SpectrogramGenerator(
    batch_size=16,
    chunk_duration=5.0
)

# 获取数据
gen = generator.generate_batch()
for epoch in range(num_epochs):
    for batch_idx in range(batches_per_epoch):
        x_batch, y_batch = next(gen)
        # x_batch: (batch_size, 513, 128) - mix的spectrogram
        # y_batch: (batch_size, 513, 128) - vocals的spectrogram
        
        # 训练模型...
```

## 输出格式

- **x_batch**: mix的magnitude spectrogram patches
  - Shape: `(batch_size, freq_bins, patch_frames)`
  - 例如: `(16, 513, 128)`
  - 数值范围: [0, 1]（已归一化）

- **y_batch**: vocals的magnitude spectrogram patches
  - Shape: `(batch_size, freq_bins, patch_frames)`
  - 例如: `(16, 513, 128)`
  - 数值范围: [0, 1]（已归一化）

## 注意事项

1. **系统依赖**: 必须先安装ffmpeg（见上面的安装步骤）
2. **首次运行**: 如果MUSDB未下载，会自动下载（约4.5GB）
3. **内存**: 如果内存不足，可以减小`batch_size`或`chunk_duration`
4. **采样率**: MUSDB原始采样率是44100Hz，代码会自动重采样到8192Hz
5. **立体声处理**: 自动转换为单声道（取平均）

## 常见问题

### 错误: "ffmpeg or ffprobe could not be found"

**原因**: 系统缺少ffmpeg，这是musdb的必需依赖。

**解决方法**:
```bash
# Ubuntu/WSL
sudo apt-get update
sudo apt-get install -y ffmpeg

# 验证
ffmpeg -version
```

## 下一步

完成这一步后，可以：
1. 验证数据形状和数值范围是否正确
2. 可视化几个spectrogram patches，检查数据质量
3. 进入第二步：实现U-Net模型

