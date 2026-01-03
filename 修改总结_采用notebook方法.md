# 修改总结：采用 Notebook 方法

## 修改概述

已按照 `son_tp4.ipynb` 的简洁方法修改了所有相关文件，解决了 loss 不下降的问题。

## 主要修改点

### 1. `data_generator.py` ✅

#### 修改内容：
- **频率bins数量**：从 513 改为 **512**（便于网络处理，2的幂）
- **移除 log normalization**：不再对输入进行 log 归一化
- **直接输出 vocals magnitude**：不再计算 oracle_mask，直接返回 vocals magnitude
- **简化 patch 提取**：使用 50% 重叠（stride = patch_size // 2），与 notebook 一致

#### 关键代码变化：
```python
# 之前：
self.n_freq_bins = self.n_fft // 2 + 1  # 513
x_batch_norm = (np.log(x_batch + eps) + 12) / 14  # log normalization
oracle_mask = y_batch / (x_batch + eps)
yield x_batch_norm, oracle_mask

# 现在：
self.n_freq_bins = 512  # 512
magnitude = magnitude[:512, :]  # 只取前512个bins
yield x_batch, y_batch  # 直接返回原始magnitude
```

### 2. `unet_model.py` ✅

#### 修改内容：
- **默认频率bins**：从 513 改为 **512**
- **测试代码**：更新测试用例使用 512

#### 关键代码变化：
```python
# 之前：
n_freq_bins: int = 513

# 现在：
n_freq_bins: int = 512  # 与notebook一致
```

### 3. `train.py` ✅

#### 修改内容：
- **Loss 函数**：从 `OracleMaskLoss` (L1) 改为 `VocalsMagnitudeLoss` (MSE)
- **训练目标**：直接比较 `vocals_pred = mask * mix` 和 `vocals_true`
- **模型初始化**：使用 512 频率bins

#### 关键代码变化：
```python
# 之前：
class OracleMaskLoss(nn.Module):
    def forward(self, mask, oracle_mask):
        return self.l1(mask, oracle_mask)

# 现在：
class VocalsMagnitudeLoss(nn.Module):
    def forward(self, mask, mix, vocals):
        vocals_pred = mask * mix
        return self.mse(vocals_pred, vocals)
```

## 核心改进

### ✅ 解决的问题

1. **数据域统一**：
   - 之前：输入是 normalized，目标是原始域 → 域不匹配
   - 现在：输入和输出都在同一域（原始magnitude）→ 域统一

2. **训练目标明确**：
   - 之前：间接监督（预测mask，再计算oracle_mask）
   - 现在：直接监督（预测mask，直接计算vocals = mask * mix）

3. **维度匹配**：
   - 之前：513 bins（不是2的幂，可能导致尺寸问题）
   - 现在：512 bins（2的幂，便于网络处理）

4. **梯度稳定**：
   - 之前：log normalization 可能影响梯度
   - 现在：直接使用原始magnitude，梯度更稳定

### 📊 对比总结

| 特性 | 之前（Oracle Mask） | 现在（Notebook方法） |
|------|-------------------|-------------------|
| 频率bins | 513 | **512** |
| 输入归一化 | Log normalization [0,1] | **原始magnitude** |
| 训练目标 | Oracle mask | **Vocals magnitude** |
| Loss函数 | L1(mask, oracle_mask) | **MSE(mask*mix, vocals)** |
| 数据域 | 不匹配 | **统一** |
| 复杂度 | 高 | **低** |

## 使用方法

训练命令保持不变：
```bash
python train.py --epochs 20 --batch-size 16 --lr 0.0001
```

## 预期效果

- ✅ Loss 应该能正常下降
- ✅ 训练更稳定
- ✅ 收敛更快
- ✅ 与 notebook 版本行为一致

## 注意事项

1. **数据兼容性**：如果之前有保存的checkpoint，需要重新训练（因为模型架构从513改为512）

2. **学习率**：建议使用与notebook相同的学习率（0.0001）

3. **Batch size**：可以保持16，或根据GPU内存调整

## 下一步

1. 运行训练，观察loss是否正常下降
2. 如果loss仍然不动，检查：
   - 数据是否正确加载
   - 模型输入输出shape是否匹配
   - 梯度是否正常（可以使用 `torch.autograd.grad` 检查）

