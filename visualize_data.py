"""
可视化生成的spectrogram patches，用于检查数据质量
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import SpectrogramGenerator

def visualize_patches():
    """可视化几个spectrogram patches"""
    
    # 创建生成器
    generator = SpectrogramGenerator(
        batch_size=4,
        chunk_duration=5.0
    )
    
    # 获取一个batch
    gen = generator.generate_batch()
    x_batch, y_batch = next(gen)
    
    # 创建图像
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spectrogram Patches (Mix vs Vocals)', fontsize=16)
    
    # 显示4个样本
    for i in range(4):
        # Mix spectrogram
        ax1 = axes[0, i]
        im1 = ax1.imshow(
            x_batch[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax1.set_title(f'Mix - Sample {i+1}')
        ax1.set_xlabel('Time Frames')
        ax1.set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=ax1)
        
        # Vocals spectrogram
        ax2 = axes[1, i]
        im2 = ax2.imshow(
            y_batch[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax2.set_title(f'Vocals - Sample {i+1}')
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('spectrogram_patches.png', dpi=150, bbox_inches='tight')
    print("图像已保存为: spectrogram_patches.png")
    plt.show()
    
    # 打印统计信息
    print("\n数据统计:")
    print(f"Batch shape: {x_batch.shape}")
    print(f"\nMix (x_batch):")
    print(f"  Min: {x_batch.min():.4f}")
    print(f"  Max: {x_batch.max():.4f}")
    print(f"  Mean: {x_batch.mean():.4f}")
    print(f"  Std: {x_batch.std():.4f}")
    
    print(f"\nVocals (y_batch):")
    print(f"  Min: {y_batch.min():.4f}")
    print(f"  Max: {y_batch.max():.4f}")
    print(f"  Mean: {y_batch.mean():.4f}")
    print(f"  Std: {y_batch.std():.4f}")
    
    # 检查overlap效果
    print("\nOverlap验证:")
    print(f"  Patch长度: {generator.patch_frames} 帧")
    print(f"  Patch hop: {generator.patch_hop} 帧")
    print(f"  重叠率: {(generator.patch_frames - generator.patch_hop) / generator.patch_frames * 100:.1f}%")
    print(f"  每个5秒chunk大约生成: {int(5.0 * generator.sample_rate / generator.hop_length / generator.patch_hop)} 个patches")


if __name__ == "__main__":
    visualize_patches()

