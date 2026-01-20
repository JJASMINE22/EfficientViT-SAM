#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/1/XX XX:XX
# @Author  : Your Name
# @FileName: simple_interactive_tool.py
# @Software: PyCharm

import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

import torch
import numpy as np

from modeling.sam import EfficientViTSam
from modeling_utils.model_config import (
    ImageEncoderArgs,
    PromptEncoderArgs,
    MaskDecoderArgs
)

from backbone.efficientvit import EfficientViTBackbone
from modeling.sam import (
    EfficientImageEncoder,
    SamNeck,
    PromptEncoder,
    MaskDecoder
)
from modeling.transformer import EfficientTwoWayTransformer

from base_utils.parse_utils import CustomArgumentParser
from base_utils.utils import load_checkpoint, latest_checkpoint_path
from data_utils.transforms import ResizeLongestSide

from predictor import SamPredictor


class InteractiveSegmentation:
    def __init__(self, ckpt_dir, device='cuda'):
        self.device = device
        self.predictor = None

        # 初始化模型
        self.setup_predictor(ckpt_dir)

        # 交互状态
        self.fig = None
        self.ax = None
        self.current_image_np = None
        self.mode = 'point'

        # 提示词数据
        self.prompt_points = []
        self.prompt_labels = []
        self.prompt_box = None
        self.box_temp_points = []

        # 结果数据
        self.current_mask = None
        self.current_mask_color = None  # <--- 新增：用于存储当前掩膜的随机颜色

    def setup_predictor(self, ckpt_dir):
        """
        设置预测器
        """

        parser = CustomArgumentParser(
            [
                ImageEncoderArgs,
                PromptEncoderArgs,
                MaskDecoderArgs
            ]
        )

        (
            image_encoder_args,
            prompt_encoder_args,
            mask_decoder_args
        ) = parser.parse_args_into_dataclasses()

        image_encoder = EfficientImageEncoder(
            backbone=EfficientViTBackbone(
                image_encoder_args.widths,
                image_encoder_args.depths,
                dim=image_encoder_args.dim,
                expand_ratio=image_encoder_args.expand_ratio
            ),
            neck=SamNeck(
                feature_list=image_encoder_args.feature_list,
                in_size_list=image_encoder_args.in_size_list,
                head_width=image_encoder_args.head_width,
                head_depth=image_encoder_args.head_depth,
                image_embedding_size=prompt_encoder_args.image_embedding_size
            )
        )

        prompt_encoder = PromptEncoder(
            prompt_encoder_args.prompt_embed_dim,
            prompt_encoder_args.image_embedding_size,
            prompt_encoder_args.input_image_size,
            prompt_encoder_args.mask_in_chans
        )

        mask_decoder = MaskDecoder(
            transformer_dim=prompt_encoder_args.prompt_embed_dim,
            transformer=EfficientTwoWayTransformer(
                mask_decoder_args.depth,
                prompt_encoder_args.prompt_embed_dim,
                mask_decoder_args.num_heads,
                mask_decoder_args.mlp_dim,
                attention_type=mask_decoder_args.attention_type
            ),
            num_multimask_outputs=mask_decoder_args.num_multimask_outputs,
            iou_head_depth=mask_decoder_args.iou_head_depth,
            iou_head_hidden_dim=mask_decoder_args.iou_head_hidden_dim
        )

        sam_model = EfficientViTSam(
            image_encoder, prompt_encoder, mask_decoder
        )

        load_checkpoint(
            latest_checkpoint_path(ckpt_dir), sam_model
        )

        sam_model.eval()
        sam_model.to(self.device)

        self.predictor = SamPredictor(
            sam_model,
            ResizeLongestSide(target_length=prompt_encoder_args.input_image_size[0]),
            self.device
        )

    def reset_state(self):
        """重置单张图片的状态"""
        self.prompt_points = []
        self.prompt_labels = []
        self.prompt_box = None
        self.box_temp_points = []
        self.current_mask = None
        self.current_mask_color = None
        self.mode = 'point'

    def update_display(self):
        """刷新画布"""
        if self.ax is None: return
        self.ax.clear()

        # 1. 画原图
        self.ax.imshow(self.current_image_np)

        # 2. 画掩膜 (随机颜色)
        if self.current_mask is not None and self.current_mask_color is not None:
            self._draw_mask(self.current_mask, self.current_mask_color)

        # 3. 画点
        if self.prompt_points:
            pts = np.array(self.prompt_points)
            lbls = np.array(self.prompt_labels)
            # 正样本绿点，负样本红点
            self.ax.scatter(pts[lbls == 1, 0], pts[lbls == 1, 1], c='lime', marker='*', s=150, edgecolors='black',
                            label='Positive')
            self.ax.scatter(pts[lbls == 0, 0], pts[lbls == 0, 1], c='red', marker='x', s=150, linewidth=2,
                            label='Negative')

        # 4. 画框
        if self.prompt_box is not None:
            x1, y1, x2, y2 = self.prompt_box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
            self.ax.add_patch(rect)

        # 画临时的框
        if self.mode == 'box' and self.box_temp_points:
            for p in self.box_temp_points:
                self.ax.plot(p[0], p[1], 'bo')

        self.ax.set_title(f"Mode: {self.mode.upper()} | 's': Segment | 'c': Clear | 'q': Quit Current Img")
        self.fig.canvas.draw()

    def _draw_mask(self, mask, color_rgba):
        """
        绘制掩膜 - 修复版 (强制使用 uint8 0-255 区间)
        mask: (H, W) 范围 0~1
        color_rgba: (4,) [r, g, b, alpha] 范围 0~1
        """
        h, w = mask.shape[-2:]

        # 1. 创建全透明的 uint8 图层 (0~255)
        mask_image = np.zeros((h, w, 4), dtype=np.uint8)

        # 2. 将 0~1 的随机颜色转换为 0~255 的整数颜色
        # color_rgba 是 [r, g, b, a]，我们需要把它乘 255 并转为整数
        color_uint8 = (np.array(color_rgba) * 255).astype(np.uint8)

        # 3. 提取掩膜区域 (mask > 0.5)
        # 注意：这里确保 mask 是布尔类型，防止形状对不齐
        mask_bool = mask > 0.5

        # 4. 填充颜色
        # 利用 numpy 的广播机制，将 (4,) 的颜色填入 (N, 4) 的像素中
        mask_image[mask_bool] = color_uint8

        # 5. 绘制
        # 因为原图是 uint8，现在掩膜也是 uint8，matplotlib 处理起来最舒服
        self.ax.imshow(mask_image)

    def perform_segmentation(self):
        if not self.prompt_points and self.prompt_box is None:
            return

        point_coords = np.array(self.prompt_points) if self.prompt_points else None
        point_labels = np.array(self.prompt_labels) if self.prompt_labels else None
        box_coords = np.array(self.prompt_box) if self.prompt_box else None

        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]

            Image.fromarray((self.current_mask * 255).astype("uint8")).show()

            # --- 调试信息：检查形状是否匹配 ---
            # print(f"DEBUG: Mask Shape: {self.current_mask.shape}, Image Shape: {self.current_image_np.shape}")

            # 生成随机颜色 (0~1)
            rng_color = np.random.random(3)
            # 拼接透明度 (0.6)，稍微调高一点透明度确保能看见
            self.current_mask_color = np.concatenate([rng_color, [0.6]])

            print(f"Segmented! Score: {scores[best_idx]:.3f}")
            self.update_display()
        except Exception as e:
            print(f"Seg Error: {e}")
            import traceback
            traceback.print_exc()

    # --- 事件处理函数 ---
    def on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata

        if self.mode == 'point':
            # 左键加正点(1)，右键加负点(0)
            lbl = 0 if event.button == 3 else 1
            self.prompt_points.append([x, y])
            self.prompt_labels.append(lbl)
        elif self.mode == 'box':
            self.box_temp_points.append([x, y])
            if len(self.box_temp_points) == 2:
                xs = [p[0] for p in self.box_temp_points]
                ys = [p[1] for p in self.box_temp_points]
                self.prompt_box = [min(xs), min(ys), max(xs), max(ys)]
                self.box_temp_points = []
        self.update_display()

    def on_key(self, event):
        k = event.key
        if k == 'q':
            plt.close(self.fig)  # 关闭当前窗口，触发循环继续
        elif k == 'p':
            self.mode = 'point'
        elif k == 'b':
            self.mode = 'box'
            self.box_temp_points = []
        elif k == 'd':
            self.perform_segmentation()
        elif k == 'c':
            self.reset_state()
            self.update_display()

    # --- 主循环入口 ---
    def start_loop(self):
        print("\n=== SAM Interactive Tool Started ===")
        print("Type 'exit' or press Ctrl+C to stop the program completely.")

        while True:
            try:
                # 1. 获取输入
                path_input = input("\n[INPUT] Enter image path: ").strip()

                # 退出条件
                if path_input.lower() in ['exit', 'quit']:
                    print("Bye!")
                    break

                # 去除引号（防止用户直接拖拽文件进终端带有引号）
                path_input = path_input.replace('"', '').replace("'", "")

                if not os.path.exists(path_input):
                    print(f"Error: File not found -> {path_input}")
                    continue

                # 2. 加载图片
                image = Image.open(path_input).convert("RGB")
                self.current_image_np = np.array(image)
                self.reset_state()
                self.predictor.set_image(image)
                print(f"Loaded: {os.path.basename(path_input)} ({image.size})")

                # 3. 启动 GUI (阻塞式)
                self.fig, self.ax = plt.subplots(figsize=(10, 8))
                self.fig.canvas.mpl_connect('button_press_event', self.on_click)
                self.fig.canvas.mpl_connect('key_press_event', self.on_key)

                # 初始绘制
                self.update_display()

                print("Interactive window opened. Close window (or press 'q') to enter next image.")
                plt.show()  # <--- 程序会停在这里，直到窗口关闭

                print("Window closed.")

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"Unexpected Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simple Interactive Segmentation Tool')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoint",
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')

    args = parser.parse_args()

    tool = InteractiveSegmentation(
        ckpt_dir=args.ckpt_dir,
        device=args.device
    )

    tool.start_loop()


if __name__ == "__main__":
    main()
