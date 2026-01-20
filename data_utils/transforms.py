import torch
# from torch.nn import functional as F
from torchvision.transforms import functional as F

from PIL import Image

from typing import Tuple, List


class RandomHFlip:
    def __init__(
            self,
            prob: float = .5
    ):
        self.prob = prob

    def __call__(self, *args, **kwargs):
        (
            image,
            masks,
            points,
            bboxs,
            shape
        ) = args

        _, w = shape

        image = torch.flip(image, [2])
        masks = torch.flip(masks, [2])
        points[:, 0] = w - points[:, 0]
        bboxs[:, 0::2] = w - bboxs[:, [2, 0]]

        return (
            image,
            masks,
            points,
            bboxs,
            shape
        )


class ResizeLongestSide:
    def __init__(
            self,
            target_length: int
    ):
        self.target_length = target_length

    def apply_image(self, image: torch.Tensor, size: Tuple[int, int]):
        oldh, oldw = size
        new_h, new_w = self.get_preprocess_shape(oldh, oldw, self.target_length)

        image = F.resize(image, [new_h, new_w])

        return image

    def apply_coords(self, coords: torch.Tensor, size: Tuple[int, int]):
        oldh, oldw = size
        new_h, new_w = self.get_preprocess_shape(oldh, oldw, self.target_length)
        coords = coords.clone().float()
        coords[:, 0] = coords[:, 0] * new_w / oldw
        coords[:, 1] = coords[:, 1] * new_h / oldh

        return coords

    def apply_boxes(self, boxes: torch.Tensor, size: Tuple[int, int]):
        boxes = self.apply_coords(
            boxes.reshape(-1, 2), size
        )

        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        scale = 1 / max(oldh, oldw) * long_side_length
        new_h = oldh * scale
        new_w = oldw * scale

        # transform to pixel-center
        new_h = int(new_h + .5)
        new_w = int(new_w + .5)

        return new_h, new_w

    def __call__(self, *args):
        (
            image,
            masks,
            points,
            bboxs,
            shape
        ) = args

        image = self.apply_image(
            image, shape
        )

        while masks.ndim < 3:
            masks = masks[None]
        masks = self.apply_image(
            masks, shape
        ).squeeze()

        points = self.apply_coords(
            points, shape
        )
        bboxs = self.apply_boxes(
            bboxs, shape
        )

        return (
            image,
            masks,
            points,
            bboxs,
            shape
        )


class Pad:
    def __init__(
            self,
            target_length: int
    ):
        self.target_length = target_length

    def __call__(self, *args):
        (
            image,
            masks,
            points,
            bboxs,
            shape
        ) = args

        h, w = image.shape[-2:]

        padh = self.target_length - h
        padw = self.target_length - w

        # right - bottom padding
        image = F.pad(image, [0, 0, padw, padh], fill=0)
        while masks.ndim < 3:
            masks = masks[None]
        masks = F.pad(masks, [0, 0, padw, padh], fill=0).squeeze()

        return (
            image,
            masks,
            points,
            bboxs,
            shape
        )


class Normalize:
    def __init__(
            self,
            mean: List[float],
            std: List[float]
    ):
        self.mean = mean
        self.std = std

    def __call__(self, *args):
        (
            image,
            masks,
            points,
            bboxs,
            shape
        ) = args

        image = F.normalize(
            image, self.mean, self.std
        )

        return (
            image,
            masks,
            points,
            bboxs,
            shape
        )


class Compose:
    def __init__(
            self,
            *transforms
    ):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)

        return args


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor, normalize, to_pil_image

    image = Image.open(
        r"E:\github\Qwen\assets\hfagent_run.png"
    ).convert("RGB")
    image = image.resize(
        (512, 512)
    )

    image = pil_to_tensor(
        image
    ).float()
    image = image[None]

    w, h = 512, 512
    grids = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
        indexing="xy"

    )

    grids = torch.stack(grids, dim=-1).float()
    grids += .5
    grids /= torch.tensor([h, w])

    grids = grids * 2 - 1
    grids = grids[None]

    image_ = torch.nn.functional.grid_sample(
        image / 255, grids
    )
    to_pil_image(image_[0]).show()
    1
