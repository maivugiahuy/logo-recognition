"""Image transforms — supports CLIP (ViT) and ImageNet (DINOv2) normalization."""
import torchvision.transforms as T

# CLIP normalization — used by ViT-B/32 (open_clip) backbone
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ImageNet normalization — used by DINOv2 backbone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Aliases for backwards compat
MEAN = CLIP_MEAN
STD = CLIP_STD

INPUT_SIZE = 160
DINOV2_INPUT_SIZE = 224


def train_transforms(input_size: int = INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def val_transforms(input_size: int = INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def train_transforms_dinov2(input_size: int = DINOV2_INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def val_transforms_dinov2(input_size: int = DINOV2_INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
