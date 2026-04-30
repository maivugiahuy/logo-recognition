"""Image transforms — 160×160, color jitter, random perspective (Sec 4.2)."""
import torchvision.transforms as T

MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
STD = (0.26862954, 0.26130258, 0.27577711)

INPUT_SIZE = 160


def train_transforms(input_size: int = INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def val_transforms(input_size: int = INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
