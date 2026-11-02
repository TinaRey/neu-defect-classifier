"""The single source of truth for image preprocessing.

Training, evaluation and the API all call `build_transform`, so an image is
turned into a tensor identically everywhere.
"""
from torchvision import transforms as T


def build_transform(image_size: int, mean: float, std: float, augment: bool = False) -> T.Compose:
    steps = []
    if augment:
        # Defects have no preferred orientation, so flips are label-preserving.
        # These act on the PIL image, before it becomes a tensor.
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    steps += [
        T.Grayscale(num_output_channels=1),  # files are RGB with R=G=B; collapse to 1 channel
        T.Resize((image_size, image_size)),
        T.ToTensor(),                        # PIL uint8 HxW -> float32 1xHxW in [0, 1]
        T.Normalize(mean=[mean], std=[std]), # (x - mean) / std, using TRAIN-set statistics
    ]
    return T.Compose(steps)


def build_raw_transform(image_size: int) -> T.Compose:
    """Same as build_transform without Normalize; used only to compute the train mean/std."""
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
