import SimpleITK as sitk
from dcmseg.settings import DATA_DIR
from pathlib import Path
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)

from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss

from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader


imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            ])

segtrans = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            ])

ds = ArrayDataset(images, imtrans, segs, segtrans)
