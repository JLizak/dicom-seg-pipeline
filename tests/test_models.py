import pytest
import torch

from dcmseg.models import build_model


@pytest.mark.parametrize("name", ["segresnet", "unet", "dynunet", "swinunetr"])
def test_forward_shape(name):
    kwargs = {"feature_size": 12} if name == "swinunetr" else {}
    model = build_model(name, in_channels=2, out_channels=2, **kwargs).eval()
    x = torch.randn(1, 2, 64, 64, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 64, 64, 32)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("resnet50", 2, 2)
