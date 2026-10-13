import torch
import timm
import torch.nn as nn

class BreastCTSegmentation(nn.Module):
    
    def __init__(self, model_name: str, in_chans: int = 1, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=in_chans)

    def forward(self, x: torch.Tensor):
            return self.backbone(x)
    
def get_model(model_name: str, in_chans: int = 1, pretrained: bool = True):
    model = BreastCTSegmentation(model_name, in_chans, pretrained)

    return model
