import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]

using_ckpt: bool = False

# ─── helper convs ─────────────────────────────────────────────────────

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding (bias False)"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1×1 convolution (bias False)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ─── basic block ──────────────────────────────────────────────────────

class IBasicBlock(nn.Module):
    """ArcFace IR basic block (two 3×3 convs, expansion = 1)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    # internal forward so we can checkpoint if desired
    def _impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.training and using_ckpt:
            return checkpoint(self._impl, x)
        return self._impl(x)


# ─── backbone ─────────────────────────────────────────────────────────

class IResNet(nn.Module):

    fc_scale: int = 7 * 7  # spatial dims after conv4

    def __init__(
        self,
        block: type[IBasicBlock],
        layers: list[int],
        *,
        dropout: float = 0.0,
        num_features: int = 512,
        fp16: bool = False,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # head
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[attr-defined]

    # ---- layer builder ------------------------------------------------
    def _make_layer(
        self,
        block: type[IBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )

        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # ---- forward ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


# ─── factory helpers ─────────────────────────────────────────────────

def _iresnet(
    layers: list[int],
    *,
    pretrained: bool = False,
    **kwargs,
) -> IResNet:
    if pretrained:
        raise ValueError("Use your own checkpoint – set pretrained=False.")
    return IResNet(IBasicBlock, layers, **kwargs)


# public ctors --------------------------------------------------------

def iresnet18(pretrained: bool = False, **kwargs) -> IResNet:  
    return _iresnet([2, 2, 2, 2], pretrained=pretrained, **kwargs)


def iresnet34(pretrained: bool = False, **kwargs) -> IResNet: 
    return _iresnet([3, 4, 6, 3], pretrained=pretrained, **kwargs)


def iresnet50(pretrained: bool = False, **kwargs) -> IResNet:
    return _iresnet([3, 4, 14, 3], pretrained=pretrained, **kwargs)


def iresnet100(pretrained: bool = False, **kwargs) -> IResNet: 
    return _iresnet([3, 13, 30, 3], pretrained=pretrained, **kwargs)


def iresnet200(pretrained: bool = False, **kwargs) -> IResNet: 
    return _iresnet([6, 26, 60, 6], pretrained=pretrained, **kwargs)
