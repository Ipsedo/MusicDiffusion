from torch import nn

from .time import TimeToScaleShift


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(
        m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    ):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, TimeToScaleShift):
        nn.init.xavier_normal_(m.first_weights)

        nn.init.zeros_(m.last_weights)
        nn.init.zeros_(m.last_bias)
