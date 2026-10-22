from baselines import Convformer_NSE, EWSNet, Liconvformer, MobileNetV2, ResNet18
from fact_ednet import FACTEDNet

MODEL_NAMES = ("fact_ednet", "convformer_nse", "ewsnet", "liconvformer", "mobilenet_v2", "resnet18")


def build_model(name, seq_len, num_classes, options=None):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if seq_len < 16:
        raise ValueError("The comparison suite requires seq_len >= 16")
    options = options or {}
    if name == "fact_ednet":
        return FACTEDNet(seq_len=seq_len, num_classes=num_classes, **options)
    if name == "convformer_nse":
        return Convformer_NSE(None, 1, num_classes)
    if name == "ewsnet":
        return EWSNet(1, num_classes, seq_len)
    if name == "liconvformer":
        return Liconvformer(1, num_classes, seq_len=seq_len)
    if name == "mobilenet_v2":
        return MobileNetV2(1, num_classes, seq_len=seq_len)
    return ResNet18(1, num_classes)
