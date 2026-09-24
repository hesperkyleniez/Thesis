# model.py
# Compatibility wrapper. The actual architectures used by the pipeline live in
# train.py so there is only one source of truth.
#
# Baselines:
#   CNNOnly
#   CNNGRUOnly
#   CNNGRUFusion
#
# Prediction-level fusion is implemented in prediction_fusion.py.

from train import CNNOnly, CNNGRUOnly, CNNGRUFusion

__all__ = ["CNNOnly", "CNNGRUOnly", "CNNGRUFusion"]


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, cls in [
        ("CNN-only", CNNOnly),
        ("CNN-GRU", CNNGRUOnly),
        ("CNN-GRU-F", CNNGRUFusion),
    ]:
        model = cls(dropout=0.5).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<12}: {params:,} parameters")

    print("\nmodel.py compatibility wrapper OK")
