import torch

from solution import detach_metrics_tree


def test_detach_metrics_tree_converts_tensors():
    tree = {
        "loss": torch.tensor(1.5),
        "per_layer": [torch.tensor([1.0, 2.0]), {"grad_norm": torch.tensor(3.0)}],
        "tag": "baseline",
    }
    detached = detach_metrics_tree(tree)
    assert detached["loss"] == 1.5
    assert detached["per_layer"][0] == [1.0, 2.0]
    assert detached["per_layer"][1]["grad_norm"] == 3.0
    assert detached["tag"] == "baseline"

