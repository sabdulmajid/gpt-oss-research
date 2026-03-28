import torch


def detach_metrics_tree(tree):
    if isinstance(tree, dict):
        return {key: detach_metrics_tree(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [detach_metrics_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(detach_metrics_tree(value) for value in tree)
    if isinstance(tree, torch.Tensor):
        tensor = tree.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    return tree

