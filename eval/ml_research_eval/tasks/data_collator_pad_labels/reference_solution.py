import torch


def pad_batch(examples, pad_token_id):
    max_len = max(len(example["input_ids"]) for example in examples)
    input_rows = []
    label_rows = []
    mask_rows = []
    for example in examples:
        pad_amount = max_len - len(example["input_ids"])
        input_rows.append(example["input_ids"] + [pad_token_id] * pad_amount)
        label_rows.append(example["labels"] + [-100] * pad_amount)
        mask_rows.append([1] * len(example["input_ids"]) + [0] * pad_amount)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }

