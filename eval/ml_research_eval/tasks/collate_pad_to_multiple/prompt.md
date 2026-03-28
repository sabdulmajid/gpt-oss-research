Implement `collate_pad_to_multiple(batch, pad_value=0, multiple_of=8)`.

Requirements:
- `batch` is a list of 1D tensors with different lengths.
- Pad them into a 2D tensor `[batch, padded_length]`.
- Round `padded_length` up to the nearest multiple of `multiple_of`.
