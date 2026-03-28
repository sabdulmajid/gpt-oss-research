Implement `shift_labels_for_causal_lm(labels, ignore_index=-100)`.

Requirements:
- Return a tensor with the same shape as `labels`.
- Each position should contain the next token label.
- The last position should be filled with `ignore_index`.
