Implement `smooth_labels(targets, num_classes, smoothing)`.

Requirements:
- `targets` is a 1D tensor of class indices.
- Return one-hot encoded and smoothed targets.
- The true class gets `1.0 - smoothing + smoothing / num_classes`.
- Other classes get `smoothing / num_classes`.