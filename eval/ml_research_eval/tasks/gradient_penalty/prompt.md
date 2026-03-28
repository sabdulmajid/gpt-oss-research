Implement `compute_gradient_penalty(critic, real_data, fake_data)`.

Requirements:
- Compute interpolated data between `real_data` and `fake_data` using a random alpha.
- Pass interpolated data through `critic`.
- Compute gradients of critic output w.r.t. interpolated data.
- Return the mean squared distance of the L2 norm of the gradients from 1.