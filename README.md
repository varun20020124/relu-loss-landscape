# Loss‐Landscape Analysis of Shallow ReLU Networks

## Introduction  
This project explores the **mathematical structure** of the loss surface in shallow (1‐hidden‐layer) ReLU neural networks. By combining theoretical analysis and numerical experiments, we aim to understand:

- **Critical Point Analysis**: Characterize minima, maxima, and saddle points via gradients and Hessians.  
- **Geometry of the Loss Landscape**: Visualize how ReLU’s piecewise‐linear activations partition input space.  
- **Gradient Dynamics**: Track how gradient norms evolve during training under different optimizers.  
- **Over‑parameterization**: Empirically study how increasing hidden‐layer width smooths the loss surface and impacts performance.  
- **Implicit Regularization**: Observe how gradient descent implicitly controls model complexity by monitoring weight norms.
