# Loss‐Landscape Analysis of Shallow ReLU Networks

## Introduction  
This project explores the **mathematical structure** of the loss surface in shallow (1‐hidden‐layer) ReLU neural networks. By combining theoretical analysis and numerical experiments, we aim to understand:

- **Critical Point Analysis**: Characterize minima, maxima, and saddle points via gradients and Hessians.  
- **Geometry of the Loss Landscape**: Visualize how ReLU’s piecewise‐linear activations partition input space.  
- **Gradient Dynamics**: Track how gradient norms evolve during training under different optimizers.  
- **Over‑parameterization**: Empirically study how increasing hidden‐layer width smooths the loss surface and impacts performance.  
- **Implicit Regularization**: Observe how gradient descent implicitly controls model complexity by monitoring weight norms.

## File Descriptions

1. **models/**  
   - `shallow_relu.py`  
     Defines the `ShallowReLU` class, a one‑hidden‑layer neural network with ReLU activation and a linear output.

2. **data/**  
   - `synthetic_generator.py`  
     Implements `SyntheticDatasetBuilder` to generate and split toy datasets (blobs, circles, moons) into train/test Torch tensors.

3. **experiments/**  
   - `shallow_trainer.py`  
     `ShallowReLUTrainer` class: loads data, trains the model, records loss history, plots loss curve, and evaluates test accuracy.  
   - `loss_surface_visualizer.py`  
     `LossSurfaceVisualizer` class: sweeps two parameters over a grid and plots the resulting loss surface as a contour.  
   - `activation_pattern_visualizer.py`  
     `ActivationPatternVisualizer` class: grids 2D input space and colors regions by ReLU activation patterns.  
   - `critical_point_classifier.py`  
     `CriticalPointClassifier` class: uses gradients and Hessian eigenvalues to label the current parameter point as local minimum, maximum, or saddle.  
   - `gradient_trajectory_analyzer.py`  
     `GradientTrajectoryAnalyzer` class: tracks gradient‐norm and loss at each epoch during training, and plots their trajectories.  
   - `overparam_experiment.py`  
     `OverparamExperiment` class: sweeps over specified hidden‐layer sizes, trains models, and records final loss and test accuracy in a DataFrame.  
   - `weight_norm_monitor.py`  
     `WeightNormMonitor` class: tracks the L₂ norm of all model weights alongside loss throughout training.

4. **analysis/**  
   - `gradient_analysis.py`  
     `GradientAnalyzer` class: computes the flat gradient vector and approximates the Hessian matrix for the loss, plus plotting helpers for gradient magnitudes and Hessian spectra.

5. **notebooks/**  
   - `main_analysis.ipynb`  
     Interactive Jupyter notebook that ties together data generation, model training, loss‐surface visualization, activation partitioning, gradient/Hessian analysis, critical‐point classification, gradient dynamics, over‑parameterization, and implicit regularization monitoring.
