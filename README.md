# Data-Driven Cloth for Interactive Applications

A neural-network-based approach for predicting 3D garment geometry from human pose parameters.

This project is inspired by the paper:

> **Data-Driven Clothing for Interactive Applications**  
> Kangrui Xue and Jane Wu  
> Stanford CS229 Project

The objective is to replace expensive cloth simulation with a learned model that can rapidly predict garment geometry from a given human pose.

---

## 1. Problem Statement

Physics-based cloth simulation can produce realistic garment deformation, but it can be computationally expensive for interactive applications.

This project investigates a data-driven alternative:

```text
Human Pose
    ↓
Neural Network
    ↓
PCA Garment Representation
    ↓
3D Cloth Geometry
```

Instead of running a physics simulation for every new pose, the trained neural network predicts the garment configuration directly.

---

## 2. Project Objective

The main objectives are:

- Learn the relationship between human pose and simulated garment geometry.
- Reduce the dimensionality of the high-dimensional garment representation using PCA.
- Train a fully connected neural network to predict PCA garment coefficients.
- Use ReLU activation functions.
- Train the network using the Adam optimizer.
- Reconstruct the predicted 3D garment from the PCA representation.
- Evaluate prediction accuracy against simulated ground-truth garments.
- Measure inference speed for interactive applications.
- Provide an interactive visualization of predicted cloth across test poses.

---

## 3. Reference Approach

The project is based on the general data-driven clothing pipeline presented in:

**Data-Driven Clothing for Interactive Applications**

The reference approach uses:

1. Large-scale cloth simulation to generate training data.
2. Pose parameters as input.
3. Dimensionality reduction of garment geometry using PCA.
4. A fully connected neural network to predict the reduced garment representation.
5. Reconstruction of garment geometry from the predicted representation.

Our implementation follows this general data-driven approach while adapting it to the available Santesteban simulated garment dataset.

---

## 4. Dataset

This project supports the T-shirt and Dress subsets of the Santesteban virtual try-on dataset. Each subset contains simulated garment sequences with pose parameters, vertices, triangular faces, and simulation metadata.

The raw dataset is approximately 11 GB and stays outside this repository. The pipeline reads up to 100 sorted `.pkl` simulation files from the selected garment's `simulations` directory. Mesh vertex and face counts are read from those files; they are not shared constants. The completed runs use 4,424 vertices and 8,710 faces for T-shirt, and 12,146 vertices and 23,949 faces for Dress.

The existing T-shirt processed arrays, checkpoint, and results are retained under the lowercase `tshirt/` artifact folders. Dress artifacts are isolated under `dress/`.

### Dataset Source

The dataset and associated project materials are provided by Santesteban et al. through their virtual try-on research project and repository.

- Project repository:  
  https://github.com/isantesteban/vto-garment-collisions

- Dataset repository:  
  https://github.com/isantesteban/vto-dataset

---

## 5. Input Representation

The available dataset provides a 72-dimensional pose representation.

Therefore:

```text
Input dimension = 72
```

Each input corresponds to one simulated garment frame.

The reference Stanford project uses joint rotation quaternions as its pose representation. This implementation uses the native 72-dimensional pose representation provided by the Santesteban dataset instead.

This is an implementation adaptation rather than an exact reproduction of the original paper's input representation.

---

## 6. Output Representation

Each garment is represented by three coordinates `(x, y, z)` per vertex. The resulting coordinate dimension is derived from that garment's vertex array; the existing T-shirt representation has 13,272 coordinates. Directly predicting all coordinates would result in a high-dimensional output space.

To reduce this dimensionality, PCA is applied to the garment geometry.

---

## 7. PCA Dimensionality Reduction

PCA is fitted only on the training data to avoid test-data leakage.

The garment representation is reduced to:

```text
64 PCA components
```

For the existing T-shirt baseline, the 64 components retain 99.9983% of the variance; for Dress, they retain 99.9964%. Variance retention is computed independently for each garment during preparation.

The neural network therefore predicts only 64 values instead of 13,272 coordinates.

The predicted 64-dimensional representation is then transformed back into the original garment space using the inverse PCA transformation.

---

## 8. Dataset Split

The selected 100 files per garment were divided into these frame-level splits:

| Garment | Total frames | Training | Validation | Testing |
|---|---:|---:|---:|---:|
| T-shirt | 12,290 | 8,603 | 1,843 | 1,844 |
| Dress | 12,645 | 8,851 | 1,897 | 1,897 |

The current implementation uses a frame-level random split.

---

## 9. Neural Network Architecture

A fully connected feed-forward neural network is used.

```text
Input
72 pose parameters
      │
      ▼
Linear Layer
72 → 128
      │
      ▼
ReLU
      │
      ▼
Linear Layer
128 → 256
      │
      ▼
ReLU
      │
      ▼
Linear Layer
256 → 128
      │
      ▼
ReLU
      │
      ▼
Linear Layer
128 → 64
      │
      ▼
PCA garment representation
```

### Architecture

```text
Input:       72
Hidden 1:    128 + ReLU
Hidden 2:    256 + ReLU
Hidden 3:    128 + ReLU
Output:      64
```

---

## 10. Training

The model is trained using:

- **Framework:** PyTorch
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Batch size:** 32
- **Epochs:** 100

Training automatically uses CUDA when available and otherwise falls back to CPU.

---

## 11. Existing T-Shirt Training Results

The training and validation loss decreased substantially during training.

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 99.117988 | 73.784012 |
| 10 | 22.067582 | 22.998413 |
| 20 | 10.563653 | 12.917937 |
| 30 | 6.222552 | 7.742776 |
| 40 | 4.285701 | 4.890603 |
| 50 | 3.869607 | 3.763932 |
| 60 | 2.743935 | 3.350342 |
| 70 | 2.284440 | 1.960781 |
| 80 | 1.527102 | 1.742367 |
| 90 | 1.748108 | 1.467753 |
| 100 | 0.974768 | 1.303923 |

The original T-shirt run retained loss values only at these 11 epochs; intermediate per-epoch values are not available. These checkpoints are also saved in `results/tshirt/training_history.csv`.

### Dress Training Results

The Dress model completed 100 epochs. Its final training loss was 3.548626 and validation loss was 6.319234. Per-epoch values are saved in `results/dress/training_history.csv`.

---

## 12. Evaluation

After predicting the 64 PCA coefficients, the coefficients are transformed back into the selected garment's original coordinate dimension.

The reconstructed representation is reshaped into:

3D garment vertices, with the vertex count obtained from that garment's prepared data.

### Existing T-Shirt Test Results

| Metric | Value |
|---|---:|
| Coordinate MSE | 0.00737465 |
| Coordinate RMSE | 0.08587576 |
| Mean vertex error | 0.09866848 |
| Maximum vertex error | 1.09905314 |

### Dress Test Results

| Metric | Value |
|---|---:|
| Coordinate MSE | 0.01157805 |
| Coordinate RMSE | 0.10760133 |
| Mean vertex error | 0.12303245 |
| Maximum vertex error | 1.64917481 |

The predicted garment is compared directly against the simulated ground-truth garment.

---

## 13. Existing T-Shirt Inference Performance

Inference was benchmarked on a CPU.

### Timing

| Operation | Time |
|---|---:|
| Neural network inference | 0.4683 ms |
| PCA reconstruction | 0.2189 ms |
| Total | 0.6872 ms |

This corresponds to an estimated prediction throughput of:

```text
1455.16 predictions/second
```

on the benchmark CPU environment.

The completed Dress benchmark on CPU measured 0.3288 ms for neural-network inference and 0.6621 ms for PCA reconstruction, for 0.9909 ms total (about 1,009.21 predictions/second).

> **Note:** This benchmark measures neural-network inference and PCA reconstruction only. It does not represent the complete GUI or 3D rendering frame rate.

---

## 14. Interactive Demo

An interactive visualization is provided using the test poses.

The user can move through the test frames using a slider.

For each pose, the system displays:

```text
Ground Truth Cloth
        vs.
Neural Network Prediction
```

The corresponding MSE, RMSE, and mean vertex error are also displayed.

Run:

```bash
python src/pipeline/interactive_demo.py --garment tshirt
python src/pipeline/interactive_demo.py --garment dress
```

---

## 15. Visualization

### Point-Cloud Comparison

Run:

```bash
python src/pipeline/visualize.py --garment tshirt
python src/pipeline/visualize.py --garment dress
```

Generates:

```text
results/tshirt/cloth_comparison.png
```

### Surface Mesh Comparison

Run:

```bash
python src/pipeline/visualize_mesh.py --garment tshirt
python src/pipeline/visualize_mesh.py --garment dress
```

Generates:

```text
results/tshirt/cloth_surface_comparison.png
```

Surface visualization uses the selected garment's triangular mesh topology.

### Training Curve

Run:

```bash
python src/pipeline/plot_training.py --garment tshirt
python src/pipeline/plot_training.py --garment dress
```

Generates:

```text
results/tshirt/training_validation_loss.png
```

---

## 16. Inference Pipeline

The complete pipeline is:

```text
                 SMPL Pose
                     │
                     ▼
            72-dimensional input
                     │
                     ▼
            Fully Connected NN
                     │
              ReLU activations
                     │
                     ▼
             64 PCA coefficients
                     │
                     ▼
                PCA inverse
                     │
                     ▼
       garment vertex count × 3 coordinates
                     │
                     ▼
                   garment-specific vertices
                     │
                     ▼
                selected garment mesh
```

---

## 17. Repository Structure

```text
Data-Driven-Cloth-For-Interactive-Applications/
│
├── data/
│   └── processed/
│       ├── tshirt/ (retained arrays, PCA, faces, metadata)
│       └── dress/  (prepared arrays, PCA, faces, metadata)
│
├── models/
│   ├── tshirt/cloth_predictor.pth
│   └── dress/cloth_predictor.pth
│
├── notebooks/
│
├── results/
│   ├── tshirt/ (retained T-shirt results)
│   └── dress/  (Dress-specific results)
│
├── src/pipeline/
│   ├── garment_config.py
│   ├── model.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   ├── evaluation_runner.py
│   ├── visualize.py
│   ├── visualize_mesh.py
│   ├── interactive_demo.py
│   ├── benchmark.py
│   └── plot_training.py
│
├── .gitignore
└── README.md
```

---

## 18. Installation

Create a Python environment if required.

Install the required packages:

```bash
pip install numpy
pip install scikit-learn
pip install joblib
pip install torch
pip install matplotlib
```

---

## 19. Running the Project

Set the external dataset root once in PowerShell (it contains `tshirt/` and `dress/`):

```powershell
$env:CLOTH_DATASET_ROOT = "D:\datasets\vto-dataset-main"
```

Alternatively, pass `--dataset-dir` with the selected garment's `simulations` folder. Neither option copies raw data into this repository.
All pipeline commands accept `--garment tshirt` or `--garment dress`; model and result files are kept separate per garment.

### Step 1 — Prepare Dataset

```bash
python src/pipeline/prepare_dataset.py --garment tshirt
python src/pipeline/prepare_dataset.py --garment dress
```

This:

- Loads the selected garment's simulation data.
- Creates train/validation/test splits.
- Fits PCA on training data.
- Saves arrays, PCA, mesh faces, and vertex metadata under `data/processed/<garment>/`.

### Step 2 — Train Model

```bash
python src/pipeline/train.py --garment tshirt
python src/pipeline/train.py --garment dress
```

The trained model is saved as:

```text
models/<garment>/cloth_predictor.pth
```

### Step 3 — Validate Model

```bash
python src/pipeline/validate.py
```

This evaluates on the validation split and saves validation-prefixed predictions, meshes, and metrics under `results/<garment>/`.

### Step 4 — Test Model

```bash
python src/pipeline/test.py
```

This evaluates once on the held-out test split and saves predictions, meshes, and metrics under `results/<garment>/`.

### Step 5 — Visualize Point Clouds

```bash
python src/pipeline/visualize.py
```

### Step 6 — Visualize Surface Meshes

```bash
python src/pipeline/visualize_mesh.py
```

### Step 7 — Run Interactive Demo

```bash
python src/pipeline/interactive_demo.py
```

### Step 8 — Benchmark Inference

```bash
python src/pipeline/benchmark.py
```

### Step 9 — Generate Training Curve

```bash
python src/pipeline/plot_training.py
```

Each command defaults to all configured garments. Pass `--garment tshirt` or `--garment dress` to run only one garment.

---

## 20. Important Implementation Note

The reference Stanford project predicts garment deformation offsets relative to a skinned garment representation.

The available Santesteban simulation files provide simulated garment vertices but do not directly provide the corresponding skinned garment representation required to reproduce that exact formulation.

Therefore, this implementation predicts:

**absolute simulated garment vertex coordinates**

rather than:

**garment deformation offsets**

The overall data-driven learning pipeline remains:

```text
Pose
  ↓
PCA garment representation
  ↓
Neural Network
  ↓
Reconstructed cloth
```

but the output formulation is adapted to the available dataset.

This distinction is important because the implementation is an adaptation of the reference approach rather than an exact reproduction.

---

## 21. Limitations

The current implementation has several limitations:

1. Results depend on the garment subsets and simulation files used for training.
2. Only 100 simulation files are used.
3. The dataset split is performed at the frame level rather than the sequence/file level.
4. The implementation predicts absolute garment coordinates rather than deformation offsets.
5. Temporal consistency between consecutive frames is not explicitly modeled.
6. The model uses pose information as the primary input.
7. The visualization and inference benchmark do not measure full rendering performance.
8. The model is trained on simulated garments and therefore depends on the distribution represented by the simulation dataset.

---

## 22. Future Improvements

Potential extensions include:

- Sequence-level train/validation/test splitting.
- Predicting garment deformation offsets instead of absolute coordinates.
- Temporal models for smoother cloth motion.
- Additional garment types such as dresses.
- More simulation data.
- Pose and body-shape conditioning.
- CNN or graph-based garment models.
- GPU acceleration.
- Collision-aware prediction.
- Interactive camera and pose controls.
- Real-time integration with a 3D graphics environment.

---

## 23. Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- PCA
- Neural Networks
- ReLU
- Adam Optimizer

---

## 24. Key Results

### Dataset

```text
12,290 frames
4,424 vertices per garment
8,710 triangular faces
72-dimensional pose input
```

### Dimensionality Reduction

```text
13,272 → 64 dimensions
```

with:

```text
99.9983% variance retained
```

### Prediction Accuracy

```text
MSE  = 0.00737465
RMSE = 0.08587576

Mean vertex error
= 0.09866848

Maximum vertex error
= 1.09905314
```

### Inference

```text
Neural network = 0.4683 ms
PCA            = 0.2189 ms
Total          = 0.6872 ms

Estimated prediction throughput
= 1455.16 predictions/second
```

---

## 25. References

### Main Reference

Xue, K., & Wu, J.  
**Data-Driven Clothing for Interactive Applications.**  
Stanford CS229 Project.

Project report:

https://cs229.stanford.edu/proj2021spr/report2/81970649.pdf

---

### Santesteban et al. — 2021

Santesteban, I., Thuerey, N., Otaduy, M. A., & Casas, D. (2021).

**Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On.**

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11763–11773.

```bibtex
@InProceedings{Santesteban_2021_CVPR,
    author = {Santesteban, Igor and Thuerey, Nils and Otaduy, Miguel A. and Casas, Dan},
    title = {Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021},
    pages = {11763-11773}
}
```

Project repository:

https://github.com/isantesteban/vto-garment-collisions

Dataset repository:

https://github.com/isantesteban/vto-dataset

---

### Santesteban et al. — 2019

Santesteban, I., Otaduy, M. A., & Casas, D. (2019).

**Learning-Based Animation of Clothing for Virtual Try-On.**

Computer Graphics Forum, 38(2), 355–366.

DOI:

https://doi.org/10.1111/cgf.13643

```bibtex
@article{santesteban2019virtualtryon,
    journal = {Computer Graphics Forum (Proc. Eurographics)},
    title = {{Learning-Based Animation of Clothing for Virtual Try-On}},
    author = {Santesteban, Igor and Otaduy, Miguel A. and Casas, Dan},
    year = {2019},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.13643}
}
```

---

## 26. Dataset Citation

If you use the Santesteban garment dataset and associated project materials, please cite:

```bibtex
@article{santesteban2021garmentcollisions,
    journal = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title = {{Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On}},
    author = {Santesteban, Igor and Thuerey, Nils and Otaduy, Miguel A and Casas, Dan},
    year = {2021}
}
```

---

## 27. Acknowledgements

We acknowledge the authors of the referenced research works and the associated Santesteban garment simulation resources for providing the research and data ecosystem that enabled this project.