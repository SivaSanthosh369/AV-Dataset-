# Autonomous Vehicle (AV) Route Generator & Simulated Dataset Pipeline

An offline synthetic data generation pipeline engineered to create high-fidelity simulated road environments and pixel-level semantic segmentation masks for training Autonomous Vehicle (AV) perception models. This project is specifically tailored for initial sanity checks, model architecture validation, and hyperparameter tuning in road and lane-detection networks (e.g., U-Net, VGG-Net based Segmentation).

---

## 🚀 Features

* **100% Offline & Free:** Complete bypass of external API constraints, eliminating Google Maps Billing limitations, Mapillary token Parsing failures (`401/500`), and `404 Not Found` open-source link breakages.
* **Deterministic Perspective Synthesis:** Generates high-definition ($1280 \times 720$) color frames mimicking a vehicle's forward-facing dashcam perspective.
* **Automated Ground Truth Masking:** Outputs synchronized 1-channel discrete class masks for multi-class Semantic Segmentation.
* **Built-in Data Augmentation:** Dynamically injects random pixel-level lighting variances and sensor noise (Gaussian emulation) to improve model generalization.

---

## 📁 Repository Structure

```text
av_route_generator/
│
├── download_street_view_images.py  # Core pipeline script
├── straight_road_routes_v0.1.csv   # Target route parameters
│
└── carla_simulated_dataset/        # Auto-generated dataset directory
    ├── images/                     # Input Features (PNG Dashcam frames)
    └── masks/                      # Ground Truth Labels (Discrete Class Masks)
