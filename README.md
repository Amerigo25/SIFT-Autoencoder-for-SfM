# SIFT Autoencoder for Structure from Motion

A deep learning approach to compressing SIFT descriptors for efficient 3D reconstruction using autoencoders. This project demonstrates how neural network-based compression can reduce feature descriptor dimensionality from 128 to 16 dimensions while maintaining reconstruction quality in Structure from Motion (SfM) pipelines.

## Overview

This project implements an autoencoder neural network to compress SIFT (Scale-Invariant Feature Transform) descriptors for 3D reconstruction tasks. By reducing descriptor size by 87.5% (from 128 to 16 dimensions), we can significantly decrease storage and transmission costs while preserving matching performance for SfM applications.

### Key Features

- **Feature Extraction**: Automated SIFT keypoint detection and descriptor computation with CLAHE preprocessing
- **Deep Compression**: Autoencoder architecture reducing descriptors from 128D to 16D
- **Match Preservation**: Maintains feature matching quality with compressed descriptors
- **COLMAP Integration**: Direct compatibility with COLMAP reconstruction pipeline
- **Evaluation Pipeline**: Comprehensive testing on multiple datasets with reconstruction error analysis

## Architecture

### Autoencoder Structure

```
Input (128D) → Encoder → Bottleneck (16D) → Decoder → Output (128D)

Encoder:  128 → 64 → 32 → 16 → 8
Decoder:  8 → 16 → 32 → 64 → 128
```

- **Activation**: ReLU throughout
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### Training

- **Training Sets**: Castle + Herz-Jesus-P8 datasets
- **Test Sets**: Fountain + Santo datasets
- **Batch Size**: 256
- **Epochs**: 10
- **Preprocessing**: CLAHE + L2 normalization

## Installation

### Requirements

```bash
pip install numpy matplotlib keras opencv-python
```

### Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Keras/TensorFlow
- OpenCV (cv2)

## Usage

### 1. Data Preparation

Organize your datasets in separate directories:

```
project/
├── castle/           # Training set 1
├── Herz-Jesus-P8/    # Training set 2
├── fountain/         # Test set 1
└── santo/            # Test set 2
```

### 2. Training the Model

```python
# Load training data
train_kp, train_des = load_dataset("castle")
train_kp2, train_des2 = load_dataset("Herz-Jesus-P8")
train_set = np.concatenate((np.concatenate(train_des), 
                            np.concatenate(train_des2)), axis=0)

# Train autoencoder
history = autoencoder.fit(train_set, train_set, epochs=10, batch_size=256)
```

### 3. Feature Compression

```python
# Extract and save keypoints
test_kp, test_des = load_dataset("fountain")
save_keypoints(test_kp, "fountain")

# Compress descriptors
compressed = []
for des in test_des:
    comp_des = encoder.predict(des)
    compressed.append(comp_des)
```

### 4. Feature Matching

```python
# Generate matches for different descriptor types
save_matches(test_des, test_kp, "fountain", "original_fountain.txt")
save_matches(reconstructed, test_kp, "fountain", "decoded_fountain.txt")
save_matches(compressed, test_kp, "fountain", "comp_fountain.txt")
```

## COLMAP Integration

The project outputs matches in COLMAP-compatible format:

### Keypoint Format

```
<num_keypoints> 128
<x> <y> <scale> <angle> <128 zeros>
...
```

### Match Format

```
<image1.jpg> <image2.jpg>
<idx1> <idx2>
...
```

These files can be directly imported into COLMAP for 3D reconstruction.

## Evaluation

### Reconstruction Error

The notebook computes MSE between original and reconstructed descriptors:

```python
ssd = []
for des1, des2 in zip(test_des, reconstructed):
    sd = np.mean((des1 - des2)**2)
    ssd.append(sd)
mse = np.mean(ssd)
```

### Matching Performance

Three matching configurations are evaluated:
1. **Original**: 128D SIFT descriptors
2. **Decoded**: Reconstructed 128D descriptors
3. **Compressed**: 16D encoded descriptors

## Dataset Download

The notebook includes optional Google Drive download functionality:

```python
DOWNLOAD = True  # Set to True to download datasets
```

Datasets are sourced from standard 3D reconstruction benchmarks.

## Matching Pipeline

### Feature Matching
- **Matcher**: FLANN (Fast Library for Approximate Nearest Neighbors)
- **Algorithm**: KD-Tree with 5 trees
- **Ratio Test**: Lowe's ratio test (threshold: 0.8)

### Optional Geometric Verification

The code includes commented geometric verification using RANSAC:

```python
# Uncomment for geometric verification
F, mask = cv.findFundamentalMat(pts1, pts2, cv.USAC_MAGSAC)
```

This filters matches using epipolar constraints but increases processing time.

## Results

The compression achieves:
- **87.5% size reduction** (128D → 16D)
- **Preserved matching quality** through learned representations
- **Low reconstruction error** on test sets
- **COLMAP-compatible outputs** for immediate use

## Project Context

**Course**: 3D Augmented Reality Final Project  
**Institution**: University of Padua  
**Program**: Master Degree in ICT for Internet and Multimedia - Cybersystems  
**Student**: Amerigo Aloisi

## Future Work

- Experiment with different bottleneck dimensions
- Test on additional datasets
- Compare with other descriptor compression methods
- Evaluate impact on final 3D reconstruction quality
- Implement real-time compression for live applications

## License

This project is submitted as academic work for the University of Padua.

## Acknowledgments

- SIFT feature extraction using OpenCV
- COLMAP for 3D reconstruction pipeline compatibility
- Keras/TensorFlow for deep learning framework
