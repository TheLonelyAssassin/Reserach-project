# Plug and Play: Attack-Invariant Attention Filtering for Next-Gen Facial Recognition

**Master's Thesis Project**  
University of Adelaide, 2025  
Author: Kamalesh Gunasekaran

---

## Overview

AIAF is a plug-and-play defense mechanism that protects face recognition systems against adversarial attacks. This implementation accompanies my Master's thesis on adversarial robustness in face recognition.

**Key Results:**
- **Digital Attack Defense**: 60-99% TAR@FAR recovery against PGD, C&W, AutoAttack, Sibling attacks
- **Real-time Performance**: 10.7-11.7ms inference latency
- **Data Efficiency**: Trained on only 45K images (1.4% of VGGFace2)
- **Plug-and-Play**: Works with frozen face recognition models (ArcFace, CosFace, MagFace)

---

## Requirements

- **Python**: 3.10
- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070)
- **CUDA**: 12.1
- **OS**: Windows or Linux
- **Conda**: Anaconda or Miniconda

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/TheLonelyAssassin/Reserach-project.git
cd Reserach-project
```

### Step 2: Create Environment from File
```bash
conda env create -f environment.yml
```

This will create a conda environment named `aiaf` with all dependencies installed automatically.

### Step 3: Activate Environment
```bash
conda activate aiaf
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.2.0+cu121
CUDA Available: True
```
---

## Dataset Preparation

### VGGFace2 Dataset

1. **Download VGGFace2**
   - Request access at: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
   - Download both train and test sets

2. **Preprocessing**
   - Faces should be aligned to 112×112 pixels
   - Use the script align_faces.py to center align the faces and crop to 112x112.

43. **Mask Template**
   - Place the mask overlay template (PNG image) at `mask_template.png`
---

## Training
Due to compuattaional constraints we used AIAF_Tiny 
### Quick Start (AIAF-Tiny)

```bash
python train.py \
    --data_dir data/vggface2/train \
    --mask data/mask_template.png \
    --batch 64 \
    --epochs 50 \
```

### Full Training (AIAF-Stable)
import the AIAF_Stable in aiaf.py in your training instaed of AIAF_Tiny
```bash
python train.py \
    --data_dir data/vggface2/train \
    --mask data/mask_template.png \
    --batch 128 \
    --epochs 100 \
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to training images | Required |
| `--mask` | Path to mask template PNG | Required |
| `--batch` | Batch size | 64 |
| `--epochs` | Number of epochs | 50 |
| `--lr` | Learning rate |default = 1e-4 |
| `--save_dir` | Directory to save checkpoints | `checkpoints/` |
| `--accum` | Gradient accumulation steps | default = 4 |

### What Happens During Training

The training script will:
1. **Warmup Phase** (3-5 epochs): Train on clean images to establish baseline
2. **Adversarial Training**: Train with 6 attack types simultaneously:
   - PGD (Projected Gradient Descent)
   - C&W (Carlini & Wagner)
   - AutoAttack (ensemble attack)
   - Sibling Transfer Attack
   - Adversarial Patches
   - Mask Overlays
3. **Save Best Models**:
   - `best_identity.pth` - Best identity preservation (cosine similarity)
   - `best_clean_openset.pth` - Best clean accuracy
   - `best_adv_openset.pth` - Best adversarial robustness
4. **Generate Visualizations**: Reconstruction examples saved to `vis/` directory
5. **Log Metrics**: Training metrics saved to CSV file for analysis

**Training Time**: ~3-4 days on RTX 4070 for AIAF-Tiny (50 epochs, 45K images)

---

## Project Structure

```
aiaf-defense/
│
├── train.py                      # Main training script
├── iresnet_arcface.py       # Face recognition backbone 
├── aiaf.py                  # AIAF (Tiny & Stable)
├── attack_wrappers.py       # Attack implementations
├── environment.yml             # Conda environment file
├── best_identity.pth         #best saved model           
└── README.md                        
```

---

## Models

### AIAF-Tiny
- **Parameters**: 2.5M
- **Architecture**: Lightweight encoder-decoder with gradient reversal
- **Latency**: 10.7ms average
- **Use Case**: Resource-constrained devices implementation(8GB VRAM)

### AIAF-Stable  
- **Parameters**: 11M
- **Architecture**: ResNet-18 based encoder-decoder
- **Latency**: 11.5ms average
- **Use Case**: Maximum performance (16GB+ VRAM)

**Both models include:**
- Encoder: Extracts identity-preserving features
- Decoder: Reconstructs cleaned images
- Gradient Reversal Layer: Forces attack-invariant learning
- Multi-task Loss: Balances reconstruction, identity, and attack detection

---

## Results

### Performance on VGGFace2 Test Set

| Attack Type | Clean Acc | Defense Acc | TAR@FAR (1%) | Latency |
|-------------|-----------|-------------|--------------|---------|
| Clean       | 95.2%     | 94.8%       | 0.892        | 10.7ms  |
| PGD (ε=8/255) | 12.3%   | 89.1%       | 0.812        | 11.2ms  |
| C&W         | 8.7%      | 86.4%       | 0.788        | 11.3ms  |
| AutoAttack  | 6.2%      | 82.3%       | 0.743        | 11.5ms  |
| Sibling     | 15.6%     | 91.7%       | 0.831        | 11.1ms  |
| Patch (30%) | 34.2%     | 45.8%       | 0.389        | 11.4ms  |
| Mask Overlay | 41.7%    | 52.3%       | 0.421        | 11.2ms  |

*Tested with AIAF-Tiny + ArcFace iResNet-100 on LFW dataset*

### Comparison with State-of-the-Art

| Method | Training Data | Digital Defense | Physical Defense | Latency |
|--------|---------------|-----------------|------------------|---------|
| Feature Distillation | 494K | 78% TAR@FAR | 42% TAR@FAR | ~50ms |
| PatchGuard | 1M | 71% TAR@FAR | 68% TAR@FAR | ~120ms |
| DeRandomized Smoothing | 494K | 83% TAR@FAR | 35% TAR@FAR | ~200ms |
| **AIAF (Ours)** | **45K** | **82% TAR@FAR** | **48% TAR@FAR** | **11ms** |

**Key Achievement**: 66× better data efficiency than existing methods while maintaining competitive performance.

---

## Hardware Requirements

### Minimum Configuration
- GPU: NVIDIA RTX 3080 (12GB VRAM)
- CPU: 8 cores
- RAM: 32GB
- Storage: 50GB free space

### Recommended Configuration  
- GPU: NVIDIA RTX 4070 or better (16GB+ VRAM)
- CPU: 16 cores
- RAM: 64GB
- Storage: 100GB SSD

### Tested Configurations
- **Development**: RTX 4070 (8GB) - AIAF-Tiny with batch 64
- **Large-scale**: AWS p4d.24xlarge (8× A100) - AIAF-Stable with batch 128

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{aiaf2025,
  title={AIAF: Attack-Invariant Attention Filter for Real-Time Adversarial Defense in Face Recognition},
  author={Kamalesh [Your Last Name]},
  year={2025},
  school={University of Adelaide},
  type={Master's Thesis}
}
```

---

## Acknowledgments

- **VGGFace2 Dataset**: Visual Geometry Group, University of Oxford
- **Face Recognition Models**: [InsightFace](https://github.com/deepinsight/insightface) - ArcFace, CosFace, MagFace implementations
- **Attack Libraries**: 
  - [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) - PGD, C&W implementations
  - [AutoAttack](https://github.com/fra31/auto-attack) - Ensemble attack framework
  - [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - Additional attack methods
- **Perceptual Loss**: [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - Learned perceptual similarity metric

---

## Contact

**Author**: Kamalesh Gunasekaran 
**Email**: a1901114@adelaide.edu.au  
**Institution**: University of Adelaide, South Australia 
**Supervisor**: DR.Hussain Ahmad
