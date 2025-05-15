# 🧼 Document Image Denoising using Convolutional Autoencoder

This project implements a convolutional autoencoder in PyTorch to clean scanned documents with heavy noise, using the "Denoising Dirty Documents" dataset from Kaggle. The model learns to reconstruct clean document images from their noisy counterparts.

---

## 🧠 Model Architecture

The denoising model is a **Convolutional Autoencoder** with the following structure:

### 🔹 Encoder

- `Conv2d(3 → 16, kernel_size=3, stride=2, padding=1)` → ReLU
- `Conv2d(16 → 32, kernel_size=3, stride=2, padding=1)` → ReLU
- `Conv2d(32 → 64, kernel_size=3, stride=2, padding=1)` → ReLU

### 🔹 Decoder

- `ConvTranspose2d(64 → 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1))` → ReLU
- `ConvTranspose2d(32 → 16, kernel_size=3, stride=2, padding=1, output_padding=(1,1))` → ReLU
- `ConvTranspose2d(16 → 3, kernel_size=3, stride=2, padding=1, output_padding=(1,0))`  
  _(Outputs 3-channel RGB image)_

---

## 📚 Dataset

- Dataset: [Denoising Dirty Documents (Kaggle)](https://www.kaggle.com/competitions/denoising-dirty-documents)
- Format: Paired noisy/clean images
- Preprocessing:
  - Resized to **(420, 540)**
  - Converted to **RGB**
  - Normalized using ImageNet stats:
    - `mean = [0.485, 0.456, 0.406]`
    - `std  = [0.229, 0.224, 0.225]`

---

## ⚙️ Training Configuration

| Hyperparameter      | Value          |
|---------------------|----------------|
| Loss Function       | `nn.BCELoss()` |
| Optimizer           | `Adam`         |
| Learning Rate       | `1e-3`         |
| Batch Size          | 16             |
| Epochs              | 20 (can adjust)|
| Image Size          | 420x540        |
| Input Channels      | 3 (RGB)        |

---

## 📁 Directory Structure

document-denoising-model.ipynb # Main notebook for training & testing
train_dataset/ # Noisy images
train_cleaned_dataset/ # Clean labels
test_dataset/ # Noisy test inputs
output/ # Output folder for denoised results


---

## 🏃 How to Run

1. Download the dataset from Kaggle and place it in the correct folders:
    - `train_dataset/train`
    - `train_cleaned_dataset/train_cleaned`
    - `test_dataset/test`

2. Open and run the Jupyter notebook:
    ```
    document-denoising-model.ipynb
    ```

3. The notebook will:
    - Load & preprocess images
    - Train the autoencoder
    - Visualize input vs. denoised outputs

---

## 🔍 Sample Results

| Noisy Input        | Denoised Output       |
|--------------------|-----------------------|
| ![input](examples/input_1.png) | ![output](examples/output_1.png) |

_(add example visuals if available)_

---

## 🛠 Tools & Libraries

- **PyTorch**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Pillow**
- **tqdm**
- Jupyter Notebook

---

## ✍️ Author

**Nguyen Minh**  
`https://github.com/MinhCong0308`

---

## 📌 Acknowledgments

- Kaggle competition: [Denoising Dirty Documents](https://www.kaggle.com/competitions/denoising-dirty-documents)
- Inspiration from CNN-based denoising and image reconstruction literature.

