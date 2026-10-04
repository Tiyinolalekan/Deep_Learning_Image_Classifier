# Deep_Learning_Image_Classifier
# CIFAR-10 Image Classifier (CNN in PyTorch)

A small convolutional neural network (CNN) built from scratch in PyTorch to classify images from the CIFAR-10 dataset. I built it as a hands-on way to learn how CNNs work, from loading the data through to training, evaluating, and inspecting predictions.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 colour images (32×32 pixels) in 10 classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

- 50,000 training images
- 10,000 test images

## Model architecture

```
Input:           3 × 32 × 32
Conv (3 → 16)   → 16 × 32 × 32  → ReLU → MaxPool → 16 × 16 × 16
Conv (16 → 32)  → 32 × 16 × 16  → ReLU → MaxPool → 32 × 8 × 8
Flatten         → 2,048 values
Fully connected → 128 values (ReLU)
Fully connected → 10 values (logits, one per class)
```

- Convolutions: 3×3 kernels, padding 1
- Pooling: 2×2 max pooling, stride 2

## Training setup

| Setting | Value |
| --- | --- |
| Loss function | `CrossEntropyLoss` (takes raw logits) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Input processing | `ToTensor()` only (no augmentation) |

## Results (baseline)

| Metric | Result |
| --- | --- |
| Final training loss | 0.7125 |
| Train accuracy | 86.49% |
| Test accuracy | 68.43% |
| Train/test gap | ~18 points |

Random guessing would score 10%, so the model has clearly learned useful features. The large gap between train and test accuracy shows **overfitting**: the model performs much better on images it trained on than on unseen ones.

## What the notebook covers

1. Setting up PyTorch and selecting a device (GPU if available)
2. Loading and inspecting CIFAR-10
3. Building the CNN and tracing the shapes through each layer
4. Training with DataLoaders, loss, backpropagation and the Adam optimizer
5. Evaluating on the test set (`model.eval()` and `torch.no_grad()`)
6. Comparing train accuracy against test accuracy
7. Visualising predictions on test images

## How to run

1. Open `Image_Classifier_CNN.ipynb` in [Google Colab](https://colab.research.google.com/) (or Jupyter).
2. Optional but recommended: enable a GPU in Colab (Runtime → Change runtime type).
3. Run the cells in order. CIFAR-10 downloads automatically on first run.

**Requirements:** Python 3, `torch`, `torchvision`, `matplotlib`

## Next steps

- Plot training loss per epoch
- Analyse the misclassified images and which classes get confused
- Reduce overfitting (data augmentation, dropout, a deeper network)
- Experiment with learning rate and batch size
- Try transfer learning with a pretrained model

## Key takeaway

Training changes the model. Testing measures the model. The test set is never used to update the weights.
