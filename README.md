# Fashion MNIST Image Classifier

A neural network built with TensorFlow/Keras that classifies grayscale images of clothing items from the Fashion MNIST dataset. The project reaches about **88.5% accuracy** on the test set.

[Open in Colab](https://colab.research.google.com/github/LalTamatar/Hackathon-file/blob/main/fashion_mnist.ipynb)

## Dataset

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is loaded directly through `keras.datasets.fashion_mnist`.

- 60,000 training images and 10,000 test images
- Each image is 28x28 pixels, grayscale (values 0-255)
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Workflow

1. **Load data** from Keras and inspect shapes and pixel ranges.
2. **Explore** sample images with Matplotlib.
3. **Preprocess**
   - Scale pixel values to the 0-1 range (divide by 255)
   - Add a channel dimension so inputs are shaped `(28, 28, 1)`
4. **Build the model** (see below).
5. **Compile and train** with early stopping.
6. **Evaluate** on the test set.

## Model Architecture

| Layer | Purpose |
|---|---|
| `Input(28, 28, 1)` | Input image |
| `RandomRotation(0.02)` | Data augmentation, small random rotations |
| `RandomTranslation(0.02, 0.02)` | Data augmentation, small random shifts |
| `Flatten` | Converts 28x28 image to a 784-length vector |
| `Dense(256, relu)` | Hidden layer |
| `Dropout(0.2)` | Reduces overfitting |
| `Dense(10, softmax)` | Class probabilities |

Total parameters: **203,530**

The augmentation layers are only active during training, so they do not affect predictions at test time.

## Training Configuration

- **Optimizer:** Adam (learning rate 0.001)
- **Loss:** Sparse categorical crossentropy
- **Metric:** Accuracy
- **Batch size:** 256
- **Epochs:** up to 100
- **Early stopping:** monitors `val_loss`, patience of 8, restores the best weights

## Results

| Metric | Value |
|---|---|
| Test accuracy | **88.51%** |
| Test loss | 0.3235 |

Training stopped early after 35 epochs.

Note: the test set is used as the validation set for early stopping, so the reported test accuracy is slightly optimistic. For a stricter evaluation, hold out a separate validation split from the training data.

## Requirements

- Python 3.9+
- TensorFlow 2.x (developed with 2.21.0)
- NumPy
- Pandas
- Matplotlib

Install with:

```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage

**In Colab:** click the badge above and run all cells. A GPU runtime (T4) is recommended but not required.

**Locally:**

```bash
git clone https://github.com/LalTamatar/Hackathon-file.git
cd Hackathon-file
jupyter notebook fashion_mnist.ipynb
```

## Possible Improvements

- Replace the dense network with a CNN (Conv2D + MaxPooling) to push accuracy above 91%
- Add a proper train/validation/test split
- Add a confusion matrix and per-class metrics
- Save the trained model and add an inference script
- Try learning rate scheduling or batch normalization

## Project Structure

```
.
├── fashion_mnist.ipynb   # Main notebook
└── README.md
```
