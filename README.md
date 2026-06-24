# MNIST Digit Prediction

An interactive drawing application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Draw a digit on a canvas and get real-time predictions with ~99% accuracy.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Real-time digit recognition** — Draw a digit and get instant predictions (0-9)
- **High accuracy** — CNN trained on MNIST achieves ~99% validation accuracy
- **Interactive canvas** — OpenCV-based drawing interface with smooth input handling
- **Automatic preprocessing** — Grayscale conversion, thresholding, resizing, normalization
- **Center-of-mass alignment** — Aligns drawn digits like official MNIST data for better accuracy
- **Pre-trained model included** — `mnist_cnn_best.pth` ready to use, no training required
- **Clean, simple UI** — Minimal dependencies, OpenCV window-based interface

## How It Works

1. **User draws a digit** on the canvas using the mouse
2. **Image preprocessing pipeline**:
   - Convert to grayscale
   - Apply binary thresholding to isolate the digit
   - Clean up noise and artifacts
   - Resize to 28×28 pixels (MNIST standard)
   - Apply center-of-mass alignment (centers the digit in the frame)
   - Normalize using MNIST statistics
3. **CNN prediction** — The trained model predicts the digit (0-9)
4. **Result displayed** — Prediction shown on screen in real-time

This preprocessing mirrors the official MNIST dataset structure, which is why the model generalizes well to hand-drawn inputs.

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/parthagarwal0811/mnist_digit_prediction.git
cd mnist_digit_prediction
```

2. **Create a virtual environment** (optional but recommended):
```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision opencv-python numpy
```

Or, if you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Drawing Application

Simply run the main script:
```bash
python draw_cnn.py
```

An OpenCV window will open with a white canvas. You can now:

- **Draw** — Use your mouse to draw a digit (0-9)
- **Get prediction** — The model predicts in real-time as you draw
- **Clear canvas** — Press `C` to clear and try again
- **Exit** — Press `Q` or close the window to quit

## Project Structure

```
mnist_digit_prediction/
├── draw_cnn.py               # Main application — drawing interface & real-time prediction
├── train_cnn_overfit.py      # Training script (reference, pre-trained model included)
├── mnist_cnn_best.pth        # Pre-trained CNN model checkpoint
├── README.md                 # This file
└── requirements.txt          # Python dependencies (if present)
```

## Model Architecture

The CNN is trained on the MNIST dataset (60,000 training images, 10,000 test images) with the following characteristics:

- **Input**: 28×28 grayscale images
- **Architecture**: Convolutional Neural Network (details in `train_cnn_overfit.py`)
- **Validation accuracy**: ~99%
- **Output**: Probability distribution over digits 0-9

The model learns to recognize various handwriting styles and stroke widths from the MNIST training set.

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | Latest | Neural network framework |
| Torchvision | Latest | Dataset utilities & image transforms |
| OpenCV (cv2) | Latest | Image processing & canvas interface |
| NumPy | Latest | Numerical operations |

**Minimum Python version**: 3.9

## Training (Optional)

A pre-trained model is included (`mnist_cnn_best.pth`), so you don't need to train. However, if you want to retrain or experiment:

```bash
python train_cnn_overfit.py
```

This script will:
- Load the MNIST dataset (auto-downloads if needed)
- Train a CNN model
- Save the best checkpoint to `mnist_cnn_best.pth`

Training typically takes several minutes depending on your GPU/CPU.

## Tips for Better Predictions

1. **Draw clearly** — Use a consistent stroke width and try to write digits similar to handwriting samples
2. **Center your digit** — The model aligns digits automatically, but centering helps
3. **Use appropriate size** — Your digit should take up a reasonable portion of the canvas, not too small
4. **Contrast matters** — Dark strokes on a light background (as the app provides) works best
5. **Speed-appropriate drawing** — Neither too fast nor too slow; natural handwriting speed is ideal

## Known Limitations

- Works best on digits similar to MNIST training data (handwritten style)
- Very stylized or printed digits may have lower accuracy
- Single digit only (0-9); does not handle multi-digit numbers
- Requires a keyboard to clear canvas (no GUI buttons yet)

## Troubleshooting

**Issue**: Model not found or loading error
- **Solution**: Ensure `mnist_cnn_best.pth` is in the same directory as `draw_cnn.py`

**Issue**: OpenCV window doesn't appear
- **Solution**: Ensure you have a display server running (on Linux, may need X11 forwarding for remote machines)

**Issue**: Poor prediction accuracy
- **Solution**: Try drawing more clearly and centering your digit; check that you're drawing on the canvas (white area)

## Future Enhancements

- [ ] Multi-digit recognition (e.g., numbers like "42")
- [ ] GUI improvements (buttons, sliders, better feedback)
- [ ] Model comparison (try different architectures)
- [ ] Confidence threshold — reject predictions below a certain confidence
- [ ] Export predictions to file
- [ ] Stylized digit support (printed, cursive, etc.)

## Performance Metrics

On the MNIST test set:
- **Validation Accuracy**: ~99%
- **Inference time**: <100ms per prediction (CPU)
- **Model size**: ~2-5 MB

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

## Acknowledgments

- **MNIST Dataset** — Yann LeCun et al., a foundational benchmark for digit recognition
- **PyTorch & OpenCV** — Excellent open-source libraries for deep learning and computer vision

## Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request.

## Author

**Parth Agarwal** — [GitHub Profile](https://github.com/parthagarwal0811)

---

**Enjoy using MNIST Digit Prediction!** If you find it useful, consider giving it a star ⭐
