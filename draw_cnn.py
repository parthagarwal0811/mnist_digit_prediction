import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# ---------------------------
# CNN Model (same as training)
# ---------------------------
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------------------
# Load trained model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_best.pth", map_location=device))
model.eval()

# ---------------------------
# Drawing Pad
# ---------------------------
canvas = np.zeros((400, 400), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 12, 255, -1)  # draw thick white strokes
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw a digit")
cv2.setMouseCallback("Draw a digit", draw)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

while True:
    cv2.imshow("Draw a digit", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'): # clear canvas 
        canvas[:] = 0
    
    if key == ord('q'): # quit
        break

    if key == ord('r'):  # recognize
        # Find bounding box of drawn digit
        coords = cv2.findNonZero(canvas)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit = canvas[y:y+h, x:x+w]

            # Resize to fit MNIST style
            digit = cv2.resize(digit, (20, 20))

            # Place on 28x28 canvas, centered
            canvas28 = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2
            y_offset = (28 - 20) // 2
            canvas28[y_offset:y_offset+20, x_offset:x_offset+20] = digit

            # Invert colors (MNIST style)
            canvas28 = 255 - canvas28

            # Transform and predict
            d = transform(canvas28).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(d)
                _, pred = torch.max(output, 1)
                print("Recognized Digit:", pred.item())


cv2.destroyAllWindows()
