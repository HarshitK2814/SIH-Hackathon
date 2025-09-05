import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Add backend folder to path ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
if current_script_path not in sys.path:
    sys.path.append(current_script_path)

# --- 1. SETUP AND CONFIGURATION ---
MODEL_PATH = r"C:\Users\Harshit Kumar\Downloads\SIH Hackathon\backend\training_outputs\checkpoints\robcrop_20250904_231418\best_model.pth"
TEST_DATA_PATH = r"C:\Users\Harshit Kumar\Downloads\SIH Hackathon\data_split\test" #<-- IMPORTANT: Update this path!
BATCH_SIZE = 32

# Import the corrected model class
from backend.model import RobCropResNet50 

# --- 2. LOAD THE MODEL AND DATA ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model with the architecture that EXACTLY matches the checkpoint
model = RobCropResNet50(num_classes=11, pretrained=False) 

# Load the entire checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Extract only the model's state dictionary
model_weights = checkpoint['model_state_dict']

# Load the weights into the model. This should now work without any errors.
model.load_state_dict(model_weights)

model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# (The rest of the script is unchanged and will now work)

# --- Define data transforms ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load test data ---
test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
print(f"Evaluating on {len(test_dataset)} images in {len(class_names)} classes.")


# --- 3. RUN THE EVALUATION LOOP ---
print("\nRunning evaluation...")
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted_indices.cpu().numpy())
print("Evaluation complete.")

# --- 4. GENERATE AND DISPLAY THE REPORT ---
print("\n" + "="*60)
print("Classification Report".center(60))
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("\n" + "="*60)
print("Confusion Matrix".center(60))
print("="*60)
print("Displaying confusion matrix in a new window...")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 12))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')

plt.title("RoboCrop Model - Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nEvaluation script finished.")
