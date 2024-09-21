# face detection
import os
from ultralytics import YOLO

MODEL_PATH = os.path.abspath(os.curdir)+"/yolo-chonnochara-detection"

# Load a pretrained model
model = YOLO(MODEL_PATH+'/chonnochara-best.pt')

# Run inference on the source
results = model(source=0, show=True, conf=0.6, save=True)