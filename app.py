from flask import Flask, request
from io import BytesIO
import re
import base64
from PIL import Image
import numpy as np
from clip_interrogator import Config, Interrogator
from groundingdino.util.inference import Model

MODEL_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT_PATH = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"
model = Model(MODEL_CONFIG_PATH, MODEL_CHECKPOINT_PATH, "cpu")
CLASSES = ["plant pot", "sofa", "table", "chair", "cushion", "lamp", "painting", "tea pot", "stool", "clock", "bed", "rug", "shelf", "desk", "cup"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

app = Flask("DesAIgner's Image and Links API")

@app.post("/")
def post():
  data = request.get_data()
  if not valid_encoded_image(data):
    return "Input was not a valid base64 string for an image", 400
  data = base64.b64decode(data.split(b',')[1])
  img = Image.open(BytesIO(data))

  if img.mode != 'RGB':
    img.convert('RGB')
  
  img_source = np.asarray(img)
  detections = get_detections(img_source)

  return sorted(detections, key=area)


def get_detections(img):
  detections = model.predict_with_classes(img, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD)
  return parse_detections(detections)

def parse_detections(detections):
  output = []
  for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
    output.append({
        "xyxy": xyxy,
        "confidence": confidence,
        "class_id": class_id
    })
  return output

def area(detection):
  xyxy = detection["xyxy"]
  width = xyxy[2] - xyxy[0]
  height = xyxy[1] - xyxy[3]
  return width * height

def valid_encoded_image(image):
  base64_regex = b'^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}={2})$'
  metadata_regex = b'^data:image/.+;base64$'
  metadata, data = image.split(b",")
  base64_match = re.match(base64_regex, data)
  metadata_match = re.match(metadata_regex, metadata)
  return metadata_match != None and base64_match != None