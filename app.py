from flask import Flask, request
from io import BytesIO
import re
import base64
import json
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from groundingdino.util.inference import Model

GREEN_COLOR = "\033[92m"
END_COLOR = "\033[0m"

CONFIDENCE_THRESHOLD = 0.5

# GroundingDINO configuration
MODEL_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT_PATH = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"
DINO_model = Model(MODEL_CONFIG_PATH, MODEL_CHECKPOINT_PATH, "cpu")
CLASSES = json.load(open('furniture_list.json'))
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

print(f"{GREEN_COLOR}GroundingDINO loaded{END_COLOR}")

# BLIP Image Captioning Large configuration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BLIP_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

print(f"{GREEN_COLOR}BLIP Image Captioning Large loaded{END_COLOR}")

app = Flask("DesAIgner's Image and Links API")

@app.post("/")
def post():
  data = request.get_data()
  if not valid_base64_encoded_image(data):
    return "Input was not a valid base64 string for an image", 400
  data = base64.b64decode(data.split(b',')[1])
  img = Image.open(BytesIO(data))

  if img.mode != 'RGB':
    img.convert('RGB')
  
  detections = get_detections(img)

  prompts = []

  for detection in detections:
    im = img.crop(detection["xyxy"])
    
    prompts.append(get_prompt(im))

  output = []
  for detection, prompt in zip(detections, prompts):
    output.append({
      "box": detection["xyxy"],
      "prompt": prompt
    })

  return sorted(output, key=area)


def get_detections(img):
  img_source = np.asarray(img)
  detections = DINO_model.predict_with_classes(img_source, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD)
  filter_function = lambda detection: detection["confidence"] > CONFIDENCE_THRESHOLD
  return list(filter(filter_function, parse_detections(detections)))

def parse_detections(detections):
  output = []
  for xyxy, confidence in zip(detections.xyxy, detections.confidence):
    xyxy = list(map(round, xyxy))
    confidence = float(confidence)
    output.append({
        "xyxy": xyxy,
        "confidence": confidence
    })
  return output

def get_prompt(img):
  text = "a detailed description of the furniture is a "
  inputs = processor(img, text, return_tensors="pt")

  out = BLIP_model.generate(**inputs)
  decoded = processor.decode(out[0], skip_special_tokens=True)
  return decoded.removeprefix(text)

def area(detection):
  xyxy = detection["box"]
  width = xyxy[2] - xyxy[0]
  height = xyxy[1] - xyxy[3]
  return width * height

def valid_base64_encoded_image(image):
  base64_regex = b'^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}={2})$'
  metadata_regex = b'^data:image/.+;base64$'
  metadata, data = image.split(b",")
  base64_match = re.match(base64_regex, data)
  metadata_match = re.match(metadata_regex, metadata)
  return metadata_match != None and base64_match != None
