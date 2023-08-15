from fastapi import FastAPI, Response, status, UploadFile
from pydantic import BaseModel
from io import BytesIO
import re
import base64
import json
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from groundingdino.util.inference import Model

load_dotenv()

TRANSLATE_URL = 'https://clients5.google.com/translate_a/t'

MERCADO_LIBRE_URL = 'https://api.mercadolibre.com/'
MERCADO_LIBRE_KEY = os.getenv('MERCADO_LIBRE_KEY')

GREEN_COLOR = "\033[92m"
END_COLOR = "\033[0m"

CONFIDENCE_THRESHOLD = 0 # TODO: Set this to a reasonable value

# GroundingDINO configuration
print(f"{GREEN_COLOR}Loading GroundingDINO...{END_COLOR}")
MODEL_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT_PATH = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"
DINO_model = Model(MODEL_CONFIG_PATH, MODEL_CHECKPOINT_PATH, "cpu")
CLASSES = json.load(open('furniture_list.json'))
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
print(f"{GREEN_COLOR}GroundingDINO loaded{END_COLOR}")

# BLIP Image Captioning Large configuration
print(f"{GREEN_COLOR}Loading BLIP Image Captioning Large...{END_COLOR}")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BLIP_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
print(f"{GREEN_COLOR}BLIP Image Captioning Large loaded{END_COLOR}")

print(f"{GREEN_COLOR}Done Loading Models{END_COLOR}")

app = FastAPI(title="DesAIgner's Image and Links API")

@app.post("/")
async def root(image: UploadFile, response: Response):
  data = image.file.read()
  img = Image.open(BytesIO(data))

  if img.mode != 'RGB':
    img.convert('RGB')
  
  detections = get_detections(img)

  prompts = []

  for detection in detections:
    im = img.crop(detection["xyxy"])
    prompt_in_english = get_prompt(im)
    result = requests.get(f'{TRANSLATE_URL}?client=dict-chrome-ex&sl=en&tl=es&q={prompt_in_english}')
    if (result.status_code != 200):
      print(f"Translation failed with status code {result.status_code}")
      print(result)
      response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
      return "Translation failed"
    prompt = result.json()[0]
    prompts.append(prompt)  

  links_list = get_links_list(prompts)

  if links_list == "Mercado Libre API failed":
    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return "Mercado Libre API failed"

  output = []
  for detection, prompt, links in zip(detections, prompts, links_list):
    output.append({
      "box": detection["xyxy"],
      "prompt": prompt,
      "links": links
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

def get_links_list(prompts):
  headers = { 'Authorization': f'Bearer {MERCADO_LIBRE_KEY}' }
  out = []

  for prompt in prompts:
    params = {
      'status': 'active',
      'site_id': 'MLA',
      'q': prompt
    }
    response = requests.get(f'{MERCADO_LIBRE_URL}/products/search', params, headers=headers)

    if response.status_code != 200:
      print(f"Mercado Libre API failed with status code {response.status_code}")
      print(response)
      return "Mercado Libre API failed"

    top3 = response.json()['results'][:3]
    links = [f'{MERCADO_LIBRE_URL}/p/{item["id"]}' for item in top3]
    out.append(links)

  return out

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
