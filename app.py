from fastapi import FastAPI, Response, Request, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict
from io import BytesIO
import json
import requests
import time
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

ORIGINS = ["https://desaigner.vercel.app/create"]
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"], 
    allow_credentials=True
)

@app.middleware("http")
async def append_process_time_header(request: Request, call_next):
  start_time = time.now()
  response = await call_next(request)
  process_time = time.time() - start_time
  response.headers["X-Process-Time"] = time.strftime("%M:%S", time.gmtime(process_time))
  return response

class Mueble(TypedDict):
  box: tuple[int, int, int, int]
  prompt: str
  links: tuple[str, str, str]

# main endpoint
@app.post("/")
async def root(image: UploadFile, response: Response) -> list[Mueble]:
  data = image.file.read()
  img = Image.open(BytesIO(data))

  if img.mode != 'RGB':
    img.convert('RGB')
  
  detections = get_detections(img)

  prompts = []

  for detection in detections:
    im = img.crop(detection["xyxy"])
    class_name = 'furniture'
    if detection['class']:
      class_name = CLASSES[detection['class']]
    prompt_in_english = get_prompt(im, class_name)
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
    output.append(Mueble(
      box=tuple(detection["xyxy"]),
      prompt=prompt,
      links=tuple(links)
    ))

  return sorted(output, key=area)

# Health check for render.com
@app.get("/health")
async def health_check(response: Response):
  response.status_code = status.HTTP_200_OK
  return "200 OK"

def get_detections(img):
  img_source = np.asarray(img)
  detections = DINO_model.predict_with_classes(img_source, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD)
  filter_function = lambda detection: detection["confidence"] > CONFIDENCE_THRESHOLD
  return list(filter(filter_function, parse_detections(detections)))

def parse_detections(detections):
  output = []
  for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
    xyxy = list(map(round, xyxy))
    confidence = float(confidence)
    output.append({
        "xyxy": xyxy,
        "confidence": confidence,
        "class": class_id
    })
  return output

def get_prompt(img, class_name):
  text = "a detailed description of the furniture is a "
  inputs = processor(img, text, return_tensors="pt")

  out = BLIP_model.generate(**inputs)
  decoded = processor.decode(out[0], skip_special_tokens=True)
  prompt = decoded.removeprefix(text)
  return prompt[0].upper() + prompt[1:]

def get_links_list(prompts):
  headers = { 'Authorization': f'Bearer {MERCADO_LIBRE_KEY}' }
  out = []

  for prompt in prompts:
    params = {
      'status': 'active',
      'site_id': 'MLA',
      'limit': 3,
      'q': prompt
    }
    response = requests.get(f'{MERCADO_LIBRE_URL}/products/search', params, headers=headers)

    if response.status_code != 200:
      print(f"Mercado Libre API failed with status code {response.status_code}")
      print(response)
      return "Mercado Libre API failed"

    top3 = response.json()['results'][:3]
    links = [f'https://mercadolibre.com.ar/p/{item["id"]}' for item in top3]
    while len(links) < 3:
      links.append("No hay link")
    out.append(links)

  return out

def area(detection):
  xyxy = detection["box"]
  width = xyxy[2] - xyxy[0]
  height = xyxy[1] - xyxy[3]
  return width * height
