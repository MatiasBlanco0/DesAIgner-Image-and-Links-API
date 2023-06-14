from flask import Flask, request
import re

app = Flask("DesAIgner's Image and Links API")

@app.post("/")
def post():
  data = request.get_data()
  if not valid_encoded_image(data):
    return "Input was not a valid base64 string for an image", 400
  return data


def valid_encoded_image(image):
  regex = b'^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}={2})$'
  metadata, data = image.split(b",")
  match = re.match(regex, data)
  return metadata == b'data:image/jpeg;base64' and match != None