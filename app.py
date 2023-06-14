from flask import Flask, request

app = Flask("DesAIgner's Image and Links API")

@app.post("/")
def post():
    data = request.json

