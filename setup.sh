pip install pipenv
pipenv install fastapi transformers numpy python-multipart Pillow typing-extensions uvicorn # API dependencies
git clone https://github.com/IDEA-Research/GroundingDINO.git
pipenv install torch
pipenv install -e ./GroundingDINO # Grounding DINO dependencies
cd GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..