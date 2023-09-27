pip install virtualenv
virtualenv .venv
source .venv/bin/activate
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install torch
pip install -e . # Grounding DINO dependencies
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
pip install -e . # API dependencies