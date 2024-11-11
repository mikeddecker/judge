# install torch first before denspose (required)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/Flode-Labs/vid2densepose.git

cd vid2densepose

pip install -r requirements.txt gradio==4.44.0

cd ..

git clone https://github.com/facebookresearch/detectron2.git

# then simply run:
# gradio app.py