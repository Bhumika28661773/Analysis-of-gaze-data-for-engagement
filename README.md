# Analysis-of-gaze-data-for-engagement

How To Run the code?

pip install opencv-python
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
pip install -e ".[demo]"
cd checkpoints && \
./download_ckpts.sh && \
cd ..
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121