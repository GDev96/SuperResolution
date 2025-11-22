

python scripts\Modello_2_prepare_data.py

python scripts\Modello_3_verifica.py


python scripts/Modello_4_train_heavy_tensor.py --target M33

python scripts\Modello_5_inference.py




tensorboard --logdir=outputs/M33/tensorboard


tensorboard --logdir /root/SuperResolution/outputs --port 6006 --bind_all






pip install tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy pyyaml lmdb addict future yapf "numpy<2.0"
pip install opencv-python lmdb addict future yapf
pip install einops timm



Versione HAT-Light per architettura se crusha