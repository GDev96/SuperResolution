Dataset_step2_1,33 creare il mosaico basta avere la cartella del osservatorio 
Dataset_step2_42 creare il mosaico basta avere la cartella del hubble 

lo step 2 non incrocia hubble e osservatorio, verranno incrociati nel file 3 per fare le patch


MODELLO

TUTTI I FILE COPY SONO QUELLI PER LA PIPELINE LIGHT 128'':512

TUTTI I FILE NORMALE SONO QUELLI PER LA PIPELINE LIGHT 80'':512


SEMPRE 

python scripts\Modello_0_creazioneenv.py



solo la prima volta

python scripts\Modello_1_setup_environment.py


HEAVY


python scripts\Modello_2_prepare_data.py

python scripts\Modello_3_verifica.py


python scripts\Modello_4_train_heavy.py

python scripts\Modello_5_inference.py


python scripts\Modello_5_inference.py ^


tensorboard --logdir=outputs/M33/tensorboard


LIGHT 


python scripts\Modello_2_prepare_data_copy.py

python scripts\Modello_3_verifica_copy.py

python scripts\Modello_4_train_light.py

python scripts\Modello_5_inference_copy.py  DARE IN INPUT ALTRO CHIEDERE CHAT



tensorboard --logdir=outputs/M33/tensorboard_light



pip install tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy pyyaml lmdb addict future yapf "numpy<2.0"
pip install opencv-python lmdb addict future yapf
pip install einops timm



Versione HAT-Light per architettura se crusha