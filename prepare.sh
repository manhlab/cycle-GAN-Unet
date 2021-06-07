mkdir saved_models
mkdir saved_models/pix2pix
mkdir dataset
mkdir horse2zebra
bash src/download_cyclegan_dataset.sh horse2zebra
pip install -r requirements.txt