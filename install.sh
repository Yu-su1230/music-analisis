#!/bin/bash

# Anaconda
sudo apt-get install -y wget
cd ~/
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
chmod 766 Anaconda3-4.2.0-Linux-x86_64.sh
./Anaconda3-4.2.0-Linux-x86_64.sh

# Tensorflow
conda install -c conda-forge tensorflow=1.1.0

# Keras
sudo pip3 install keras==2.0.5

# nano
apt-get install nano

# keras configration file
mkdir ~/.keras
cat << EOS > ~/.keras/keras.json
{
	"image_dim_ordering": "tf",
	"epsilon": 1e-07,
	"floatx": "float32",
	"backend": "tensorflow"
}

EOS

# librosa
conda install -c conda-forge librosa
