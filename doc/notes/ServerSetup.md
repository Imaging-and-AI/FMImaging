# General information

### Mounting

The shared network drives are mounted under /export.

### Install CUDA
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### Install CUDNN
```
# the pre-saved deb file is at /export/Lab-Xue/software
# Instructions are at https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

cp /export/Lab-Xue/software/*.deb ~/software
cd ~/software
```

### Install docker

Follow the instructions at : https://docs.docker.com/engine/install/ubuntu/
```
sudo usermod -aG docker $USER
```

Install nvidia-container-toolkit
```
sudo su
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
apt update
apt -y install nvidia-container-toolkit
systemctl restart docker
exit
```

### Install tensorRT
```
# Instructions at https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
sudo apt-get install tensorrt
pip3 install --upgrade tensorrt
```
