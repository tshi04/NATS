wget http://us.download.nvidia.com/tesla/384.125/NVIDIA-Linux-x86_64-384.125.run
chmod u+x NVIDIA-Linux-x86_64-384.125.run
sudo ./NVIDIA-Linux-x86_64-384.125.run
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo apt install nvidia-cuda-toolkit

sudo pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
sudo pip3 install torchvision
