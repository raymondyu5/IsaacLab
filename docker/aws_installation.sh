# # install package for the usd
# sudo apt-get update
# sudo apt-get install git-lfs
# git lfs install

# # create data
# sudo mkdir /data
# sudo mount /dev/nvme2n1 /data
# sudo chmod -R 777 /data
# cd /data/IsaacLab
# git pull
# git lfs pull

# # install nvidia driver 
sudo apt-get update
sudo apt install build-essential -y
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run
chmod +x NVIDIA-Linux-x86_64-535.129.03.run
sudo ./NVIDIA-Linux-x86_64-535.129.03.run 
sudo docker pull lemonla/deform:latest