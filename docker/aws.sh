#!/bin/bash

# Path to your private key
KEY="entong.pem"

# The SSH command to connect to your EC2 instance
HOST="ubuntu@ec2-54-212-205-77.us-west-2.compute.amazonaws.com"

# SSH into the EC2 instance
ssh -i $KEY $HOST





sudo docker run --name deform --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
-e "PRIVACY_CONSENT=Y" \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
docker.io/lemonla/deform:latest