#!/bin/sh

apt-get update
apt-get install -y --no-install-recommends ocl-icd-libopencl1 clinfo ocl-icd-opencl-dev

mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

wget https://github.com/dragmz/rsagg/releases/download/dev-linux/bacon
chmod +x bacon

./bacon optimize --time 60000
