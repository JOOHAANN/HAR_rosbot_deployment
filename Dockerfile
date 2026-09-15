ARG BASE_IMAGE=clipgcn-rosbot:jazzy
FROM ${BASE_IMAGE}

# The coordinator normally runs in this HAR container so it can consume the
# person-follow messages and the VPOCLIP streams.  Keep Nav2's action type
# available here; the actual Nav2 servers remain in the separate
# jazzy-rosbot container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-nav2-msgs \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    LD_LIBRARY_PATH=/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/clipgcn-venv/lib/python3.12/site-packages/nvidia/nvtx/lib

# The inherited image already supplies CUDA/PyTorch/ROS2/YOLO/RTMPose/ONNX
# Runtime. These small packages cover the archived X3D and DDQN import paths.
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check \
    einops==0.8.0 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.10 \
    openpyxl==3.1.5 \
    simplejson==3.20.2 \
    timm==1.0.11 \
    yacs==0.1.8

WORKDIR /workspace/CLIPGCN
COPY . /workspace/CLIPGCN

RUN chmod +x /workspace/CLIPGCN/scripts/*.sh \
    /workspace/CLIPGCN/*.py

# Match VPOCLIP_rosbot: the container is a persistent ROS/ML environment and
# the inference process is started with docker exec after selecting a robot.
CMD ["sleep", "infinity"]
