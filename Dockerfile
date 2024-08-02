FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH="7.0;7.5;8.0;8.6"

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-12.1/
# Ensure CUDA is correctly set up
ENV PATH /usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything/

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano vim=2:* ninja-build g++ -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything

# Install essential Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy

# Install segment_anything package in editable mode
RUN python -m pip install --no-cache-dir -e segment_anything

# Install GroundingDINO package in editable mode without build isolation
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

# Install additional Python dependencies
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

WORKDIR /home/appuser/Grounded-Segment-Anything/segment_anything_2

# Set PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/home/appuser/Grounded-Segment-Anything/segment_anything_2

# Debugging: Show current directory and contents
RUN echo "Current directory: $(pwd)" && ls -al

# Debugging: Show installed packages
RUN pip list

# Install the package in editable mode
RUN pip install -e . || { echo "pip install -e . failed"; exit 1; }

# ENTRYPOINT ["python", "/home/appuser/Grounded-Segment-Anything/grounded_sam_demo_batch.py"]

# Set default arguments using CMD
# CMD ["-i", "/tmp/shuo/input", "-o", "/tmp/shuo/output"]
