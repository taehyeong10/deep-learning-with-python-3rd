FROM python:3.12-slim

# 노트북의 셸 명령(!wget, !unzip, !tar)에 필요한 도구.
# g++는 torch.compile이 첫 호출 때 C++ 코드를 컴파일하는 데 필요하다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git wget unzip tar g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    # 세 프레임워크가 한 GPU를 나눠 쓰도록 메모리 선점을 끈다
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# TensorFlow는 pip로 설치된 CUDA 라이브러리(site-packages/nvidia/*/lib)를 스스로 찾지 못해
# GPU를 인식하지 못한다. 동적 링커 캐시에 등록해 준다.
RUN python -c "import glob, sysconfig; print('\n'.join(sorted(glob.glob(sysconfig.get_paths()['purelib'] + '/nvidia/*/lib'))))" \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && test -s /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

COPY check_gpu.py /opt/check_gpu.py

WORKDIR /workspace
EXPOSE 8888

# 접속 토큰은 JUPYTER_TOKEN 환경 변수로 전달한다 (jupyter가 직접 읽음)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
