FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
ENV HOME=/tmp

RUN pip install --no-cache-dir \
    langchain_core \
    langchain_nvidia_ai_endpoints \
    tavily-python \
    pubchempy==1.0.4 \
    rcsb-api==1.2.0

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints-biomedical:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints-biomedical:0.0.1
