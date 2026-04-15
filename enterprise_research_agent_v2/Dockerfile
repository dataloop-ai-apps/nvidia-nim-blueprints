FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.11_opencv

USER 1000
ENV HOME=/tmp

RUN pip install --no-cache-dir \
    deepagents \
    langgraph \
    langchain_core \
    langchain_nvidia_ai_endpoints \
    tavily-python \
    jinja2 \
    requests \
    pydantic

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints-v2:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints-v2:0.0.1
