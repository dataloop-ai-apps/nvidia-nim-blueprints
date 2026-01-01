FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
ENV HOME=/tmp
RUN pip install langgraph langchain_community langchain_core tavily-python langchain_nvidia_ai_endpoints elevenlabs

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints:0.0.3 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints:0.0.3

