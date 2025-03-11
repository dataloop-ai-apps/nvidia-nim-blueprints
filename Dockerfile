FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv

USER 1000
ENV HOME=/tmp
RUN pip install langgraph langchain_community langchain_core tavily-python langchain_nvidia_ai_endpoints

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/nvidia-nim-blueprints:0.0.1

