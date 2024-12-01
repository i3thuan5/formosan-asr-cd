FROM registry.hf.space/ithuan-formosan-tts:latest

RUN pip install gradio==4.44.1

COPY models.yaml configs/models.yaml

EXPOSE 7860
CMD ["python", "app.py"]
