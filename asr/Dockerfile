FROM registry.hf.space/ithuan-formosan-asr:latest

COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app.py models.yaml DEMO.md ./
EXPOSE 7860
CMD ["python", "app.py"]
