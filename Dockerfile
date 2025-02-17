FROM python:3.12

WORKDIR /

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY app/ /app
COPY data/ /data

CMD ["python3.12", "/app/main.py"]