FROM python:3.11.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8080
EXPOSE 8000

VOLUME [ "/data" ]

CMD ["bash", "run.sh"]