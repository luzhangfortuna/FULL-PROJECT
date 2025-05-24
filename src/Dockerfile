FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . src/

EXPOSE 45450

ENTRYPOINT ["gunicorn", "src.app:app"]
