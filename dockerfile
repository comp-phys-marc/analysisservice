FROM python:3.6
ADD requirements.txt /app/requirements.txt
ADD ./analysisservice/ /app/
WORKDIR /app/
RUN pip install -r requirements.txt
ENTRYPOINT python app.py