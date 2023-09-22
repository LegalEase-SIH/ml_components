FROM python:3.11-slim

# Install libraries
# COPY /home/dhairya/projects/sih/ml_components/classifier/server/ ./
COPY ./requirements.txt ./
RUN pip install -r requirements.txt && \
    rm ./requirements.txt

# Setup container directories
WORKDIR /server

# Copy local code to the container
COPY classifier/server/app/ /server/app
COPY classifier/server/model/ /server/model

# launch server with gunicorn
EXPOSE 8080
CMD ["python", "app/api.py"]
# CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
#      "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]