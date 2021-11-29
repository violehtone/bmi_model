# Set base image
FROM python:3.8.0-slim-buster

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

# Install python packages from requirements file
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy files to /app directory
COPY utils/ utils/
COPY src/ src/
COPY data/ data/

# Run the API
CMD ["python", "src/api.py"]