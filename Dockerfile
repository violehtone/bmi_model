# Set base image
FROM python:3.8.0-slim-buster

ENV DEBIAN_FRONTEND noninteractive

# Set working directory
WORKDIR /app

# Install python packages from requirements file
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy files to /app directory
COPY . /app

# Run the API
CMD ["python", "api.py"]