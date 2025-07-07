# Note: Docker does not accept single quotes ('...'). Use "..." instead.

#   ---

# Starts from a minimal Python 3.12 image.
# "Slim" = smaller, faster, but requires explicitly installing any missing tools or libraries.
FROM python:3.12.1-slim

# All following operations (COPY, RUN, etc.) will happen in /app inside the container (it will created if it doesn't exist)
WORKDIR /app

# this command is optional but recommended because base images often come with outdated pip versions
# that may fail to install newer packages and lack support for modern features. 
# So, we'll skip it here because we're using a recent Python image (3.12.1) → most probably pip is up-to-date
# RUN pip install -U pip

# Required since the project uses Pipfile and Pipfile.lock instead of requirements.txt
RUN pip install pipenv

# we copy these first alone to leverage Docker layer caching.
# Docker caches each instruction, so if Pipfile and Pipfile.lock haven't changed, it will skip re-installing dependencies.
# Speeds up builds dramatically when only app code changes.
COPY ["Pipfile", "Pipfile.lock", "./"]

# Installs packages directly into the system Python environment (no virtualenv).
# --deploy makes sure the Pipfile.lock is valid and respected strictly.
RUN pipenv install --system --deploy

# This copies the actual application logic and the model file
COPY ["lin_reg.bin", "predict.py", "./"]

# Informs Docker that the app listens on port 9696
EXPOSE 9696

# this line defines what the container does when it runs (always at the end of the file) 
# Runs the predict:app Flask app using Gunicorn on port 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

# Use this command when building this docker image: `docker build -t ride-duration-prediction-service:v1 .`
# Use this command when running this docker image : `docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1`