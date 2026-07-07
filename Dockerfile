FROM python:3.11-slim

WORKDIR /app

# System deps for PySpark (Java) and pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Cloud deps are optional — install the ones you need by passing --build-arg
# Example: docker build --build-arg CLOUD_EXTRAS=s3 .
ARG CLOUD_EXTRAS=""
RUN if [ -n "$CLOUD_EXTRAS" ]; then \
        pip install --no-cache-dir $(grep -E "^(boto3|azure|google)" requirements-cloud.txt | \
            grep -E "$CLOUD_EXTRAS" | tr '\n' ' '); \
    fi

COPY . .

# Streamlit dashboard
EXPOSE 8501
# Pipeline runner (no dashboard)
EXPOSE 4040

# Default: launch dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
