FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV PORT=7860
ENV DASH_DEBUG=0
# Make the project root importable as a package
ENV PYTHONPATH=/app

CMD ["gunicorn", "indra_variants.app.variant_network:server", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120", \
     "--preload"]
