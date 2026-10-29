FROM python:3.11-slim

# LightGBM üçün OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
# glibc arena-ları məhdudlaşdır: thread-lər çox olanda RSS-in şişməsinin qarşısını alır (Render free = 512 MB)
ENV MALLOC_ARENA_MAX=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh

EXPOSE 8000 8501
CMD ["sh", "scripts/start_cloud.sh"]
