# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file
COPY dashboard_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r dashboard_requirements.txt

# Copy application files
COPY Dashboard.py .
COPY data/ data/
COPY figures/ figures/

# Expose port 8501 (Streamlit default)
EXPOSE 8501

# Health check (optional but professional!)
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the dashboard
CMD ["streamlit", "run", "Dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]