## Docker Deployment

### Prerequisites
- Docker Desktop installed
- 4GB+ RAM available

### Quick Start
```bash
# Build the image
docker build -t persona-vectors-dashboard .

# Run the container
docker run -p 8501:8501 persona-vectors-dashboard

# Access dashboard
http://localhost:8501
```

### Docker Commands Reference
```bash
# Build image
docker build -t persona-vectors-dashboard:latest .

# Run container (foreground)
docker run -p 8501:8501 persona-vectors-dashboard

# Run container (background)
docker run -d -p 8501:8501 --name persona-dashboard persona-vectors-dashboard

# Stop container
docker stop persona-dashboard

# Remove container
docker rm persona-dashboard

# View logs
docker logs persona-dashboard

# List running containers
docker ps

# List all containers
docker ps -a

# Remove image
docker rmi persona-vectors-dashboard
```

### Troubleshooting

**Port already in use:**
```bash
docker run -p 8502:8501 persona-vectors-dashboard
# Then access: http://localhost:8502
```

**Container won't stop:**
```bash
docker ps
docker kill <container-id>
```

### Technical Details

- **Base Image:** python:3.10-slim
- **Image Size:** ~187MB
- **Port:** 8501 (Streamlit default)
- **Working Directory:** /app