# Docker Deployment Guide

Complete guide for deploying both dashboard versions via Docker.

---

## Two Dashboard Versions

### Dashboard v0 (Discovery Process)
- **Port:** 8501
- **Content:** Initial breakthrough + overfitting lessons
- **Image:** `fjordhauler/persona-vectors-dashboard:v0`

### Dashboard v1 (Complete Research)
- **Port:** 8502
- **Content:** Validated results across 3 models
- **Image:** `fjordhauler/persona-vectors-dashboard:v1`

---

## Prerequisites

- Docker Desktop installed
- 4GB+ RAM available
- Ports 8501 and 8502 available

---

## Quick Start - Docker Hub (Recommended)

Pull and run pre-built images:
```bash
# Dashboard v0 (Discovery)
docker run -d -p 8501:8501 --name dashboard-v0 fjordhauler/persona-vectors-dashboard:v0

# Dashboard v1 (Complete Research)
docker run -d -p 8502:8502 --name dashboard-v1 fjordhauler/persona-vectors-dashboard:v1
```

**Access dashboards:**
- v0: http://localhost:8501
- v1: http://localhost:8502

---

## Building From Source

### Build Both Versions
```bash
# Build v0
docker build -f dashboard/Dockerfile.v0 -t persona-vectors-dashboard:v0 .

# Build v1
docker build -f dashboard/Dockerfile.v1 -t persona-vectors-dashboard:v1 .
```

### Run Locally
```bash
# Run v0
docker run -d -p 8501:8501 --name dashboard-v0 persona-vectors-dashboard:v0

# Run v1
docker run -d -p 8502:8502 --name dashboard-v1 persona-vectors-dashboard:v1
```

---

## Docker Commands Reference

### Container Management
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop containers
docker stop dashboard-v0 dashboard-v1

# Start containers
docker start dashboard-v0 dashboard-v1

# Remove containers
docker rm dashboard-v0 dashboard-v1

# View logs
docker logs dashboard-v0
docker logs dashboard-v1

# Follow logs in real-time
docker logs -f dashboard-v1
```

### Image Management
```bash
# List images
docker images

# Remove images
docker rmi fjordhauler/persona-vectors-dashboard:v0
docker rmi fjordhauler/persona-vectors-dashboard:v1

# Pull latest
docker pull fjordhauler/persona-vectors-dashboard:v0
docker pull fjordhauler/persona-vectors-dashboard:v1
docker pull fjordhauler/persona-vectors-dashboard:latest  # Points to v1
```

---

## EC2 Deployment

### Setup on EC2
```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Install Docker (Amazon Linux 2)
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Logout and login again for group changes
exit
ssh -i your-key.pem ec2-user@your-ec2-ip

# Pull and run dashboards
docker run -d -p 8501:8501 --restart unless-stopped --name dashboard-v0 \
  fjordhauler/persona-vectors-dashboard:v0

docker run -d -p 8502:8502 --restart unless-stopped --name dashboard-v1 \
  fjordhauler/persona-vectors-dashboard:v1
```

### Security Group Configuration

**Inbound rules required:**
- Port 8501 (TCP) - Dashboard v0
- Port 8502 (TCP) - Dashboard v1
- Port 22 (TCP) - SSH access

---

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8501
sudo lsof -i :8502

# Or use different ports
docker run -p 8503:8501 fjordhauler/persona-vectors-dashboard:v0
```

### Container Won't Start
```bash
# Check logs
docker logs dashboard-v1

# Run in foreground to see errors
docker run -p 8502:8502 fjordhauler/persona-vectors-dashboard:v1
```

### Container Won't Stop
```bash
# Force kill
docker kill dashboard-v1
docker rm dashboard-v1
```

### Out of Memory
```bash
# Check Docker stats
docker stats

# Run with memory limit
docker run -p 8502:8502 -m 2g fjordhauler/persona-vectors-dashboard:v1
```

---

## Technical Details

### Dashboard v0
- **Base Image:** python:3.10-slim
- **Port:** 8501
- **Image Size:** ~180MB
- **Command:** `streamlit run dashboard_v0.py --server.port=8501`

### Dashboard v1
- **Base Image:** python:3.10-slim
- **Port:** 8502
- **Image Size:** ~180MB
- **Command:** `streamlit run dashboard_v1.py --server.port=8502`

### Files Included
- Dashboard Python file (v0 or v1)
- `data/` directory (vectors, results)
- `figures/` directory (visualizations)
- `dashboard_requirements.txt`

---

## Production Deployment

### With Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  dashboard-v0:
    image: fjordhauler/persona-vectors-dashboard:v0
    ports:
      - "8501:8501"
    restart: unless-stopped
    container_name: dashboard-v0

  dashboard-v1:
    image: fjordhauler/persona-vectors-dashboard:v1
    ports:
      - "8502:8502"
    restart: unless-stopped
    container_name: dashboard-v1
```

Deploy:
```bash
docker-compose up -d
```

---

## Push to Docker Hub
```bash
# Login
docker login

# Tag images
docker tag persona-vectors-dashboard:v0 fjordhauler/persona-vectors-dashboard:v0
docker tag persona-vectors-dashboard:v1 fjordhauler/persona-vectors-dashboard:v1
docker tag persona-vectors-dashboard:v1 fjordhauler/persona-vectors-dashboard:latest

# Push
docker push fjordhauler/persona-vectors-dashboard:v0
docker push fjordhauler/persona-vectors-dashboard:v1
docker push fjordhauler/persona-vectors-dashboard:latest
```

---

## Monitoring
```bash
# View resource usage
docker stats dashboard-v0 dashboard-v1

# Check health
docker inspect dashboard-v1 | grep -i health

# Auto-restart on failure
docker update --restart unless-stopped dashboard-v0 dashboard-v1
```

---

## Links

- **Docker Hub:** https://hub.docker.com/r/fjordhauler/persona-vectors-dashboard
- **Live Demo v0:** http://3.106.128.216:8501
- **Live Demo v1:** http://3.106.128.216:8502
- **GitHub:** [Your repository URL]
