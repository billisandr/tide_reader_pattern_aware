# Docker Deployment Guide

> **Important Notice**: The Docker functionality in this directory has not been thoroughly tested and may require adjustments for production use. This containerized deployment configuration is maintained to facilitate potential future needs for dockerized deployment scenarios. For reliable operation, local installation following the main README is recommended.

This directory contains all Docker-related files for deploying the Water Level Measurement System in containerized environments.

## Quick Start

### Production Deployment

```bash
# From project root directory
cd docker
docker compose up -d
```

### Development with GUI Support

```bash
# From project root directory
cd docker
docker compose -f docker-compose.dev.yml up
```

## Directory Structure

```
docker/
├── README.md                  # This file
├── Dockerfile                 # Multi-stage container definition
├── docker-compose.yml         # Production deployment
├── docker-compose.dev.yml     # Development with GUI support
├── .dockerignore             # Build context exclusions
└── .env.docker.example       # Environment variables template
```

## Files Overview

### `Dockerfile`

- **Multi-stage build** for optimized final image size
- **Python 3.12** slim-bookworm base image
- **Non-root user** (`appuser`) for security
- **Health checks** for container monitoring
- **OpenCV dependencies** pre-installed
- **Build stage** with development tools
- **Runtime stage** with minimal dependencies

### `docker-compose.yml` (Production)

- **Resource limits** (1GB RAM, 1 CPU)
- **Health checks** and restart policies
- **Logging configuration** (10MB max, 5 files)
- **Networks** for service isolation
- **Volume mounts** for data persistence
- **Environment variables** for configuration

### `docker-compose.dev.yml` (Development)

- **Extends** production configuration
- **Live development** with project volume mounting
- **X11 forwarding** for GUI applications
- **Host networking** for easier GUI access
- **Debug mode** enabled by default
- **Faster processing** intervals

### `.dockerignore`

Excludes unnecessary files from build context:

- Git files and IDE configurations
- Log files and debug output
- Documentation and scripts
- Build artifacts and cache

### `.env.docker.example`

Template for environment configuration with all available variables.

## Prerequisites

### System Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB available RAM
- 1GB available disk space

### For GUI Applications (Development)

**Linux:**

```bash
# Enable X11 forwarding
xhost +local:docker
export DISPLAY=:0
```

**Windows with WSL2:**

```bash
# Install X11 server (VcXsrv, Xming)
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
```

**macOS:**

```bash
# Install XQuartz
export DISPLAY=host.docker.internal:0
xhost + 127.0.0.1
```

## Configuration

### 1. Environment Setup

```bash
# Copy and customize environment file
cp .env.docker.example .env

# Edit configuration
nano .env
```

### 2. Key Environment Variables

```bash
# Image source (where your photos are located)
IMAGE_SOURCE_DIR=../data/input

# Processing settings
PROCESS_INTERVAL=60          # Processing interval in seconds
CALIBRATION_MODE=false       # Enable calibration mode
DEBUG_MODE=false            # Enable debug output

# GUI settings (development only)
USE_GUI_SELECTOR=false      # Enable GUI folder selection
DISPLAY=:0                  # X11 display for GUI apps
```

### 3. Volume Configuration

The system uses the following volumes:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../data/input` | `/app/data/input` | Input images (read-only) |
| `../data/output` | `/app/data/output` | Results and database |
| `../data/calibration` | `/app/data/calibration` | Calibration data |
| `../data/debug` | `/app/data/debug` | Debug images |
| `../logs` | `/app/logs` | Application logs |
| `../config.yaml` | `/app/config.yaml` | Configuration file |

## Deployment Scenarios

### 1. Production Server Deployment

**Setup:**

```bash
cd docker

# Configure environment
cp .env.docker.example .env
nano .env  # Set production values

# Deploy
docker compose up -d

# Monitor
docker compose logs -f
```

**Configuration for production:**

```bash
# .env file
IMAGE_SOURCE_DIR=/path/to/your/images
PROCESS_INTERVAL=300
DEBUG_MODE=false
USE_GUI_SELECTOR=false
```

### 2. Development Environment

**Setup:**

```bash
cd docker

# Configure for development
cp .env.docker.example .env
nano .env  # Enable debug features

# Enable X11 (Linux)
xhost +local:docker

# Start development container
docker compose -f docker-compose.dev.yml up
```

**Configuration for development:**

```bash
# .env file
DEBUG_MODE=true
USE_GUI_SELECTOR=true
PROCESS_INTERVAL=30
CALIBRATION_MODE=true
DISPLAY=:0
```

### 3. Calibration Setup

**For initial system calibration:**

```bash
# Place calibration image
mkdir -p ../data/calibration
cp your_calibration_image.jpg ../data/calibration/calibration_image.jpg

# Run calibration
docker compose -f docker-compose.dev.yml run --rm water-level-processor \
  python src/calibration/analyze_scale_photo.py
```

### 4. Batch Processing

**Process a directory of images:**

```bash
# Set image directory
export IMAGE_SOURCE_DIR=/path/to/images

# Run processing
docker compose up -d

# Monitor progress
docker compose logs -f water-level-processor
```

## Troubleshooting

### Common Issues

**Container won't start:**

```bash
# Check logs
docker compose logs water-level-processor

# Check system resources
docker stats

# Verify file permissions
ls -la ../data/
```

**GUI applications not working:**

```bash
# Linux: Check X11 permissions
xauth list
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# Test X11 forwarding
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw python:3.12-slim python -c "import tkinter; tkinter.Tk()"
```

**Volume mount issues:**

```bash
# Check paths exist
ls -la ../data/
ls -la ../config.yaml

# Verify permissions
sudo chown -R $(id -u):$(id -g) ../data/
```

**Memory/CPU issues:**

```bash
# Check container resources
docker stats water-level-measurement

# Adjust limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2.0'
```

### Performance Optimization

**For better performance:**

1. **Increase resource limits** in `docker-compose.yml`
2. **Use SSD storage** for volume mounts
3. **Disable debug mode** in production
4. **Optimize processing intervals**

**Monitor resource usage:**

```bash
# Container stats
docker stats

# System resources
htop

# Disk usage
df -h
```

## Development Workflow

### 1. Code Changes

```bash
# Development container auto-mounts project
# Changes are reflected immediately
docker compose -f docker-compose.dev.yml up
```

### 2. Testing

```bash
# Run tests in container
docker compose -f docker-compose.dev.yml run --rm water-level-processor \
  python -m pytest

# Interactive shell
docker compose -f docker-compose.dev.yml run --rm water-level-processor bash
```

### 3. Debugging

```bash
# Enable debug mode
echo "DEBUG_MODE=true" >> .env

# Check debug output
ls -la ../data/debug/

# View logs
docker compose logs -f water-level-processor
```

## Advanced Usage

### Custom Build Arguments

```bash
# Build with custom Python version
docker compose build --build-arg PYTHON_VERSION=3.11

# Build development image
docker compose -f docker-compose.dev.yml build
```

### Network Configuration

```bash
# Custom network
docker network create water-level-network

# Use external network in docker-compose.yml
networks:
  default:
    external: true
    name: water-level-network
```

### Health Check Customization

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import cv2; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

## Security Considerations

### Container Security

- **Non-root user** (`appuser`) used for application
- **Read-only** volume mounts where possible
- **Resource limits** prevent resource exhaustion
- **Health checks** for monitoring

### Network Security

- **Bridge network** isolates containers
- **No exposed ports** (unless web interface needed)
- **Host network** only in development

### Data Security

- **Volume permissions** properly configured
- **Sensitive data** kept in environment variables
- **Logs** properly rotated and limited

## Maintenance

### Regular Tasks

```bash
# Update containers
docker compose pull
docker compose up -d

# Clean up old images
docker image prune -f

# Monitor disk usage
docker system df

# Backup data volumes
docker run --rm -v $(pwd)/../data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/data-$(date +%Y%m%d).tar.gz /data
```

### Log Management

```bash
# View recent logs
docker compose logs --tail=50 -f

# Clear logs
docker compose down
docker system prune -f
docker compose up -d
```

## Support

For issues specific to Docker deployment:

1. Check this README
2. Verify your environment configuration
3. Check container logs: `docker compose logs`
4. Test with development configuration first
5. Refer to main project README for application-specific issues

---

**Note:** This Docker setup is designed for both development and production use. Choose the appropriate configuration files based on your needs.
