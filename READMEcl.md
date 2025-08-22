# Water Level Measurement System
*Automated tide/water level detection using computer vision*

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)

## Overview

This system provides automated water/tide level detection and measurement using computer vision techniques. Designed for fixed camera setups with calibrated measurement scales, it processes images continuously and stores precise measurements in a database.

**Key Applications:**
- Tide monitoring stations
- River water level monitoring
- Laboratory water measurement
- Environmental research projects

## Features

- **Automated Processing**: Continuous monitoring of input directories
- **Precise Calibration**: One-time setup for fixed camera-scale configurations  
- **Multiple Detection Methods**: Edge detection, color-based analysis, and gradient methods
- **Database Storage**: SQLite database with comprehensive measurement history
- **Docker Ready**: Fully containerized deployment with HTTPS server
- **GUI Interface**: Optional tkinter interface for directory selection
- **Export Options**: CSV, JSON, and database export capabilities
- **Web Interface**: HTTPS server for remote access and image upload
- **Reporting**: Automated measurement reports and visualizations

## Quick Start

### Using Docker (Recommended)

1. **Clone and setup:**
```bash
git clone <your-repo-url>
cd tide-level-img-proc
```

2. **Generate SSL certificates:**
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Deploy:**
```bash
docker compose up --build -d
```

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run calibration:**
```bash
# Place calibration image in data/calibration/
CALIBRATION_MODE=true python src/main.py
```

3. **Start processing:**
```bash
python src/main.py
```

## Installation

### Prerequisites

- **Python 3.9+** with pip
- **Docker & Docker Compose** (for containerized deployment)
- **OpenSSL** (for HTTPS certificates)

### System Dependencies

The system requires several Python packages for image processing:

```bash
# Core dependencies
pip install opencv-python numpy scipy
pip install PyYAML Pillow matplotlib pandas
pip install sqlalchemy python-dotenv
pip install Flask Flask-Login Flask-WTF
```

Or simply:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file with the following settings:

```bash
# Processing Configuration
USE_GUI_SELECTOR=false          # Enable GUI directory selection
CALIBRATION_MODE=false          # Run in calibration mode
PROCESS_INTERVAL=60            # Processing interval (seconds)
DB_PATH=./data/measurements.db # Database file path

# Web Server (Docker deployment)
SECRET_KEY=your-secret-key-here
ADMIN_USER=admin
ADMIN_PASS=secure-password
```

### System Configuration (`config.yaml`)

```yaml
# Scale configuration
scale:
  total_height: 100.0    # Known height in cm
  width: 10.0           # Scale width in cm

# Detection parameters  
detection:
  method: 'edge'        # 'edge', 'color', 'gradient'
  edge_threshold_low: 50
  edge_threshold_high: 150

# Processing settings
processing:
  resize_width: 800
  save_processed_images: true
```

## Usage

### Calibration Process

**Automatic calibration** (recommended for known scale heights):
```bash
# 1. Place calibration image in data/calibration/
# 2. Set scale height in config.yaml
# 3. Run calibration
CALIBRATION_MODE=true python src/main.py
```

**Interactive calibration** (for manual setup):
```bash
# 1. Place calibration image in data/calibration/  
# 2. Run interactive mode
CALIBRATION_MODE=true python src/main.py
# 3. Click top and bottom of scale when prompted
# 4. Enter actual height in centimeters
```

### Processing Images

**Batch processing:**
```bash
# Place images in data/input/ directory
python src/main.py
```

**GUI directory selection:**
```bash
USE_GUI_SELECTOR=true python src/main.py
# Select directory via file dialog
```

**Visual debugging mode:**
```bash
DEBUG_MODE=true python src/main.py
# Generates step-by-step annotated debug images
```

**Docker deployment:**
```bash
docker compose up -d
# Access web interface at https://localhost:5000
```

### Visual Debugging

The system includes a comprehensive visual debugging mode that creates annotated images showing each step of the processing pipeline.

**Enable debug mode:**
```bash
# Via environment variable
DEBUG_MODE=true python src/main.py

# Or in .env file
DEBUG_MODE=true
```

**Debug output structure:**
```
data/debug/
└── debug_session_20250822_160530/
    ├── original/          # Input images
    ├── preprocessed/      # Resized images with metadata
    ├── scale_detection/   # Scale region highlighting
    ├── edges/            # Edge detection results
    ├── contours/         # Detected lines and features
    ├── final_result/     # Final measurements with annotations
    └── summary images    # Processing summary
```

**Debug annotations include:**
- **Scale regions** highlighted with blue rectangles
- **Detected lines** color-coded (yellow=horizontal water lines, red=other)
- **Water level measurements** overlaid in green
- **Processing parameters** and timing information
- **Confidence scores** and calibration data

**Configure debug options** in `config.yaml`:
```yaml
debug:
  save_debug_images: true
  debug_output_dir: 'data/debug'
  steps_to_save:
    - 'original'
    - 'preprocessed' 
    - 'edges'
    - 'final_result'
```

## Project Structure

```
tide-level-img-proc/
├── src/                        # Source code
│   ├── main.py                 # Main application entry point
│   ├── water_level_detector.py # Core processing logic
│   ├── calibration.py          # Calibration management
│   ├── database.py            # Database operations
│   └── utils.py               # Utility functions
├── data/                      # Data directories
│   ├── input/                 # Input images
│   ├── processed/            # Processed images  
│   ├── calibration/          # Calibration data
│   └── output/               # Export files
├── raspi_files/              # Raspberry Pi integration
├── logs/                     # Application logs
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Deployment configuration
├── .env.example           # Environment template
└── README.md             # Documentation
```

## Development

### Setting Up Development Environment

1. **Clone repository:**
```bash
git clone <repo-url>
cd tide-level-img-proc
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup development environment:**
```bash
# Create local configuration
cp .env.example .env

# Create data directories
mkdir -p data/{input,processed,calibration,output} logs
```

### Running Tests

```bash
# Run with debug logging
python src/main.py

# Test GUI interface
USE_GUI_SELECTOR=true python src/main.py

# Test calibration
CALIBRATION_MODE=true python src/main.py
```

### Code Style

The project follows standard Python conventions:
- PEP 8 style guidelines
- Comprehensive logging throughout
- Type hints where applicable
- Docstrings for all functions

## Troubleshooting

### Common Issues

**"System not calibrated" error:**
- Ensure calibration file exists in `data/calibration/`
- Check calibration file path in logs
- Run calibration mode first

**No images being processed:**
- Verify image directory permissions
- Check database path configuration
- Clear database if paths have changed: `rm data/measurements.db`

**GUI directory selector not appearing:**
- Ensure `USE_GUI_SELECTOR=true` in environment
- Check that `.env` file is being loaded
- Verify display environment (especially in Docker)

**Docker container won't start:**
- Check SSL certificate files exist
- Verify environment variables are set
- Check port 5000 is available

### Debug Mode

Enable detailed logging:
```bash
# Modify src/main.py temporarily
setup_logging(logging.DEBUG)
```

Enable visual debugging:
```bash
DEBUG_MODE=true python src/main.py
```

View logs:
```bash
tail -f logs/water_level.log
```

### Visual Debug Analysis

When troubleshooting detection issues, use visual debugging to:
- **Verify scale detection**: Check if blue rectangles properly highlight the measurement scale
- **Examine edge detection**: Review edge images to ensure proper feature extraction
- **Validate line detection**: Confirm yellow lines indicate detected water surfaces
- **Check final measurements**: Review annotated results with overlaid measurements

## Deployment Options

### Production Docker Deployment

1. **Setup SSL certificates:**
```bash
# Generate self-signed certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Or use existing certificates
cp your-cert.pem cert.pem
cp your-key.pem key.pem
```

2. **Configure environment:**
```bash
# Production .env file
SECRET_KEY=production-secret-key
ADMIN_USER=your-username  
ADMIN_PASS=secure-password
DB_PATH=/app/data/measurements.db
```

3. **Deploy with Docker Compose:**
```bash
docker compose -f docker-compose.yml up -d
```

### Raspberry Pi Integration

The system includes integration files for Raspberry Pi deployment:

```bash
# Copy files to Raspberry Pi
scp raspi_files/* pi@raspberrypi:/tmp/

# Install on Raspberry Pi  
sudo cp /tmp/gps_to_rtc_sync.sh /usr/local/bin/
sudo cp /tmp/photo_logger.py /usr/local/bin/
sudo cp /tmp/*.service /etc/systemd/system/
sudo cp /tmp/*.timer /etc/systemd/system/

# Enable services
sudo systemctl enable gps-to-rtc.timer
sudo systemctl start gps-to-rtc.timer
```

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** following the existing code style
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit changes:** `git commit -m 'Add amazing feature'`
7. **Push to branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Follow existing code patterns and style
- Add comprehensive logging for new features
- Update configuration documentation
- Test on both local and Docker environments
- Validate measurement accuracy with known test cases

## Performance

**Typical Performance Metrics:**
- **Processing Speed**: ~2-3 seconds per image
- **Accuracy**: ±0.5cm with proper calibration
- **Memory Usage**: ~200MB baseline + image processing overhead
- **Storage**: ~1MB per 1000 measurements

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Image Input    │───▶│  Water Level     │───▶│  Database       │
│  (Files/Upload) │    │  Detection       │    │  Storage        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Calibration     │
                       │  Management      │ 
                       └──────────────────┘
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

This work incorporates ideas and code from several open-source projects:

- [**Object-Size-Measurement-Using-OpenCV**](https://github.com/khan-tahir/Object-Size-Measurement-Using-OpenCV) - MIT License
- [**object-size-detector-python**](https://github.com/intel-iot-devkit/object-size-detector-python) - BSD 3-Clause License  
- [**Object-Detection-Size-Measurement**](https://github.com/Ali619/Object-Detection-Size-Measurement) - Unspecified license

**Research Context:**
This work is part of prototype research conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/en/).

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Additional documentation in `/docs` directory
- **Research**: Contact SenseLAB for research collaboration

---

*For more detailed technical documentation, see the complete system documentation in the `/docs` directory.*