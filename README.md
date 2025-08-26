# Water Level Measurement System

*Automated tide/water level detection using computer vision*

> Disclaimer: This work is part of a non-funded prototype research idea conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [TUC](https://www.tuc.gr/el/archi).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)

## Overview

This system provides automated water/tide level detection and measurement using computer vision techniques. Designed for fixed camera setups with calibrated measurement scales, it processes images continuously and stores precise measurements in a database.

**🔴 Key Applications:**

- Tide monitoring stations
- River water level monitoring
- Laboratory water measurement
- Environmental research projects

## Features

- **Automated Processing**: Continuous monitoring of input directories
- **Precise Calibration**: One-time setup for fixed camera-scale configurations  
- **Advanced Scale Detection**: RGB/HSV color filtering for various scale types and markings
- **Multiple Detection Methods**: Edge detection, color-based analysis, and gradient methods
- **🔴 Database Storage**: SQLite database with comprehensive measurement history
- **Docker Ready**: Fully containerized deployment with HTTPS server
- **GUI Interface**: Optional tkinter interface for directory selection
- **🔴 Export Options**: CSV, JSON, and database export capabilities
- **🔴 Web Interface**: HTTPS server for remote access and image upload
- **Reporting**: Automated measurement reports and visualizations

## Quick Start

### Using Docker (Recommended)

1. **Clone and setup:**

```bash
git clone <your-repo-url>
cd tide-level-img-proc
```

2. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Deploy:**

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
set CALIBRATION_MODE=true && python src/main.py
```

3. **Start processing:**

```bash
python src/main.py
```

## Installation

### Prerequisites

- **Python 3.9+** with pip
- **Docker & Docker Compose** (for containerized deployment)

### System Dependencies

The system requires several Python packages for image processing:

```bash
# Core dependencies
pip install opencv-python numpy scipy
pip install PyYAML Pillow matplotlib pandas
pip install sqlalchemy python-dotenv
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
DB_PATH=./data/output/measurements.db # Database file path

# Web Server (Docker deployment)
SECRET_KEY=your-secret-key-here
ADMIN_USER=admin
ADMIN_PASS=secure-password
```

### System Configuration (`config.yaml`)

```yaml
# Camera and Scale Configuration
scale:
  # Known height of scale in cm
  total_height: 450.0
  # Width of scale in cm (for reference)
  width: 8.0
  # Scale position in image (if fixed)
  expected_position:
    x_min: 81
    x_max: 181
    y_min: 1
    y_max: 603
  
  # RGB/HSV color-based scale detection
  color_detection:
    enabled: true
    adaptive_thresholding: true
    debug_color_masks: true
    
    # Scale color definitions (HSV ranges)
    scale_colors:
      yellow:
        enabled: true
        hsv_lower: [20, 100, 100]
        hsv_upper: [30, 255, 255]
        description: "Yellow scale background"
      
      white:
        enabled: true
        hsv_lower: [1, 0, 0]
        hsv_upper: [164, 108, 255]
        description: "White scale background"
      
      blue:
        enabled: true
        hsv_lower: [94, 12, 155]
        hsv_upper: [114, 112, 255]
        description: "Blue scale background"

      black:
        enabled: true
        hsv_lower: [97, 0, 0]
        hsv_upper: [129, 92, 87]
        description: "Black scale markings"
      
      red:
        enabled: true
        hsv_lower: [0, 100, 100]
        hsv_upper: [10, 255, 255]
        description: "Red scale background"
      
      red_alt:
        enabled: true
        hsv_lower: [170, 100, 100]
        hsv_upper: [180, 255, 255]
        description: "Red scale background (alt range)"
      
      white_markings:
        enabled: true
        hsv_lower: [0, 0, 220]
        hsv_upper: [180, 20, 255]
        description: "White scale markings on colored backgrounds"
    
    # Morphological operations for mask cleanup
    morphology:
      kernel_size: 3
      close_iterations: 1
      open_iterations: 1

calibration:
  # Pixels per centimeter (will be calculated during calibration)
  pixels_per_cm: null
  # Reference water level for calibration (cm)
  reference_water_level: 50.0

detection:
  # Water line detection parameters
  edge_threshold_low: 50
  edge_threshold_high: 150
  blur_kernel_size: 7
  min_contour_area: 500
  
  # Water detection method: 'edge', 'color', 'gradient'
  method: 'edge'
  
  # Color-based detection (HSV ranges for water)
  water_hsv_lower: [100, 50, 50]
  water_hsv_upper: [130, 255, 255]

processing:
  # Image processing settings
  resize_width: 800  # Resize images for faster processing
  save_processed_images: true
  image_format: 'jpg'
  
output:
  # Output settings
  csv_export: true
  json_export: true
  database: true

debug:
  # Visual debugging options
  enabled: false
  save_debug_images: true
  debug_output_dir: 'data/debug'
  annotation_color: [0, 255, 0]  # Green BGR
  annotation_thickness: 2
  font_scale: 0.7
  steps_to_save:
    - 'original'
    - 'preprocessed' 
    - 'edges'
    - 'contours'
    - 'scale_detection'
    - 'water_detection'
    - 'final_result'
```

## Usage

### Scale Configuration Analysis

**Before calibration**, use the scale analysis tool to determine optimal configuration values:

```bash
# 1. Place your scale image as data/calibration/calibration_image.jpg
# 2. Run the scale analysis tool
python src/calibration/analyze_scale_photo.py
```

**The analysis tool provides:**

- **Interactive coordinate picker**: Click on scale corners to define boundaries
- **Interactive color selection**: Click on scale background and marking colors
- **Automatic edge detection**: Finds potential scale edges using computer vision
- **Color analysis**: Analyzes selected scale colors for RGB/HSV detection setup
- **Configuration suggestions**: Generates optimal config.yaml values with custom color ranges

**Analysis workflow:**

1. **Basic image analysis**: Shows dimensions and file info
2. **Interactive boundary selection**: Click 4 corner points (top-left, top-right, bottom-left, bottom-right of scale)
3. **Interactive color sampling**: Click on scale background color, then on marking/text color
4. **Edge detection**: Automatically finds vertical/horizontal lines
5. **Color analysis**: Processes selected colors and generates precise HSV ranges
6. **Config generation**: Provides ready-to-use config.yaml values with your specific colors

### Calibration Process

The `CalibrationManager` provides two calibration methods. **Important**: The `analyze_scale_photo.py` script is a separate analysis tool and is NOT automatically called during calibration.

#### Method 1: Automatic Calibration (Known Scale Height)

**When used**: If `config['scale']['total_height']` is set in config.yaml  
**How it works**:

- Uses `detect_scale_height_pixels()` to find the scale using edge detection  
- Automatically detects the tallest vertical object (assumed to be the scale)
- Calculates `pixels_per_cm = detected_pixels / known_height_cm`

**Setup steps**:

```bash
# 1. Place calibration image in data/calibration/ as calibration_image.jpg
# 2. (Optional) Run scale analysis first: python src/calibration/analyze_scale_photo.py
# 3. Set scale height in config.yaml (e.g., total_height: 450.0)
# 4. Run calibration
set CALIBRATION_MODE=true && python src/main.py
```

#### Method 2: Interactive Calibration (Manual Setup)

**When used**: If `config['scale']['total_height']` is NOT set or is null  
**How it works**:

- Opens OpenCV window showing the calibration image
- User clicks on two points: top and bottom of the scale
- System prompts: `"Enter the actual height in cm:"`
- User types the real-world distance between clicked points
- Calculates `pixels_per_cm = pixel_distance / entered_height_cm`

**Setup steps**:

```bash
# 1. Place calibration image in data/calibration/ as calibration_image.jpg
# 2. (Optional) Run scale analysis: python src/calibration/analyze_scale_photo.py
# 3. Ensure config.yaml has total_height: null or not set
# 4. Run interactive mode
set CALIBRATION_MODE=true && python src/main.py
# 5. Click top and bottom of scale when prompted
# 6. Enter actual height in centimeters
```

**Important Notes**:

- Interactive calibration won't work in headless Docker containers (no display/keyboard)
- For Docker deployments, use Method 1 (known height) or pre-generated calibration files
- The `analyze_scale_photo.py` script is a helpful **analysis tool** but is separate from the calibration process

### Processing Images

**Batch processing:**

```bash
# Place images in data/input/ directory
python src/main.py
```

**GUI directory selection:**

```bash
set USE_GUI_SELECTOR=true && python src/main.py
# Select directory via file dialog
```

**Visual debugging mode:**

```bash
set DEBUG_MODE=true && python src/main.py
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
set DEBUG_MODE=true && python src/main.py

# Or in .env file
DEBUG_MODE=true
```

**Debug output structure:**

```
data/debug/
└── debug_session_20250822_160530/
    ├── original/                # Input images
    ├── preprocessed/            # Resized images with metadata
    ├── hsv_conversion/          # HSV color space conversion
    ├── color_mask_yellow/       # Individual color masks
    ├── color_mask_white/
    ├── color_mask_red/
    ├── color_mask_blue/
    ├── color_mask_combined/     # Combined color masks
    ├── edges_color_enhanced/    # Color-enhanced edge detection
    ├── scale_contours_analysis/ # Scale detection analysis
    ├── scale_detection/         # Scale region highlighting
    ├── edges/                   # Standard edge detection results
    ├── contours/                # Detected lines and features
    ├── final_result/            # Final measurements with annotations
    └── summary images           # Processing summary
```

**Debug annotations include:**

- **Color masks** for different scale types (yellow, white, red backgrounds; blue, black, white markings)
- **Scale regions** highlighted with blue rectangles
- **Contour analysis** showing scoring for scale detection candidates
- **Enhanced edge detection** combining multiple color-based methods
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
    - 'hsv_conversion'
    - 'color_mask_combined'
    - 'edges_color_enhanced'
    - 'scale_contours_analysis' 
    - 'edges'
    - 'final_result'
```

### Scale Detection and Color Options

The system uses advanced RGB/HSV color filtering to detect various types of measurement scales:

**Supported Scale Types:**

- **Yellow scales with blue markings** (common tide gauges)
- **White scales with black markings** (laboratory equipment)
- **Red scales with white markings** (industrial applications)
- **Custom color combinations** via configuration

**Color Detection Features:**

- **Multi-channel edge detection** combining RGB and HSV analysis
- **Adaptive thresholding** for varying lighting conditions  
- **Morphological operations** for noise reduction and gap filling
- **Contour scoring** based on size, aspect ratio, location, and color overlap
- **Fallback methods** when color detection fails

**Color Configuration:**
Each color is defined by HSV ranges for robust detection across lighting conditions:

```yaml
scale_colors:
  yellow:             # HSV: [20-30, 100-255, 100-255]
  white:              # HSV: [0-180, 0-30, 200-255]  
  red:                # HSV: [0-10, 100-255, 100-255]
  red_alt:            # HSV: [170-180, 100-255, 100-255] (red wraparound)
  blue:               # HSV: [100-120, 100-255, 50-255]
  black:              # HSV: [0-180, 0-255, 0-50]
  white_markings:     # HSV: [0-180, 0-20, 220-255] (high brightness, low saturation)
```

The color detection can be **enabled/disabled per color** and includes **extensive debug visualization** showing mask generation, edge enhancement, and contour analysis.

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
# 1. Scale configuration analysis
python src/calibration/analyze_scale_photo.py

# 2. Run with debug logging
python src/main.py

# 3. Test GUI interface
set USE_GUI_SELECTOR=true && python src/main.py

# 4. Test calibration
set CALIBRATION_MODE=true && python src/main.py

# 5. Test color detection with debug
set DEBUG_MODE=true && python src/main.py
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
- Clear database if paths have changed: `rm data/output/measurements.db`

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
set DEBUG_MODE=true && python src/main.py
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

1. **🔴 Setup SSL certificates:**

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
DB_PATH=/app/data/output/measurements.db
```

3. **Deploy with Docker Compose:**

```bash
docker compose -f docker-compose.yml up -d
```

**For GUI applications in Docker (X11 forwarding):**

```bash
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix your-image
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

## 🔴 Acknowledgments

This work incorporates ideas and code from several open-source projects:

- [**Object-Size-measurement-in-an-image-using-OpenCV4.0-and-imutils**](https://github.com/prateekralhan/Object-Size-measurement-in-an-image-using-openCV4.0-and-imutils) - MIT License
- [**Object-Size-Measurement-Using-OpenCV**](https://github.com/khan-tahir/Object-Size-Measurement-Using-OpenCV) - MIT License
- [**object-size-detector-python**](https://github.com/intel-iot-devkit/object-size-detector-python) - BSD 3-Clause License  
- [**Object-Detection-Size-Measurement**](https://github.com/Ali619/Object-Detection-Size-Measurement) - Unspecified license

**Research Context:**
This work is part of prototype research conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/en/).

## 🔴 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Additional documentation in `/docs` directory
- **Research**: Contact SenseLAB for research collaboration

---

*For more detailed technical documentation, see the complete system documentation in the `/docs` directory.*
