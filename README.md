# Water Level Measurement System

*Automated tide/water level detection using computer vision*

> Disclaimer: This work is part of a non-funded prototype research idea conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [TUC](https://www.tuc.gr/el/archi).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Use Guide](#quick-use-guide)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Enhanced Scale Configuration Analysis](#enhanced-scale-configuration-analysis)
  - [Calibration System](#calibration-system)
  - [Processing Images](#processing-images)
  - [Visual Debugging](#visual-debugging)
  - [Water Detection Methods](#water-detection-methods)
  - [Scale Detection and Color Options](#scale-detection-and-color-options)
- [Data Export Options](#data-export-options)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Deployment Options](#deployment-options)
- [Contributing](#contributing)
- [Performance](#performance)
- [Architecture](#architecture)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

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

## Quick Use Guide

This guide provides a step-by-step workflow for getting the system running quickly:

### Step 1: Setup and Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd tide-level-img-proc-backup

# Install Python dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/input data/calibration data/output data/debug
```

### Step 2: Run Enhanced Scale Analysis

Before calibration, analyze your scale setup:

```bash
# Place your calibration image as data/calibration/calibration_image.jpg
# Run the interactive analysis tool
python src/calibration/analyze_scale_photo.py
```

**Follow the interactive prompts:**

1. Enter actual scale readings (top and waterline positions)
2. Click 4 scale corners (even if partially underwater)  
3. Mark waterline position (left and right edges)
4. Optionally sample scale colors and water color
5. Review generated config.yaml suggestions

**Apply the results:** Copy the suggested values to your `config.yaml`

### Step 3: Configure Detection Method

Edit `config.yaml` to set detection method and forced method:

```yaml
detection:
  method: 'integrated'                    # Run all methods
  forced_method: 'enhanced_gradient'      # Force best method
  
  # Environment variables for testing
  USE_GUI_SELECTOR: true                  # Enable folder selection GUI
  DEBUG_MODE: true                        # Enable debug images
  CALIBRATION_MODE: false                 # Set to true for calibration
```

### Step 4: Set Environment Variables

```bash
# Windows Command Prompt
set USE_GUI_SELECTOR=true
set DEBUG_MODE=true
set CALIBRATION_MODE=false
```

Or create a `.env` file:

```bash
USE_GUI_SELECTOR=true
DEBUG_MODE=true  
CALIBRATION_MODE=false
```

### Step 5: Run System Calibration

```bash
# Run calibration mode to generate calibration.yaml
set CALIBRATION_MODE=true & python src/main.py
```

**Calibration generates:**

- `data/calibration/calibration.yaml` with precise pixel/cm ratio
- Enhanced waterline reference data
- Scale marking analysis (if available)

### Step 6: Add Test Images

```bash
# Place test images in any folder (e.g., data/input_tests/)
# System will prompt for folder selection if USE_GUI_SELECTOR=true
```

### Step 7: Run Water Level Detection

```bash
# Run main detection system
python src/main.py
```

**When prompted:** Select your image folder via GUI dialog

### Step 8: Monitor Logs and Results

**Key log messages to watch for:**

```bash
# Successful startup
INFO - Using enhanced waterline-aware calibration (confidence: 0.98)
INFO - FORCED METHOD CONFIGURED: Will always use 'enhanced_gradient' method

# Detection results for each image
INFO - Detection summary: 4 method(s) found results
INFO -   enhanced_gradient: Y=17, confidence=0.874
INFO -   edge: Y=274, confidence=0.563  
INFO -   color: Y=0, confidence=0.267
INFO -   gradient: Y=278, confidence=1.000
INFO - FORCED METHOD: Using 'enhanced_gradient' method (Y=17, confidence=0.874)

# Final measurements
INFO - Processed IMG_0154.JPG: Water level = 445.5cm
```

### Step 9: Check Debug Images

Debug images are saved in `data/debug/debug_session_TIMESTAMP/`:

```bash
# Enhanced debug images structure:
data/debug/debug_session_20250829_102135/
├── original/                      # Original input images
├── preprocessed/                  # Resized images with metadata
├── scale_detection/              # Scale boundary detection (blue rectangles)
├── scale_bounds_enhanced/        # Enhanced scale detection with method info
├── scale_region_extracted/       # Extracted scale regions for analysis
├── edges/                        # Canny edge detection results
├── contours/                     # Hough line detection results
├── gradient_analysis/            # Vertical gradient analysis (Sobel)
├── water_detection/              # Water detection results with annotations
├── water_color_mask/            # Water color mask visualization
├── waterline_within_scale/      # Waterline detection within scale bounds
├── integrated_detection_methods/ # Multi-method comparison visualization
├── final_result/                # Final annotated results with measurements
└── [conditional folders]        # Additional folders based on detection methods used
```

**What to check in debug images:**

- Scale boundaries are correctly detected (blue rectangles)
- Waterline detection is accurate (green lines)
- Final measurements are overlaid correctly
- Multiple detection methods are compared

### Step 10: Review Output Data

Check generated measurement data:

```bash
# View output files
data/output/
├── measurements.db                         # Main SQLite database
├── measurements_20250829_102137.csv       # CSV export
├── measurements_20250829_102137.json      # JSON export  
├── measurements_20250829_102137.db        # Database backup
└── annotated/                              # Visual output images
    ├── annotated_success_20250829_102140.jpg    # Successful detection
    ├── annotated_failed_20250829_102142.jpg     # Failed detection (diagnostic)
    └── annotated_success_20250829_102145.jpg    # Another successful detection
```

#### Understanding Annotated Images

**Important**: Annotated images are created for ALL processed images, not just successful detections:

- **Success Images** (`annotated_success_*.jpg`): Show successful water level detection with:
  - Green water line overlay
  - Blue scale boundary lines  
  - Measurement text (water level, scale above water)
  - Processing time and calibration info

- **Failed Images** (`annotated_failed_*.jpg`): Show diagnostic information for failed detections:
  - Red water line (if detected) or "No water line detected"
  - Blue scale boundaries (if detected) or "Scale bounds not detected"
  - "DETECTION FAILED" status text
  - Processing diagnostics to help troubleshoot issues

**Key Point**: Even if some images fail detection, you'll still get annotated output showing what was attempted. This helps debug why certain images couldn't be processed successfully.

**Configuration**: Enable/disable annotated image creation in `config.yaml`:

```yaml
processing:
  save_processed_images: true  # Set to false to disable annotated images
```

### Step 11: Run Cleanup (Optional)

```bash
# Run the automated cleanup script
.\cleanup.bat

# Or manually clean specific directories:
# - Clean debug images (optional - useful for disk space)
# - Reset processed image tracking (to reprocess same images) 
# - Clean processed images directory
```

## Key Configuration Options

**Force specific detection method while running all methods:**

```yaml
detection:
  method: 'integrated'              # Always run all 4 methods
  forced_method: 'enhanced_gradient' # But use this method's result
  # Options: 'enhanced_gradient', 'edge', 'color', 'gradient', null
```

**Environment variables for testing:**

```bash
set DEBUG_MODE=true              # Enable debug images and detailed logging
set USE_GUI_SELECTOR=true        # Enable folder selection GUI
set CALIBRATION_MODE=true        # Run calibration mode
set PROCESS_INTERVAL=60          # Processing interval in seconds
```

**Common troubleshooting:**

- **Scale not detected:** Disable image resizing in config: `resize_width: null`
- **Waterline inaccurate:** Use `forced_method: 'enhanced_gradient'` for clear water
- **No images processed:** Check folder permissions and database path
- **Calibration fails:** Ensure `data/calibration/calibration_image.jpg` exists

This guide covers the essential workflow from setup to measurement results. For detailed configuration options, see the full documentation sections below.

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
set CALIBRATION_MODE=true & python src/main.py
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
  # Use enhanced calibration data with waterline gradient analysis
  use_enhanced_data: true
  enhanced_data_path: 'data/calibration/calibration.yaml'

detection:
  # Water line detection parameters
  edge_threshold_low: 50
  edge_threshold_high: 150
  blur_kernel_size: 7
  min_contour_area: 500
  
  # Water detection method: 'edge', 'color', 'gradient'
  method: 'edge'              # Choose detection method (see Water Detection Methods section)
  
  # Color-based detection (HSV ranges for water) - used when method: 'color'
  water_hsv_lower: [100, 50, 50]   # Lower HSV bounds for water color detection
  water_hsv_upper: [130, 255, 255] # Upper HSV bounds for water color detection
  
  # Pattern-aware detection configuration (used with pattern_processing.mode: 'pattern_aware')
  pattern_aware:
    # Hybrid waterline verification system
    waterline_verification:
      enabled: true                   # Enable hybrid waterline verification
      min_pattern_confidence: 0.7     # Minimum confidence for waterline detection
      gradient_kernel_size: 3         # Sobel kernel size for gradient analysis
      gradient_threshold: 10          # Threshold for significant gradient changes
      transition_search_height: 20    # Search height around suspicious regions (pixels)
      
      # Consolidated pattern analysis thresholds
      pattern_analysis:
        # Consecutive good pattern detection thresholds
        scale_consistency_threshold: 0.15    # Scale factor consistency for consecutive good patterns (±15%)
        size_consistency_threshold: 0.25     # Size consistency for consecutive good patterns (±25%) 
        spacing_consistency_threshold: 0.50  # Spacing consistency for consecutive good patterns (±50%)
        min_consecutive_patterns: 3          # Minimum consecutive good patterns required
        
        # Anomaly detection thresholds (applied after establishing baseline)
        scale_anomaly_threshold: 0.15        # Scale factor change to flag suspicious regions (±15%)
        size_anomaly_threshold: 0.20         # Size change to flag suspicious regions (±20%)
        aspect_ratio_anomaly_threshold: 0.20 # Aspect ratio change to flag suspicious regions (±20%)
        max_gap_ratio: 2.0                   # Max gap between patterns (2x expected spacing)

processing:
  # Image processing settings
  resize_width: null  # WARNING: Set to null to disable resizing (recommended)
                      # If enabled (e.g., 800), scale coordinates may become incorrect
                      # Use enhanced calibration workflow to avoid coordinate mismatch
  save_processed_images: true
  image_format: 'jpg'
  
output:
  # Output settings - Enable/disable export formats
  csv_export: true      # Export measurements to timestamped CSV files
  json_export: true     # Export measurements to timestamped JSON files  
  database: true        # Create timestamped database backup copies

debug:
  # Visual debugging options - can be overridden by DEBUG_MODE environment variable
  enabled: false            # Set to true to enable debug mode via config
  save_debug_images: true
  debug_output_dir: 'data/debug'
  annotation_color: [0, 255, 0]  # Green BGR
  annotation_thickness: 2
  font_scale: 0.7
  steps_to_save:
    # Core debugging steps (always available)
    - 'original'                      # Original input images
    - 'preprocessed'                  # Resized/prepared images
    - 'scale_detection'              # Scale boundary detection
    - 'scale_bounds_enhanced'        # Enhanced scale detection with method info
    - 'scale_region_extracted'       # Extracted scale regions
    - 'edges'                        # Canny edge detection
    - 'contours'                     # Hough line detection
    - 'gradient_analysis'            # Vertical gradient analysis
    - 'water_detection'              # Water detection results
    - 'integrated_detection_methods' # Multi-method comparison
    - 'final_result'                 # Final annotated results
    
    # Enhanced color-based detection steps (conditional)
    - 'color_mask_blue'              # Blue color detection masks
    - 'color_mask_black'             # Black color detection masks  
    - 'color_mask_combined'          # Combined color masks
    - 'hsv_conversion'               # HSV color space conversion
    - 'water_color_mask'            # Water color mask visualization
    - 'masked_grayscale'             # Color-filtered grayscale images
    
    # Enhanced edge detection steps (conditional)
    - 'edges_color_enhanced'         # Color-enhanced edge detection
    - 'edges_masked_gray'            # Edges on color-masked grayscale
    - 'edges_multi_channel'          # Combined edges from RGB channels
    - 'edges_hue_transitions'        # Hue channel transition edges
    - 'edges_individual_colors'      # Edges from individual color masks
    - 'edges_final_combined'         # Final combined color-enhanced edges
    
    # Analysis and validation steps (conditional)
    - 'scale_contours_analysis'      # Contour scoring for scale detection
    - 'waterline_within_scale'       # Waterline detection within scale bounds
```

## Usage

### Enhanced Scale Configuration Analysis

**Before calibration**, use the enhanced scale analysis tool to determine optimal configuration values with waterline detection:

```bash
# 1. Place your scale image as data/calibration/calibration_image.jpg
# 2. Run the enhanced scale analysis tool
python src/calibration/analyze_scale_photo.py
```

**The enhanced analysis tool provides:**

- **Enhanced waterline detection**: Mark actual waterline position on scale for precise calibration
- **Real measurement inputs**: Enter actual scale readings for accurate cm/pixel calculation
- **Interactive scale boundary picker**: Click on full visible scale corners (even if partially underwater)
- **Interactive color selection**: Click on scale background and marking colors (optional)
- **Interactive water color sampling**: Click on water color for detection parameter calibration (optional)
- **Automatic edge detection**: Finds potential scale edges using computer vision
- **Color analysis**: Analyzes selected scale and water colors for RGB/HSV detection setup
- **Configuration suggestions**: Generates optimal config.yaml values with custom color ranges

**Enhanced Analysis Workflow:**

1. **Image display**: Shows calibration image window before requesting measurements
2. **Scale measurement input**: Enter actual scale readings while viewing the image:
   - Enter scale reading at TOP of visible scale (e.g., 485 cm)
   - Enter scale reading at current WATERLINE position (e.g., 420 cm)
3. **Scale boundary selection**: Click 4 corner points of FULL visible scale (top-left, top-right, bottom-left, bottom-right)
   - Select entire visible scale even if parts are underwater
4. **Waterline marking**: Click 2 points on left and right edges of scale at waterline position
5. **Interactive color sampling** (optional): Click on scale background color, then on marking/text color
6. **Interactive water sampling** (optional): Click on water color (if visible) for detection calibration
7. **Automatic analysis**: Edge detection and color analysis
8. **Waterline gradient analysis**: Advanced analysis of color differences above/below waterline
9. **Enhanced config generation**: Provides ready-to-use config.yaml values plus accurate calibration data
10. **Summary display**: Clear summary of suggested config.yaml changes at the end

**Key Advantages of Enhanced Workflow:**

- **More Accurate Calibration**: Uses actual measured scale segment instead of estimated total height
- **Waterline Aware**: Directly captures current water level during calibration
- **Handles Partial Submersion**: Works with scales that are partially underwater
- **Precise Measurements**: Real scale readings provide exact cm/pixel ratio
- **Better Water Detection**: Waterline reference improves water level detection accuracy

### Calibration System

The system provides two integrated calibration workflows that work together to provide accurate water level measurements.

#### Workflow 1: Enhanced Interactive Analysis + Calibration (Recommended)

**Best for**: New setups, precise measurements, waterline-aware calibration, partially submerged scales

**Enhanced Complete Process**:

```bash
# Step 1: Enhanced interactive scale analysis with waterline detection
python src/calibration/analyze_scale_photo.py
# → Select 4 corners of FULL visible scale (even if partially underwater)
# → Mark waterline position on scale (2 points: left and right edges)
# → Enter actual scale readings at top and waterline positions
# → System calculates precise pixels/cm from real measurements
# → Choose scale background/marking colors (optional)
# → Sample water color (optional)
# → Generates config.yaml suggestions AND accurate calibration data

# Step 2: Apply the enhanced analysis results
# - Update your config.yaml with suggested values
# - Enhanced calibration.yaml is automatically generated with waterline data

# Step 3: Run calibration (uses enhanced waterline-aware data)
set CALIBRATION_MODE=true & python src/main.py
# → Loads enhanced calibration.yaml with waterline reference
# → Uses actual measured cm/pixel ratio
# → Ready for highly accurate water level detection
```

**Enhanced Advantages**:

- **Highest accuracy**: Uses actual scale measurements instead of estimates
- **Waterline-aware**: Directly incorporates current water level into calibration
- **Handles submersion**: Works perfectly with partially underwater scales
- **Real measurements**: No guessing - uses actual scale readings for calibration
- **Complete setup**: Handles configuration, calibration, AND waterline detection
- **Color calibration**: Includes water color detection setup
- **User control**: Full control over scale boundaries and measurement points

#### Workflow 2: Direct Configuration Calibration (Legacy)

**Best for**: Quick setup, when you already have precise config.yaml settings

**Process**:

```bash
# Step 1: Ensure config.yaml has accurate scale settings
# scale:
#   total_height: 450.0        # Actual scale height in cm
#   expected_position:         # Approximate scale location
#     x_min: 75
#     x_max: 190
#     y_min: 1
#     y_max: 605

# Step 2: Place calibration image
# - Save image as: data/calibration/calibration_image.jpg

# Step 3: Run direct calibration
set CALIBRATION_MODE=true & python src/main.py
# → Uses config.yaml values to generate calibration.yaml
# → Calculates pixels_per_cm from scale dimensions
```

#### Calibration Data Integration

**How the methods work together**:

1. **Data Generation**: Both workflows create `data/calibration/calibration.yaml`
2. **Source Tracking**: File includes metadata about which method generated it
3. **Consistent Format**: Same data structure regardless of generation method
4. **Automatic Updates**: Each calibration run updates the file with current settings

**Recent Improvements (Latest Version)**:

- **Fixed YAML serialization errors**: Resolved numpy object serialization issues in calibration.yaml
- **Improved image display timing**: Calibration image now shows before measurement input
- **Enhanced waterline gradient analysis**: Fixed KeyError issues with gradient data structure
- **Clean output messages**: Removed Unicode tick marks that caused display issues
- **Windows command syntax**: Updated all command examples to use proper Windows syntax (`set VAR=value & command`)
- **Better error handling**: Improved numpy type conversion for YAML compatibility
- **Enhanced summary display**: Clear config.yaml suggestions shown at end of calibration process

**Enhanced generated calibration.yaml contains**:

```yaml
pixels_per_cm: 12.546                  # Precise conversion factor from real measurements
image_path: data/calibration/...       # Source calibration image
calibration_date: '2024-08-26T...'     # Generation timestamp
image_dimensions:                      # Image metadata
  width: 3024
  height: 4032
  resized: false
scale_measurements:                    # Actual scale readings
  top_measurement_cm: 485.0           # Scale reading at top
  waterline_measurement_cm: 420.0     # Scale reading at waterline
  measurement_difference_cm: 65.0      # Actual measured difference
  current_water_level_cm: 420.0       # Current water level
reference_points:                      # Enhanced reference points
  top_of_scale: {x: 75, y: 45}        # Top of scale boundary
  bottom_of_scale: {x: 190, y: 580}   # Bottom of scale boundary
  waterline:                          # Waterline position data
    x_left: 78
    y_left: 245
    x_right: 185
    y_right: 247
    y_average: 246
scale_boundaries:                      # Complete boundary data
  x_min: 75
  x_max: 190
  y_min: 45
  y_max: 580
  x_min_pct: 0.025                    # Percentage coordinates
  x_max_pct: 0.063
  y_min_pct: 0.011
  y_max_pct: 0.144
calibration_method: 'enhanced_interactive_waterline'  # Enhanced method
confidence: 0.98                       # Higher confidence with real measurements
notes: 'Generated by enhanced analyze_scale_photo.py with waterline detection...'
```

#### Method Selection Guide

**Use Enhanced Interactive Analysis When**:

- Setting up system for first time
- Scale is partially underwater/submerged
- Need waterline-aware calibration
- Want maximum measurement accuracy with real scale readings
- Scale boundaries are unclear or complex
- Need water color detection setup
- Scale position varies between images
- **Recommended for all new installations**

**Use Direct Calibration When**:

- Config.yaml is already precisely configured from previous interactive analysis
- Scale position and size are completely consistent
- Quick calibration update needed (not recommended for accuracy)
- Batch processing multiple identical setups

#### Calibration Validation

After either method, verify calibration accuracy:

```bash
# Test with debug mode to see detection results
set DEBUG_MODE=true & python src/main.py
# Check generated debug images in data/debug/
# Verify scale detection boundaries and water line detection
```

### Processing Images

**Batch processing:**

```bash
# Place images in data/input/ directory
python src/main.py
```

**GUI directory selection:**

```bash
set USE_GUI_SELECTOR=true & python src/main.py
# Select directory via file dialog
```

**Visual debugging mode:**

```bash
# Option 1: Enable via environment variable (overrides config)
set DEBUG_MODE=true & python src/main.py

# Option 2: Enable via config.yaml (set debug.enabled: true)
python src/main.py
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
set DEBUG_MODE=true & python src/main.py

# Or in .env file
DEBUG_MODE=true
```

**Debug output structure:**

```
data/debug/
└── debug_session_20250822_160530/
    # Core detection pipeline folders (always present)
    ├── original/                      # Input images
    ├── preprocessed/                  # Resized images with metadata
    ├── scale_detection/              # Scale boundary detection
    ├── scale_bounds_enhanced/        # Enhanced scale detection with method info
    ├── scale_region_extracted/       # Extracted scale regions
    ├── edges/                        # Canny edge detection results
    ├── contours/                     # Hough line detection results
    ├── gradient_analysis/            # Vertical gradient analysis
    ├── water_detection/              # Water detection results
    ├── integrated_detection_methods/ # Multi-method comparison
    ├── final_result/                 # Final annotated results
    
    # Enhanced detection folders (conditional - when color detection enabled)
    ├── hsv_conversion/               # HSV color space conversion
    ├── color_mask_blue/             # Blue color detection masks
    ├── color_mask_black/            # Black color detection masks
    ├── color_mask_combined/         # Combined color masks
    ├── water_color_mask/            # Water color mask visualization
    ├── masked_grayscale/            # Color-filtered grayscale images
    ├── edges_color_enhanced/        # Color-enhanced edge detection
    ├── edges_masked_gray/           # Edges on color-masked grayscale
    ├── edges_multi_channel/         # Combined edges from RGB channels
    ├── edges_hue_transitions/       # Hue channel transition edges
    ├── edges_individual_colors/     # Edges from individual color masks
    ├── edges_final_combined/        # Final combined color-enhanced edges
    └── waterline_within_scale/      # Waterline detection within scale bounds
```

**Enhanced debug annotations include:**

**Core Detection Visualizations:**

- **Scale regions** highlighted with blue rectangles showing detection method
- **Edge detection** results with Canny parameters and kernel information  
- **Line detection** with Hough transform results and horizontal line filtering
- **Gradient analysis** with Sobel operator visualization and peak detection
- **Water detection** results with detected waterlines and confidence annotations
- **Multi-method comparison** showing results from all detection algorithms

**Advanced Color-Based Analysis:**

- **Color masks** for scale types (blue, black markings; various backgrounds)
- **HSV conversion** showing color space transformation for analysis
- **Water color masks** highlighting detected water regions
- **Enhanced edge detection** combining RGB channels and color transitions
- **Color-filtered analysis** showing masked grayscale and individual color processing

**Detection Pipeline Information:**

- **Processing parameters** including thresholds, kernel sizes, and calibration data
- **Method selection** showing which detection approach was used (integrated/forced)
- **Timing information** and performance metrics for each processing step
- **Confidence scores** for individual detection methods and final consensus
- **Scale measurement calculations** with pixel-to-cm conversions and reference points

**Configure debug options** in `config.yaml`:

```yaml
debug:
  enabled: false              # Set to true, or use DEBUG_MODE=true environment variable
  save_debug_images: true
  debug_output_dir: 'data/debug'
  steps_to_save:
    # Core steps (always available)
    - 'original'
    - 'preprocessed'
    - 'scale_detection'
    - 'edges'
    - 'contours'
    - 'gradient_analysis'
    - 'water_detection'
    - 'integrated_detection_methods'
    - 'final_result'
    
    # Enhanced steps (conditional on color detection)
    - 'hsv_conversion'
    - 'color_mask_combined'
    - 'edges_color_enhanced'
    - 'scale_contours_analysis'
```

### Water Detection Methods

The system provides three distinct water detection methods with **enhanced multi-color-space analysis** for improved accuracy, especially for clear water scenarios:

```yaml
detection:
  method: 'gradient'  # Options: 'edge', 'color', 'gradient' (recommended for clear water)
```

**Enhanced Detection Features (Latest Version):**

- **Multi-Color-Space Analysis**: Analyzes water effects across RGB, HSV, LAB, and YUV color spaces
- **Clear Water Optimization**: Specifically designed for clear water that darkens scale background
- **Edge-Gradient Fusion**: Combines Sobel, Scharr, and Laplacian edge detection with gradient analysis
- **Texture Variance Analysis**: Detects water's smoothing effect on surface texture
- **Enhanced Calibration Integration**: Uses waterline reference data from interactive calibration

**Method Selection Guide:**

- **`gradient`** **RECOMMENDED**: Best for clear water and enhanced accuracy
  - **Multi-color-space gradient analysis** across 5 color spaces (RGB, HSV, LAB, YUV, Grayscale)
  - **Clear water darkening detection** - identifies 10-40% brightness reduction patterns
  - **Edge-based validation** using horizontal gradient analysis
  - **Texture analysis** - detects water's smoothing effects
  - **Enhanced calibration integration** - uses waterline reference data
  - **High confidence scoring** based on multiple validation factors
  - **Best for**: Clear water, subtle transitions, enhanced calibration setups

- **`edge`**: Best for clear water-air interfaces with sharp contrast
  - Uses Canny edge detection with optional color enhancement
  - Reliable for typical water level detection with strong boundaries
  - Works well with various lighting conditions
  - **Best for**: Sharp water interfaces, good lighting

- **`color`**: Best for colored water (algae, sediment, specific water conditions)
  - Uses HSV color ranges to identify water regions: `water_hsv_lower/upper`
  - Ideal when water has distinct color different from background
  - Configure HSV ranges using the analyze_scale_photo.py tool
  - **Best for**: Distinctly colored water, non-clear water conditions

**Enhanced Gradient Method Details:**

The enhanced gradient method uses a comprehensive **5-layer analysis system**:

1. **Multi-Color-Space Consistency (35% weight)**:
   - Grayscale brightness analysis
   - HSV Value channel analysis  
   - LAB Lightness analysis (perceptually accurate)
   - YUV Luminance analysis
   - Validates darkening patterns across all color spaces

2. **BGR Channel Analysis (20% weight)**:
   - Individual Blue, Green, Red channel analysis
   - Compares with calibrated expected values
   - Detects subtle color shifts per channel

3. **Edge Gradient Validation (20% weight)**:
   - Prioritizes horizontal gradients (waterline characteristic)
   - Uses Sobel, Scharr, and Laplacian edge detection
   - Validates transitions across multiple scales

4. **Spatial Width Consistency (15% weight)**:
   - Ensures uniform pattern across waterline width
   - Samples multiple points along waterline
   - Validates expected darkening consistency

5. **Texture Variance Analysis (10% weight)**:
   - Detects water's surface smoothing effects
   - Analyzes texture changes above/below waterline
   - Validates expected variance reduction patterns

**Enhanced Calibration Integration:**

When using enhanced calibration data (`use_enhanced_data: true`), the gradient method gains access to:

- **Waterline reference position** for validation
- **Expected darkening ratios** specific to your setup
- **Above/below water color characteristics**
- **Calibrated brightness differences** from real measurements
- **95% base confidence** vs 80% for standard methods

**Color-Based Water Detection Setup:**
When using `method: 'color'`, configure the HSV ranges for your water:

```yaml
detection:
  method: 'color'
  water_hsv_lower: [100, 50, 50]   # Adjust based on your water color
  water_hsv_upper: [130, 255, 255] # Use analyze_scale_photo.py to calibrate
```

Use the interactive analysis tool to determine optimal HSV ranges:

```bash
python src/calibration/analyze_scale_photo.py
# Step 4: Click on water color for HSV range calibration
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

### Pattern Continuity Analysis Configuration

**⚠️ NEW FEATURE**: Advanced pattern continuity analysis for E-pattern sequential detection with configurable thresholds.

The pattern-aware detection system includes sophisticated **pattern continuity analysis** that ensures gradient analysis only runs after establishing a baseline of consecutive good patterns. This prevents false waterline detection from early pattern inconsistencies.

#### How Pattern Continuity Analysis Works

The system follows this logical sequence:

1. **Consecutive Good Pattern Detection**: Analyzes E-patterns from top to bottom to find patterns with consistent scale factors, sizes, and spacing
2. **Baseline Establishment**: Uses only the consecutive good patterns to calculate reliable baseline metrics  
3. **Anomaly Detection**: Only analyzes patterns that come **after** the established good sequence
4. **Gradient Analysis**: Performs water interface gradient analysis **only** in suspicious regions after good patterns

#### Consolidated Pattern Analysis Configuration

```yaml
detection:
  pattern_aware:
    waterline_verification:
      pattern_analysis:
        # Consecutive good pattern detection thresholds
        scale_consistency_threshold: 0.15    # Scale factor consistency (±15%)
        size_consistency_threshold: 0.25     # Template size consistency (±25%) 
        spacing_consistency_threshold: 0.50  # Pattern spacing consistency (±50%)
        min_consecutive_patterns: 3          # Minimum consecutive patterns required
        
        # Anomaly detection thresholds (applied after establishing baseline)
        scale_anomaly_threshold: 0.15        # Scale factor change to flag suspicious regions (±15%)
        size_anomaly_threshold: 0.20         # Size change to flag suspicious regions (±20%)
        aspect_ratio_anomaly_threshold: 0.20 # Aspect ratio change to flag suspicious regions (±20%)
        max_gap_ratio: 2.0                   # Max gap between patterns (2x expected spacing)
```

#### Parameter Details

**🎯 `scale_consistency_threshold` (Default: 0.15)**
- **Purpose**: Controls scale factor consistency for consecutive good patterns
- **Range**: 0.05 (very strict) to 0.30 (very lenient)
- **Effect**: Lower values = stricter consecutive pattern requirements
- **Example**: 0.15 means patterns with scale factors within ±15% of the first pattern are considered "good"
- **Use Case**: 
  - **Decrease to 0.10**: For very uniform scales with consistent pattern sizes
  - **Increase to 0.20**: For scales with natural size variations or lighting changes

**📏 `size_consistency_threshold` (Default: 0.25)**  
- **Purpose**: Controls template size consistency for consecutive good patterns
- **Range**: 0.10 (very strict) to 0.40 (very lenient)
- **Effect**: Lower values = stricter size consistency requirements
- **Example**: 0.25 means patterns with sizes within ±25% of the first pattern are considered "good"
- **Use Case**:
  - **Decrease to 0.15**: For scales with very uniform pattern sizes
  - **Increase to 0.35**: For scales with perspective distortion or variable pattern sizes

**📐 `spacing_consistency_threshold` (Default: 0.50)**
- **Purpose**: Controls pattern spacing consistency for consecutive good patterns  
- **Range**: 0.20 (very strict) to 0.80 (very lenient)
- **Effect**: Lower values = stricter spacing consistency requirements
- **Example**: 0.50 means pattern spacing within ±50% of average spacing is considered "good"
- **Use Case**:
  - **Decrease to 0.30**: For scales with very uniform pattern spacing
  - **Increase to 0.70**: For scales with variable spacing or perspective effects

**🔢 `min_consecutive_patterns` (Default: 3)**
- **Purpose**: Minimum number of consecutive good patterns required before analyzing anomalies
- **Range**: 2 to 6 patterns
- **Effect**: Higher values = more reliable baseline but requires more visible patterns
- **Example**: 3 means the system needs at least 3 consecutive good patterns to establish baseline
- **Use Case**:
  - **Decrease to 2**: For images with fewer visible patterns or partially submerged scales
  - **Increase to 4-5**: For very high accuracy requirements with many visible patterns

#### Configuration Examples

**🔧 High Precision Setup** (Strict thresholds for laboratory conditions):
```yaml
pattern_analysis:
  # Consecutive pattern detection (strict)
  scale_consistency_threshold: 0.10    # Very strict scale consistency
  size_consistency_threshold: 0.15     # Very strict size consistency  
  spacing_consistency_threshold: 0.30  # Very strict spacing consistency
  min_consecutive_patterns: 4          # Require 4 good patterns
  
  # Anomaly detection (strict)
  scale_anomaly_threshold: 0.10        # Very strict scale anomaly detection
  size_anomaly_threshold: 0.15         # Very strict size anomaly detection
  aspect_ratio_anomaly_threshold: 0.15 # Very strict aspect anomaly detection
  max_gap_ratio: 1.5                   # Stricter gap detection
```

**🌊 Field Conditions Setup** (Lenient thresholds for outdoor/variable conditions):
```yaml
pattern_analysis:
  # Consecutive pattern detection (lenient)
  scale_consistency_threshold: 0.20    # More lenient scale consistency
  size_consistency_threshold: 0.30     # More lenient size consistency
  spacing_consistency_threshold: 0.60  # More lenient spacing consistency  
  min_consecutive_patterns: 2          # Require only 2 good patterns
  
  # Anomaly detection (lenient)
  scale_anomaly_threshold: 0.20        # More lenient scale anomaly detection
  size_anomaly_threshold: 0.25         # More lenient size anomaly detection
  aspect_ratio_anomaly_threshold: 0.25 # More lenient aspect anomaly detection
  max_gap_ratio: 3.0                   # More lenient gap detection
```

**🔬 Research Setup** (Balanced thresholds for scientific applications):
```yaml
pattern_analysis:
  # Consecutive pattern detection (balanced)
  scale_consistency_threshold: 0.12    # Moderately strict scale consistency
  size_consistency_threshold: 0.20     # Moderately strict size consistency
  spacing_consistency_threshold: 0.40  # Moderately strict spacing consistency
  min_consecutive_patterns: 3          # Standard requirement
  
  # Anomaly detection (balanced)
  scale_anomaly_threshold: 0.15        # Balanced scale anomaly detection
  size_anomaly_threshold: 0.20         # Balanced size anomaly detection
  aspect_ratio_anomaly_threshold: 0.20 # Balanced aspect anomaly detection
  max_gap_ratio: 2.0                   # Standard gap detection
```

#### Troubleshooting Pattern Continuity

**Problem**: "Only X consecutive good patterns found - insufficient for reliable waterline analysis"

**Solutions**:
- **Increase threshold values** to be more lenient with pattern variations
- **Decrease `min_consecutive_patterns`** if fewer patterns are visible
- **Check E-pattern templates** to ensure they match your scale markings
- **Verify scale detection** is properly identifying the scale region

**Problem**: Gradient analysis running too early (before actual water interface)

**Solutions**:
- **Decrease threshold values** to be stricter with pattern consistency
- **Increase `min_consecutive_patterns`** to require more baseline patterns
- **Check debug images** to see where consecutive patterns break

**Problem**: System too strict, not detecting any waterline

**Solutions**:
- **Increase all threshold values** by 0.05-0.10 increments
- **Decrease `min_consecutive_patterns`** to 2
- **Enable debug mode** to see pattern continuity analysis results

#### Debug Information

When debug mode is enabled, the pattern continuity analysis provides:

- **Console logging** showing where pattern sequences break
- **Debug panel information** showing consecutive pattern count and baseline position  
- **Visual annotations** distinguishing good patterns from anomalous patterns
- **Threshold comparison details** for each pattern analyzed

This advanced pattern continuity system ensures that waterline detection is physically consistent and only occurs after establishing reliable pattern baselines.

## Data Export Options

The system automatically exports measurement data in multiple formats based on your `config.yaml` settings. All exports are created in the `data/output/` directory with timestamped filenames to prevent overwrites.

### Export Configuration

```yaml
output:
  csv_export: true      # Export to CSV format
  json_export: true     # Export to JSON format  
  database: true        # Create database backups
```

### Export Formats

**CSV Export (`csv_export: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.csv` files
- Includes all measurement data in tabular format
- **New**: `detection_method` column shows which method was actually used (e.g., 'e_pattern_sequential', 'enhanced_gradient', 'standard', etc.)
- Compatible with Excel, data analysis tools
- Best for: Spreadsheet analysis, reporting, graphs, method performance tracking

**JSON Export (`json_export: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.json` files
- Structured data with proper type handling
- Easy parsing for web applications, APIs
- Best for: Integration with other systems, custom analysis

**Database Backup (`database: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.db` backup files
- Complete SQLite database copies
- Preserves all data including processing metadata
- Best for: Data archiving, system recovery

### Export Behavior

- **Automatic**: Exports occur after each batch of images is processed
- **Timestamped**: Files include date/time to prevent overwrites
- **Conditional**: Only enabled formats create files
- **Error Handling**: Export failures don't stop image processing
- **Directory Creation**: Output directories are created automatically

**Example Output:**

```
data/output/
├── measurements.db                    # Main database (always present)
├── measurements_20250826_143022.csv  # CSV export (if enabled)
├── measurements_20250826_143022.json # JSON export (if enabled)  
└── measurements_20250826_143022.db   # Database backup (if enabled)
```

**Disable Exports:**
Set any format to `false` to disable:

```yaml
output:
  csv_export: false     # No CSV files created
  json_export: false    # No JSON files created
  database: false       # No database backups
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
set USE_GUI_SELECTOR=true & python src/main.py

# 4. Test calibration
set CALIBRATION_MODE=true & python src/main.py

# 5. Test color detection with debug
set DEBUG_MODE=true & python src/main.py
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

- Ensure calibration file exists in `data/calibration/calibration.yaml`
- Check calibration file path in logs
- Run enhanced calibration: `python src/calibration/analyze_scale_photo.py` then `set CALIBRATION_MODE=true & python src/main.py`

**Scale detection completely missing/wrong:**

- **Most likely cause**: Image resizing mismatch with hardcoded coordinates
- **Immediate solution**: Disable resizing in config.yaml: `resize_width: null`
- **Root cause**: When resizing is enabled, hardcoded scale coordinates become incorrect
- **Important**: Current system uses absolute pixel coordinates that don't scale properly
- **Best practice**: Always disable resizing (`resize_width: null`) until relative coordinate system is implemented
- **Workaround**: Run `python src/calibration/analyze_scale_photo.py` to get correct coordinates for your setup

**Waterline detection inaccurate:**

- **For clear water**: Use `method: 'gradient'` in config.yaml (recommended)
- Use enhanced calibration workflow: `python src/calibration/analyze_scale_photo.py`
- Mark waterline position during calibration for reference
- Enter actual scale measurements for precise cm/pixel calculation
- Ensure scale is properly outlined even if partially underwater
- Enable enhanced calibration: `use_enhanced_data: true` in config.yaml
- **Clear water optimization**: The enhanced gradient method specifically handles water that darkens scale background

**Missing annotated images (fewer than expected):**

- **Symptom**: All images show in debug session, but only some have annotated images in `data/output/annotated/`
- **Root cause**: Previous versions only created annotated images for 100% successful detections
- **What this means**:
  - If scale detection OR water line detection failed → no annotated image created
  - Even partial success (e.g., found scale but no water line) → no annotated image
  - Debug images still created for all processed images regardless of success/failure
- **Solution**: Updated system now creates annotated images for ALL processed images:
  - `annotated_success_*.jpg` for successful detections (green lines, measurements)
  - `annotated_failed_*.jpg` for failed detections (red lines, diagnostic info)
- **Configuration**: Ensure `processing.save_processed_images: true` in config.yaml
- **Benefit**: You can now see what went wrong with failed detections instead of getting no visual feedback

**No images being processed:**

- Verify image directory permissions
- Check database path configuration
- Clear database if paths have changed: `rm data/output/measurements.db`

**Enhanced calibration tool not starting:**

- Ensure display is available (not headless environment)
- Check OpenCV GUI support: `python -c "import cv2; cv2.imshow('test', cv2.imread('image.jpg')); cv2.waitKey(1000)"`
- For WSL: Enable X11 forwarding or use Windows Python
- Run from project root directory

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
set DEBUG_MODE=true & python src/main.py
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

This work has used ideas from several open-source projects:

- [**Object-Size-measurement-in-an-image-using-OpenCV4.0-and-imutils**](https://github.com/prateekralhan/Object-Size-measurement-in-an-image-using-openCV4.0-and-imutils) - MIT License
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
