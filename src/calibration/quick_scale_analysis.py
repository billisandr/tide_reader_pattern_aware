#!/usr/bin/env python3
"""
Quick analysis of the uploaded scale image
"""

import cv2
import numpy as np

# Based on visual inspection of your uploaded image, I can see:
# - White/light colored scale with black markings
# - Scale appears to be vertical on the left side of the image
# - Water level appears to be around the 01-02 mark
# - Scale shows measurements from approximately 00 to 04

def analyze_uploaded_image():
    print("ANALYSIS OF YOUR UPLOADED SCALE IMAGE")
    print("="*60)
    
    print("Visual observations:")
    print("- Scale type: White background with black markings")
    print("- Scale orientation: Vertical")
    print("- Scale position: Left side of image") 
    print("- Water level: Approximately at 01-02 cm mark")
    print("- Visible range: 00 to 04+ cm")
    print("- Scale appears to be partially submerged")
    
    print("\nRecommended config.yaml values based on visual analysis:")
    print("-" * 50)
    
    # Estimating from the image proportions
    # The scale appears to take up roughly the left 1/4 of the image width
    # and most of the image height
    
    print("""
scale:
  # Adjust based on your actual scale measurements
  total_height: 5.0     # Visible range appears to be ~5cm, corrected from dm scale reading
  width: 5.0            # Typical ruler width
  expected_position:
    x_min: 50           # Scale starts from left edge
    x_max: 150          # Scale width approximately 100 pixels
    y_min: 50           # Small margin from top
    y_max: 450          # Most of image height
    
  # RGB/HSV color detection for your white scale with black markings
  color_detection:
    enabled: true
    debug_color_masks: true
    scale_colors:
      white:              # White scale background
        enabled: true
        hsv_lower: [0, 0, 180]      # Lower brightness threshold
        hsv_upper: [180, 30, 255]   # Allow some color variation
      
      black:              # Black markings and text
        enabled: true  
        hsv_lower: [0, 0, 0]
        hsv_upper: [180, 255, 80]   # Dark regions
        
      # You may also want to enable blue detection for water
      blue:
        enabled: true
        hsv_lower: [100, 100, 50]
        hsv_upper: [130, 255, 255]
""")
    
    print("\nNext steps:")
    print("1. Save your image to: C:/tide-level-img-proc/test_image.jpg")
    print("2. Run the interactive analysis tool:")
    print("   python analyze_scale_photo.py")
    print("3. Use DEBUG_MODE=true to test the detection:")
    print("   DEBUG_MODE=true PYTHONPATH=src python src/main.py")
    print("4. Adjust the values based on the debug output")
    
    print("\nTesting commands:")
    print("# Test color detection")
    print("DEBUG_MODE=true python -c \"")
    print("import cv2, yaml, numpy as np")
    print("from src.water_level_detector import WaterLevelDetector")
    print("image = cv2.imread('test_image.jpg')")
    print("with open('config.yaml', 'r') as f: config = yaml.safe_load(f)")
    print("detector = WaterLevelDetector(config, 10.0)")
    print("detector.process_image('test_image.jpg')")
    print("\"")

if __name__ == "__main__":
    analyze_uploaded_image()