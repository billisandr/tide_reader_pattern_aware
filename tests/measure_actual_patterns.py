"""
Tool to help measure actual E-pattern sizes in input images
This will help determine if template sizing is correct
"""

import cv2
import yaml
import numpy as np
from pathlib import Path

def measure_pattern_sizes(image_path):
    """
    Interactive tool to measure actual E-pattern sizes in input images.
    Click on top and bottom of E-patterns to measure their height.
    """
    
    print(f"=== Pattern Size Measurement Tool ===")
    print(f"Image: {image_path}")
    
    # Load calibration for reference
    calibration_path = Path('data/calibration/calibration.yaml')
    if calibration_path.exists():
        with open(calibration_path, 'r') as f:
            calibration_data = yaml.safe_load(f)
        pixels_per_cm = calibration_data.get('pixels_per_cm', 1.796)
        print(f"Calibration: {pixels_per_cm:.3f} pixels per cm")
        expected_height = pixels_per_cm * 5.0
        print(f"Expected E-pattern height: {expected_height:.1f} pixels (5cm)")
    else:
        pixels_per_cm = 1.796
        expected_height = 9.0
        print("No calibration found, using default values")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    print()
    print("Instructions:")
    print("1. Click on the TOP of an E-pattern")
    print("2. Click on the BOTTOM of the same E-pattern") 
    print("3. Press 'q' to quit, 'r' to reset measurements")
    print("4. Press 's' to save measurements")
    print()
    
    # Create display image
    display_image = image.copy()
    
    # Measurement state
    measurements = []
    current_measurement = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_measurement, measurements, display_image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_measurement.append((x, y))
            
            if len(current_measurement) == 1:
                # First click - mark top
                cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(display_image, "TOP", (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                print(f"Top marked at: ({x}, {y})")
                
            elif len(current_measurement) == 2:
                # Second click - mark bottom and calculate
                cv2.circle(display_image, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(display_image, "BOTTOM", (x + 5, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Calculate height
                top_y = current_measurement[0][1]
                bottom_y = y
                height_pixels = abs(bottom_y - top_y)
                
                # Draw line connecting points
                cv2.line(display_image, current_measurement[0], (x, y), (0, 0, 255), 2)
                
                # Calculate real-world size
                height_cm = height_pixels / pixels_per_cm
                
                # Add measurement text
                mid_x = (current_measurement[0][0] + x) // 2
                mid_y = (top_y + bottom_y) // 2
                cv2.putText(display_image, f"{height_pixels}px", (mid_x + 10, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(display_image, f"{height_cm:.1f}cm", (mid_x + 10, mid_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                measurements.append({
                    'top': current_measurement[0],
                    'bottom': (x, y),
                    'height_pixels': height_pixels,
                    'height_cm': height_cm
                })
                
                print(f"Bottom marked at: ({x}, {y})")
                print(f"Pattern height: {height_pixels} pixels = {height_cm:.1f} cm")
                print(f"Expected height: {expected_height:.1f} pixels")
                print(f"Size ratio: {height_pixels/expected_height:.2f}x expected")
                print()
                
                current_measurement = []
    
    cv2.namedWindow('Pattern Measurement', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Pattern Measurement', mouse_callback)
    
    while True:
        cv2.imshow('Pattern Measurement', display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset measurements
            display_image = image.copy()
            measurements = []
            current_measurement = []
            print("Measurements reset")
        elif key == ord('s') and measurements:
            # Save measurements
            save_measurements(measurements, image_path, pixels_per_cm, expected_height)
    
    cv2.destroyAllWindows()
    
    # Print summary
    if measurements:
        print("\n=== Measurement Summary ===")
        avg_height = sum(m['height_pixels'] for m in measurements) / len(measurements)
        avg_cm = sum(m['height_cm'] for m in measurements) / len(measurements)
        
        print(f"Number of measurements: {len(measurements)}")
        print(f"Average height: {avg_height:.1f} pixels = {avg_cm:.1f} cm")
        print(f"Expected height: {expected_height:.1f} pixels = 5.0 cm")
        print(f"Size discrepancy: {avg_height/expected_height:.2f}x")
        
        if avg_height / expected_height > 1.5:
            print("Templates are likely too small! Consider:")
            print(f"- Scaling templates up by {avg_height/expected_height:.2f}x")
            print("- Check if calibration is correct")
        elif avg_height / expected_height < 0.7:
            print("Templates may be too large! Consider:")
            print(f"- Scaling templates down by {expected_height/avg_height:.2f}x")

def save_measurements(measurements, image_path, pixels_per_cm, expected_height):
    """Save measurements to file."""
    output_dir = Path('data/debug/pattern_measurements')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    output_file = output_dir / f"{image_name}_measurements.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Pattern Size Measurements\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Calibration: {pixels_per_cm:.3f} pixels per cm\n")
        f.write(f"Expected E-pattern height: {expected_height:.1f} pixels\n\n")
        
        for i, m in enumerate(measurements):
            f.write(f"Measurement {i+1}:\n")
            f.write(f"  Top: {m['top']}\n")
            f.write(f"  Bottom: {m['bottom']}\n")
            f.write(f"  Height: {m['height_pixels']} pixels = {m['height_cm']:.1f} cm\n")
            f.write(f"  Ratio to expected: {m['height_pixels']/expected_height:.2f}x\n\n")
        
        avg_height = sum(m['height_pixels'] for m in measurements) / len(measurements)
        f.write(f"Average height: {avg_height:.1f} pixels\n")
        f.write(f"Scaling factor needed: {avg_height/expected_height:.3f}\n")
    
    print(f"Measurements saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python measure_actual_patterns.py <image_path>")
        print("Example: python measure_actual_patterns.py data/images/scale_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    measure_pattern_sizes(image_path)