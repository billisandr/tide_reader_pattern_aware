#!/usr/bin/env python3
"""
Interactive scale analysis tool for determining optimal config.yaml values
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path

def analyze_image_basic(image_path):
    """Basic image analysis"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    height, width, channels = image.shape
    print("="*60)
    print("BASIC IMAGE ANALYSIS")
    print("="*60)
    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Channels: {channels}")
    print(f"File size: {os.path.getsize(image_path)} bytes")
    
    return image

def interactive_coordinate_picker(image, image_path):
    """Enhanced interactive tool for scale analysis with waterline detection"""
    print("\n" + "="*60)
    print("ENHANCED SCALE & WATERLINE ANALYSIS")
    print("="*60)
    print("STEP 1: Outline the FULL visible scale (4 corner points):")
    print("Click in order - even if parts are underwater!")
    print("1. Top-left corner of visible scale")
    print("2. Top-right corner of visible scale") 
    print("3. Bottom-left corner of visible scale")
    print("4. Bottom-right corner of visible scale")
    print("\nSTEP 2: Mark the waterline on the scale:")
    print("5. Click on LEFT edge of scale at waterline")
    print("6. Click on RIGHT edge of scale at waterline")
    print("\nSTEP 3: After waterline, click on color samples (optional):")
    print("7. Scale background color (press 's' to skip)")
    print("8. Scale markings color") 
    print("9. Water color (press 'w' to skip)")
    print("\nControls: 'r' to reset, 'q' to quit, 's' to skip colors, 'w' to skip water")
    
    # Create a copy for drawing
    display_image = image.copy()
    scale_corners = []  # 4 corner points of scale
    waterline_points = []  # 2 points marking waterline on scale
    color_samples = []
    water_samples = []
    
    # Labels and colors for visualization
    corner_labels = ["Top-Left Scale", "Top-Right Scale", "Bottom-Left Scale", "Bottom-Right Scale"]
    waterline_labels = ["Waterline Left", "Waterline Right"]
    color_labels = ["Scale Background", "Scale Markings", "Water Color"]
    
    corner_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]  # Green, Cyan, Red, Magenta
    waterline_colors = [(255, 255, 0), (0, 128, 255)]  # Yellow, Orange
    color_colors = [(128, 255, 128), (255, 128, 128), (128, 128, 255)]
    
    # State tracking
    picking_corners = True
    picking_waterline = False
    picking_colors = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal scale_corners, waterline_points, color_samples, water_samples, display_image
        nonlocal picking_corners, picking_waterline, picking_colors
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if picking_corners and len(scale_corners) < 4:
                # Handle scale corner selection
                scale_corners.append((x, y))
                color = corner_colors[len(scale_corners)-1]
                cv2.circle(display_image, (x, y), 8, color, -1)
                label = corner_labels[len(scale_corners)-1]
                cv2.putText(display_image, f"{len(scale_corners)}.{label}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if len(scale_corners) == 4:
                    # Draw scale outline
                    pts = np.array(scale_corners, np.int32)
                    cv2.polylines(display_image, [pts], True, (0, 255, 0), 2)
                    picking_corners = False
                    picking_waterline = True
                    print(f"\n✓ Scale corners marked. Now click on waterline (left and right edges of scale)")
                    
            elif picking_waterline and len(waterline_points) < 2:
                # Handle waterline point selection
                waterline_points.append((x, y))
                color = waterline_colors[len(waterline_points)-1]
                cv2.circle(display_image, (x, y), 8, color, -1)
                label = waterline_labels[len(waterline_points)-1]
                cv2.putText(display_image, f"{len(waterline_points)+4}.{label}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if len(waterline_points) == 2:
                    # Draw waterline
                    cv2.line(display_image, waterline_points[0], waterline_points[1], (255, 255, 0), 3)
                    picking_waterline = False
                    picking_colors = True
                    print(f"\n✓ Waterline marked. Now click on scale colors (or press 's' to skip)")
                    
            elif picking_colors and len(color_samples) < 2:
                # Handle color sample selection
                sample_idx = len(color_samples)
                color_samples.append((x, y))
                color = color_colors[sample_idx]
                
                # Sample the color at clicked location
                clicked_bgr = image[y, x]
                clicked_hsv = cv2.cvtColor(np.array([[clicked_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # Draw sample point
                cv2.circle(display_image, (x, y), 8, color, -1)
                cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display_image, f"{sample_idx+7}.{color_labels[sample_idx]}", 
                           (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Show color info
                sample_type = "Background" if len(color_samples) == 1 else "Markings"
                print(f"\n{sample_type} color sampled at ({x}, {y}):")
                print(f"  BGR: ({clicked_bgr[0]}, {clicked_bgr[1]}, {clicked_bgr[2]})")
                print(f"  HSV: ({clicked_hsv[0]}, {clicked_hsv[1]}, {clicked_hsv[2]})")
                
                # Create color swatch
                swatch_y = 20 + len(color_samples) * 40
                cv2.rectangle(display_image, (20, swatch_y), (60, swatch_y+30), 
                             tuple(map(int, clicked_bgr)), -1)
                cv2.putText(display_image, sample_type, (70, swatch_y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Scale Analysis', display_image)
                
                if len(color_samples) == 2:
                    analyze_color_samples(image, points, color_samples)
                    picking_colors = False
                    picking_water = True
                    print("\n" + "="*60)
                    print("STEP 3: WATER COLOR SELECTION")
                    print("="*60)
                    print("Now click on the WATER COLOR:")
                    print("- Click on a representative water color in the image")
                    print("- This helps calibrate water detection parameters")
                    print("- Press 'w' to skip water color selection if no water visible")
                    print("- Press 'q' to finish and continue with analysis")
            
            elif picking_water and len(water_samples) < 1:
                # Handle water color sample selection
                sample_idx = 6  # Water color index
                water_samples.append((x, y))
                color = colors[sample_idx]
                
                # Sample the color at clicked location
                clicked_bgr = image[y, x]
                clicked_hsv = cv2.cvtColor(np.array([[clicked_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # Draw sample point
                cv2.circle(display_image, (x, y), 8, color, -1)
                cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display_image, f"{sample_idx+1}: {point_labels[sample_idx]}", 
                           (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show color info
                print(f"\nWater color sampled at ({x}, {y}):")
                print(f"  BGR: ({clicked_bgr[0]}, {clicked_bgr[1]}, {clicked_bgr[2]})")
                print(f"  HSV: ({clicked_hsv[0]}, {clicked_hsv[1]}, {clicked_hsv[2]})")
                
                # Create color swatch for water
                swatch_y = 20 + 3 * 40  # Below scale color swatches
                cv2.rectangle(display_image, (20, swatch_y), (60, swatch_y+30), 
                             tuple(map(int, clicked_bgr)), -1)
                cv2.putText(display_image, "Water", (70, swatch_y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Scale Analysis', display_image)
                
                # Analyze water color
                analyze_water_color_sample(image, water_samples[0])
                picking_water = False
                print("\n" + "="*60)
                print("ALL SELECTIONS COMPLETE - Press 'q' to continue with analysis")
                print("="*60)
    
    cv2.namedWindow('Scale Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Scale Analysis', 800, 600)
    cv2.setMouseCallback('Scale Analysis', mouse_callback)
    cv2.imshow('Scale Analysis', display_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset everything
            points = []
            color_samples = []
            water_samples = []
            display_image = image.copy()
            picking_corners = True
            picking_colors = False
            picking_water = False
            cv2.imshow('Scale Analysis', display_image)
            print("Reset points, colors, and water samples")
        elif key == ord('s') and picking_colors:
            # Skip color selection
            print("Skipping color selection")
            picking_colors = False
            break
        elif key == ord('w') and picking_water:
            # Skip water color selection
            print("Skipping water color selection")
            picking_water = False
            print("\n" + "="*60)
            print("WATER COLOR SKIPPED - Press 'q' to continue with analysis")
            print("="*60)
    
    cv2.destroyAllWindows()
    
    # Return the enhanced data structure
    result_data = {
        'scale_corners': scale_corners,
        'waterline_points': waterline_points, 
        'color_samples': color_samples,
        'water_samples': water_samples
    }
    
    return result_data

def analyze_picked_points(points, image_shape):
    """Analyze the manually picked points"""
    if len(points) != 4:
        return
    
    height, width = image_shape[:2]
    
    # Calculate bounding box
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    scale_width = x_max - x_min
    scale_height = y_max - y_min
    
    print(f"\nSCALE BOUNDARY ANALYSIS:")
    print(f"X range: {x_min} to {x_max} (width: {scale_width} pixels)")
    print(f"Y range: {y_min} to {y_max} (height: {scale_height} pixels)")
    print(f"Scale width as % of image: {(scale_width/width)*100:.1f}%")
    print(f"Scale height as % of image: {(scale_height/height)*100:.1f}%")
    print(f"Aspect ratio (h/w): {scale_height/scale_width:.2f}")

def analyze_color_samples(image, corner_points, color_samples):
    """Analyze the interactively selected color samples"""
    if len(color_samples) != 2:
        return
    
    print("\n" + "="*60)
    print("COLOR SAMPLE ANALYSIS")
    print("="*60)
    
    # Convert image to HSV for analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    sample_data = []
    sample_names = ["Background", "Markings"]
    
    for i, (x, y) in enumerate(color_samples):
        # Sample area around clicked point (5x5 region)
        sample_region_bgr = image[max(0, y-2):min(image.shape[0], y+3), 
                                 max(0, x-2):min(image.shape[1], x+3)]
        sample_region_hsv = hsv_image[max(0, y-2):min(image.shape[0], y+3), 
                                     max(0, x-2):min(image.shape[1], x+3)]
        
        # Calculate statistics
        mean_bgr = np.mean(sample_region_bgr.reshape(-1, 3), axis=0)
        mean_hsv = np.mean(sample_region_hsv.reshape(-1, 3), axis=0)
        std_hsv = np.std(sample_region_hsv.reshape(-1, 3), axis=0)
        
        sample_data.append({
            'name': sample_names[i],
            'position': (x, y),
            'mean_bgr': mean_bgr,
            'mean_hsv': mean_hsv,
            'std_hsv': std_hsv
        })
        
        print(f"{sample_names[i]} color analysis:")
        print(f"  Position: ({x}, {y})")
        print(f"  Average BGR: ({mean_bgr[0]:.0f}, {mean_bgr[1]:.0f}, {mean_bgr[2]:.0f})")
        print(f"  Average HSV: ({mean_hsv[0]:.0f}, {mean_hsv[1]:.0f}, {mean_hsv[2]:.0f})")
        print(f"  HSV Std Dev: ({std_hsv[0]:.1f}, {std_hsv[1]:.1f}, {std_hsv[2]:.1f})")
        
        # Generate HSV range suggestions with tolerance
        h_tolerance = max(10, 2 * std_hsv[0])
        s_tolerance = max(50, 2 * std_hsv[1])  
        v_tolerance = max(50, 2 * std_hsv[2])
        
        h_lower = max(0, mean_hsv[0] - h_tolerance)
        h_upper = min(179, mean_hsv[0] + h_tolerance)
        s_lower = max(0, mean_hsv[1] - s_tolerance)
        s_upper = min(255, mean_hsv[1] + s_tolerance)
        v_lower = max(0, mean_hsv[2] - v_tolerance)
        v_upper = min(255, mean_hsv[2] + v_tolerance)
        
        print(f"  Suggested HSV range: [{h_lower:.0f}, {s_lower:.0f}, {v_lower:.0f}] to [{h_upper:.0f}, {s_upper:.0f}, {v_upper:.0f}]")
        print()
        
        sample_data[i]['hsv_range'] = {
            'lower': [int(h_lower), int(s_lower), int(v_lower)],
            'upper': [int(h_upper), int(s_upper), int(v_upper)]
        }
    
    return sample_data

def analyze_water_color_sample(image, water_sample):
    """Analyze the interactively selected water color sample"""
    if not water_sample:
        return None
    
    print("\n" + "="*60)
    print("WATER COLOR SAMPLE ANALYSIS")
    print("="*60)
    
    # Convert image to HSV for analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    x, y = water_sample
    
    # Sample area around clicked point (7x7 region for water)
    sample_region_bgr = image[max(0, y-3):min(image.shape[0], y+4), 
                             max(0, x-3):min(image.shape[1], x+4)]
    sample_region_hsv = hsv_image[max(0, y-3):min(image.shape[0], y+4), 
                                 max(0, x-3):min(image.shape[1], x+4)]
    
    # Calculate statistics
    mean_bgr = np.mean(sample_region_bgr.reshape(-1, 3), axis=0)
    mean_hsv = np.mean(sample_region_hsv.reshape(-1, 3), axis=0)
    std_hsv = np.std(sample_region_hsv.reshape(-1, 3), axis=0)
    
    print(f"Water color analysis:")
    print(f"  Position: ({x}, {y})")
    print(f"  Average BGR: ({mean_bgr[0]:.0f}, {mean_bgr[1]:.0f}, {mean_bgr[2]:.0f})")
    print(f"  Average HSV: ({mean_hsv[0]:.0f}, {mean_hsv[1]:.0f}, {mean_hsv[2]:.0f})")
    print(f"  HSV Std Dev: ({std_hsv[0]:.1f}, {std_hsv[1]:.1f}, {std_hsv[2]:.1f})")
    
    # Generate HSV range suggestions with tolerance for water
    h_tolerance = max(15, 2.5 * std_hsv[0])  # Larger tolerance for water hue
    s_tolerance = max(60, 2.5 * std_hsv[1])  # Larger tolerance for water saturation
    v_tolerance = max(60, 2.5 * std_hsv[2])  # Larger tolerance for water value
    
    h_lower = max(0, mean_hsv[0] - h_tolerance)
    h_upper = min(179, mean_hsv[0] + h_tolerance)
    s_lower = max(0, mean_hsv[1] - s_tolerance)
    s_upper = min(255, mean_hsv[1] + s_tolerance)
    v_lower = max(0, mean_hsv[2] - v_tolerance)
    v_upper = min(255, mean_hsv[2] + v_tolerance)
    
    print(f"  Suggested HSV range: [{h_lower:.0f}, {s_lower:.0f}, {v_lower:.0f}] to [{h_upper:.0f}, {s_upper:.0f}, {v_upper:.0f}]")
    print()
    
    water_data = {
        'position': (x, y),
        'mean_bgr': mean_bgr,
        'mean_hsv': mean_hsv,
        'std_hsv': std_hsv,
        'hsv_range': {
            'lower': [int(h_lower), int(s_lower), int(v_lower)],
            'upper': [int(h_upper), int(s_upper), int(v_upper)]
        }
    }
    
    return water_data

def automatic_edge_detection(image):
    """Automatic scale detection using edge detection"""
    print("\n" + "="*60)
    print("AUTOMATIC EDGE DETECTION")
    print("="*60)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Apply different edge detection methods
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        vertical_lines = []
        horizontal_lines = []
        
        # Classify lines as vertical or horizontal
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
            
            if angle > 85 and angle < 95:  # Nearly vertical
                vertical_lines.append(line[0])
            elif angle < 5 or angle > 175:  # Nearly horizontal
                horizontal_lines.append(line[0])
        
        print(f"Found {len(vertical_lines)} vertical lines")
        print(f"Found {len(horizontal_lines)} horizontal lines")
        
        # Analyze vertical lines for scale boundaries
        if vertical_lines:
            x_positions = []
            for x1, y1, x2, y2 in vertical_lines:
                x_positions.extend([x1, x2])
            
            x_positions.sort()
            potential_left = x_positions[:len(x_positions)//3]  # Left third
            potential_right = x_positions[2*len(x_positions)//3:]  # Right third
            
            if potential_left and potential_right:
                suggested_x_min = int(np.median(potential_left))
                suggested_x_max = int(np.median(potential_right))
                
                print(f"Suggested X range: {suggested_x_min} to {suggested_x_max}")
        
        # Analyze horizontal lines for scale top/bottom
        if horizontal_lines:
            y_positions = []
            for x1, y1, x2, y2 in horizontal_lines:
                y_positions.extend([y1, y2])
            
            y_positions.sort()
            if len(y_positions) >= 2:
                suggested_y_min = int(np.median(y_positions[:len(y_positions)//3]))
                suggested_y_max = int(np.median(y_positions[2*len(y_positions)//3:]))
                
                print(f"Suggested Y range: {suggested_y_min} to {suggested_y_max}")
        
        # Create visualization
        line_image = image.copy()
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        cv2.imshow('Detected Lines (Green=Vertical, Blue=Horizontal)', line_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def color_analysis(image):
    """Analyze colors in the image for RGB detection setup"""
    print("\n" + "="*60)
    print("COLOR ANALYSIS")
    print("="*60)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Sample some areas to understand color ranges
    height, width = image.shape[:2]
    
    # Sample center area (likely scale region)
    center_y = height // 2
    center_x = width // 2
    sample_size = 50
    
    sample_bgr = image[center_y-sample_size:center_y+sample_size, 
                      center_x-sample_size:center_x+sample_size]
    sample_hsv = hsv[center_y-sample_size:center_y+sample_size, 
                    center_x-sample_size:center_x+sample_size]
    
    # Calculate color statistics
    mean_bgr = np.mean(sample_bgr, axis=(0,1))
    mean_hsv = np.mean(sample_hsv, axis=(0,1))
    std_hsv = np.std(sample_hsv, axis=(0,1))
    
    print(f"Sample region (center {sample_size*2}x{sample_size*2} pixels):")
    print(f"Average BGR: ({mean_bgr[0]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[2]:.1f})")
    print(f"Average HSV: ({mean_hsv[0]:.1f}, {mean_hsv[1]:.1f}, {mean_hsv[2]:.1f})")
    print(f"HSV Std Dev: ({std_hsv[0]:.1f}, {std_hsv[1]:.1f}, {std_hsv[2]:.1f})")
    
    # Suggest HSV ranges based on the sample
    h_range = (max(0, mean_hsv[0] - 2*std_hsv[0]), min(180, mean_hsv[0] + 2*std_hsv[0]))
    s_range = (max(0, mean_hsv[1] - 2*std_hsv[1]), min(255, mean_hsv[1] + 2*std_hsv[1]))
    v_range = (max(0, mean_hsv[2] - 2*std_hsv[2]), min(255, mean_hsv[2] + 2*std_hsv[2]))
    
    print(f"Suggested HSV ranges for dominant color:")
    print(f"  H: [{h_range[0]:.0f}, {h_range[1]:.0f}]")
    print(f"  S: [{s_range[0]:.0f}, {s_range[1]:.0f}]") 
    print(f"  V: [{v_range[0]:.0f}, {v_range[1]:.0f}]")
    
    # Show color visualization
    color_display = np.zeros((200, 400, 3), dtype=np.uint8)
    color_display[:100, :200] = mean_bgr.astype(np.uint8)
    cv2.putText(color_display, "Dominant Color", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(color_display, f"HSV: {mean_hsv.astype(int)}", (10, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Color Analysis', color_display)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_config_suggestions(image, manual_points=None, color_sample_data=None, water_sample_data=None):
    """Generate config.yaml suggestions based on analysis"""
    print("\n" + "="*60)
    print("CONFIG.YAML SUGGESTIONS")
    print("="*60)
    
    height, width = image.shape[:2]
    
    # Use manual points if provided, otherwise estimate
    if manual_points and len(manual_points) == 4:
        x_coords = [p[0] for p in manual_points]
        y_coords = [p[1] for p in manual_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
    else:
        # Conservative estimates for a typical scale
        scale_width_percent = 0.15  # 15% of image width
        scale_height_percent = 0.8   # 80% of image height
        
        scale_width = int(width * scale_width_percent)
        scale_height = int(height * scale_height_percent)
        
        # Assume scale is in right portion of image
        x_min = int(width * 0.6)  # 60% across
        x_max = x_min + scale_width
        y_min = int(height * 0.1)  # 10% from top
        y_max = y_min + scale_height
    
    print("Suggested config.yaml values:")
    print("-" * 30)
    print("scale:")
    print("  # Adjust total_height based on your actual scale")
    print("  total_height: 100.0  # cm")
    print("  width: 8.0           # cm (estimate)")
    print("  expected_position:")
    print(f"    x_min: {x_min}")
    print(f"    x_max: {x_max}")
    print(f"    y_min: {y_min}")
    print(f"    y_max: {y_max}")
    print()
    # Add color detection configuration if color samples were provided
    if color_sample_data:
        print("\n  # RGB/HSV color-based scale detection")
        print("  color_detection:")
        print("    enabled: true")
        print("    debug_color_masks: true")
        print("    scale_colors:")
        
        # Map color samples to appropriate color names
        background_data = color_sample_data[0]  # First sample is background
        marking_data = color_sample_data[1]     # Second sample is markings
        
        # Determine color names based on HSV values
        bg_color_name = determine_color_name(background_data['mean_hsv'])
        mark_color_name = determine_color_name(marking_data['mean_hsv'])
        
        print(f"      {bg_color_name}:")
        print("        enabled: true")
        print(f"        hsv_lower: {background_data['hsv_range']['lower']}")
        print(f"        hsv_upper: {background_data['hsv_range']['upper']}")
        print(f"        description: \"{bg_color_name.capitalize()} scale background\"")
        print()
        print(f"      {mark_color_name}:")
        print("        enabled: true")
        print(f"        hsv_lower: {marking_data['hsv_range']['lower']}")
        print(f"        hsv_upper: {marking_data['hsv_range']['upper']}")
        print(f"        description: \"{mark_color_name.capitalize()} scale markings\"")
    
    # Add water color configuration if water sample was provided
    if water_sample_data:
        print()
        print("  # Water color detection (for color-based method)")
        print("  water_hsv_lower:", water_sample_data['hsv_range']['lower'])
        print("  water_hsv_upper:", water_sample_data['hsv_range']['upper'])
    
    print()
    print("Additional notes:")
    print(f"- Scale region: {x_max-x_min} x {y_max-y_min} pixels")
    print(f"- Scale width: {((x_max-x_min)/width)*100:.1f}% of image width")
    print(f"- Scale height: {((y_max-y_min)/height)*100:.1f}% of image height")
    if color_sample_data:
        print(f"- Background color: {determine_color_name(color_sample_data[0]['mean_hsv']).capitalize()}")
        print(f"- Marking color: {determine_color_name(color_sample_data[1]['mean_hsv']).capitalize()}")
    if water_sample_data:
        water_color_name = determine_color_name(water_sample_data['mean_hsv'])
        print(f"- Water color: {water_color_name.capitalize()}")

def get_scale_measurements():
    """Get scale measurement inputs from user"""
    print("\n" + "="*60)
    print("SCALE MEASUREMENT INPUT")
    print("="*60)
    print("Enter the scale readings to calculate accurate cm/pixel ratio:")
    
    try:
        top_measurement = float(input("Enter scale measurement at TOP of outlined scale (cm): "))
        waterline_measurement = float(input("Enter scale measurement at WATERLINE position (cm): "))
        
        if top_measurement <= waterline_measurement:
            print("Warning: Top measurement should be higher than waterline measurement")
            print("(assuming scale increases going up)")
            
        print(f"\nMeasurement difference: {abs(top_measurement - waterline_measurement):.2f} cm")
        return top_measurement, waterline_measurement
        
    except (ValueError, EOFError):
        print("Invalid input - using default measurements")
        return None, None

def generate_calibration_data_enhanced(image, result_data, top_measurement=None, waterline_measurement=None):
    """Generate enhanced calibration.yaml data from scale analysis"""
    scale_corners = result_data['scale_corners']
    waterline_points = result_data['waterline_points']
    
    if len(scale_corners) != 4:
        print("Cannot generate calibration data: Need exactly 4 corner points")
        return None
    
    if len(waterline_points) != 2:
        print("Cannot generate calibration data: Need exactly 2 waterline points")
        return None
    
    # Calculate pixel distances
    scale_x_coords = [p[0] for p in scale_corners]
    scale_y_coords = [p[1] for p in scale_corners]
    
    # Find top and bottom of scale
    scale_top_y = min(scale_y_coords)
    scale_bottom_y = max(scale_y_coords)
    
    # Calculate waterline Y position (average of two waterline points)
    waterline_y = (waterline_points[0][1] + waterline_points[1][1]) / 2
    
    # Get measurements from user if not provided
    if top_measurement is None or waterline_measurement is None:
        top_measurement, waterline_measurement = get_scale_measurements()
        if top_measurement is None or waterline_measurement is None:
            print("Cannot calculate calibration without valid measurements")
            return None
    
    # Calculate pixel distance from top of scale to waterline
    pixel_distance_top_to_waterline = abs(waterline_y - scale_top_y)
    
    # Calculate cm distance from measurements
    cm_distance_top_to_waterline = abs(top_measurement - waterline_measurement)
    
    # Calculate pixels per cm from this measurement
    if cm_distance_top_to_waterline > 0:
        pixels_per_cm = pixel_distance_top_to_waterline / cm_distance_top_to_waterline
    else:
        print("Error: No measurement difference - cannot calculate calibration")
        return None
    
    print(f"\nEnhanced Calibration Calculations:")
    print(f"- Top of scale position: Y={scale_top_y}")
    print(f"- Waterline position: Y={waterline_y:.1f}")
    print(f"- Pixel distance (top to waterline): {pixel_distance_top_to_waterline:.1f} pixels")
    print(f"- Measurement at top: {top_measurement} cm")
    print(f"- Measurement at waterline: {waterline_measurement} cm")
    print(f"- Real distance (top to waterline): {cm_distance_top_to_waterline} cm")
    print(f"- Calculated pixels per cm: {pixels_per_cm:.3f}")
    print(f"- Current water level: {waterline_measurement} cm")
    
    # Generate enhanced calibration data
    from datetime import datetime
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate scale boundaries
    x_min, x_max = min(scale_x_coords), max(scale_x_coords)
    y_min, y_max = min(scale_y_coords), max(scale_y_coords)
    
    calib_data = {
        'pixels_per_cm': float(round(pixels_per_cm, 3)),
        'image_path': 'data/calibration/calibration_image.jpg',
        'calibration_date': datetime.now().isoformat(),
        'image_dimensions': {
            'width': img_width,
            'height': img_height,
            'resized': False  # analyze_scale_photo works with original images
        },
        'scale_measurements': {
            'top_measurement_cm': float(top_measurement),
            'waterline_measurement_cm': float(waterline_measurement),
            'measurement_difference_cm': float(cm_distance_top_to_waterline),
            'current_water_level_cm': float(waterline_measurement)
        },
        'reference_points': {
            'top_of_scale': {
                'x': int(x_min),
                'y': int(scale_top_y)
            },
            'bottom_of_scale': {
                'x': int(x_max), 
                'y': int(scale_bottom_y)
            },
            'waterline': {
                'x_left': int(waterline_points[0][0]),
                'y_left': int(waterline_points[0][1]),
                'x_right': int(waterline_points[1][0]),
                'y_right': int(waterline_points[1][1]),
                'y_average': int(waterline_y)
            }
        },
        'scale_boundaries': {
            'x_min': int(x_min),
            'x_max': int(x_max),
            'y_min': int(y_min),
            'y_max': int(y_max),
            # Add percentage coordinates for config compatibility
            'x_min_pct': round(x_min / img_width, 3),
            'x_max_pct': round(x_max / img_width, 3),
            'y_min_pct': round(y_min / img_height, 3),
            'y_max_pct': round(y_max / img_height, 3)
        },
        'calibration_method': 'enhanced_interactive_waterline',
        'confidence': 0.98,
        'notes': f'Generated by enhanced analyze_scale_photo.py with waterline detection on {img_width}x{img_height} image'
    }
    
    return calib_data

def save_calibration_data(calib_data):
    """Save calibration data to calibration.yaml"""
    import yaml
    from pathlib import Path
    
    calib_file = Path('data/calibration/calibration.yaml')
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(calib_file, 'w') as f:
        f.write('# Calibration data - generated by analyze_scale_photo.py\n')
        f.write('# This file can be used by the main calibration system\n')
        f.write('# Run with CALIBRATION_MODE=true to use this data\n\n')
        yaml.dump(calib_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCalibration data saved to: {calib_file}")
    return calib_file

def determine_color_name(hsv_values):
    """Determine the most appropriate color name based on HSV values"""
    h, s, v = hsv_values
    
    # Define color ranges in HSV space
    if v < 50:  # Very dark
        return "black"
    elif s < 60:  # Low saturation (white/gray family)
        if v > 180:  # High brightness = white
            return "white"
        elif v > 100:  # Medium brightness = light gray
            return "light_gray"
        else:  # Lower brightness = gray
            return "gray"
    else:  # Colored regions - higher saturation
        if 0 <= h <= 10 or 170 <= h <= 179:  # Red range (wraps around)
            return "red"
        elif 10 < h <= 25:
            return "orange"
        elif 25 < h <= 35:
            return "yellow"
        elif 35 < h <= 85:
            return "green"
        elif 85 < h <= 130:
            return "blue"
        elif 130 < h <= 170:
            return "purple"
        else:
            return "unknown"

def main():
    # Default to calibration image in data/calibration/ directory
    # Adjust path since we're now in src/calibration/ subdirectory
    default_image_path = "data/calibration/calibration_image.jpg"
    
    print("Scale Configuration Analysis Tool")
    print("="*50)
    print(f"Default image path: {default_image_path}")
    
    # Check if default image exists
    if os.path.exists(default_image_path):
        image_path = default_image_path
        print(f"Using calibration image: {image_path}")
    else:
        print(f"Calibration image not found at {default_image_path}")
        print("Please place your scale image as 'data/calibration/calibration_image.jpg'")
        print("Or enter a custom path:")
        
        custom_path = input("Enter custom image path (or press Enter to exit): ").strip()
        if not custom_path:
            print("Exiting...")
            return
        
        if not os.path.exists(custom_path):
            print(f"Error: Image not found at {custom_path}")
            return
        
        image_path = custom_path
    
    # 1. Basic analysis
    image = analyze_image_basic(image_path)
    if image is None:
        return
    
    # 2. Enhanced interactive coordinate and color picker
    print(f"\nStarting enhanced interactive analysis...")
    result_data = interactive_coordinate_picker(image, image_path)
    
    # Check if we have the required data
    if len(result_data['scale_corners']) != 4:
        print("Error: Need exactly 4 scale corner points")
        return
        
    if len(result_data['waterline_points']) != 2:
        print("Error: Need exactly 2 waterline points")
        return
    
    # Analyze color samples if provided
    color_sample_data = None
    if len(result_data['color_samples']) == 2:
        color_sample_data = analyze_color_samples(image, result_data['scale_corners'], result_data['color_samples'])
    
    # Analyze water sample if provided
    water_sample_data = None
    if len(water_samples) == 1:
        water_sample_data = analyze_water_color_sample(image, water_samples[0])
    
    # 3. Automatic edge detection
    automatic_edge_detection(image)
    
    # 4. Color analysis (traditional method for comparison)
    color_analysis(image)
    
    # 5. Generate suggestions with color and water data
    generate_config_suggestions(image, manual_points, color_sample_data, water_sample_data)
    
    # 6. Optionally generate calibration data
    if len(manual_points) == 4:
        print("\n" + "="*60)
        print("CALIBRATION DATA GENERATION")
        print("="*60)
        print("Would you like to generate calibration data for the detected scale?")
        print("This will create/update data/calibration/calibration.yaml")
        
        # Load config to get scale height if available
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            config_height = config.get('scale', {}).get('total_height')
        except:
            config_height = None
        
        try:
            response = input("Generate calibration data? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                calib_data = generate_calibration_data(image, manual_points, config_height)
                if calib_data:
                    save_calibration_data(calib_data)
                    print("\nCalibration data generated successfully!")
                    print("You can now use this with: CALIBRATION_MODE=true python src/main.py")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping calibration data generation")
    
    print(f"\nAnalysis complete! Use the suggested values to update your config.yaml")

if __name__ == "__main__":
    main()