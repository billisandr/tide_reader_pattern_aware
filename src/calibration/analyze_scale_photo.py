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
    """Interactive tool to pick scale coordinates and colors"""
    print("\n" + "="*60)
    print("INTERACTIVE COORDINATE & COLOR PICKER")
    print("="*60)
    print("STEP 1: Click on the following scale boundary points in order:")
    print("IMPORTANT: Select the FULL scale boundaries, even if parts are underwater!")
    print("1. Top-left corner of scale (above or below water)")
    print("2. Top-right corner of scale (above or below water)") 
    print("3. Bottom-left corner of scale (above or below water)")
    print("4. Bottom-right corner of scale (above or below water)")
    print("\nSTEP 2: After 4 points, click on scale color samples:")
    print("5. Click on scale background color (preferably on visible portions)")
    print("6. Click on scale marking/text color (preferably on visible portions)")
    print("\nSTEP 3: After scale colors, click on water color sample:")
    print("7. Click on water color (if visible in image)")
    print("\nControls: 'r' to reset, 'q' to quit, 's' to skip color selection, 'w' to skip water color")
    
    # Create a copy for drawing
    display_image = image.copy()
    points = []
    color_samples = []
    water_samples = []
    point_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Scale Background", "Scale Markings", "Water Color"]
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 255, 128)]
    
    # State tracking
    picking_corners = True
    picking_colors = False
    picking_water = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, color_samples, water_samples, display_image, picking_corners, picking_colors, picking_water
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if picking_corners and len(points) < 4:
                # Handle corner point selection
                points.append((x, y))
                color = colors[len(points)-1]
                cv2.circle(display_image, (x, y), 5, color, -1)
                cv2.putText(display_image, f"{len(points)}: {point_labels[len(points)-1]}", 
                           (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"Point {len(points)}: {point_labels[len(points)-1]} at ({x}, {y})")
                cv2.imshow('Scale Analysis', display_image)
                
                if len(points) == 4:
                    analyze_picked_points(points, image.shape)
                    print("\n" + "="*60)
                    print("STEP 2: COLOR SELECTION")
                    print("="*60)
                    print("Now click on representative colors:")
                    print("- Click on the BACKGROUND color of your scale")
                    print("- Then click on the MARKING/TEXT color")
                    print("- Press 's' to skip color selection")
                    picking_corners = False
                    picking_colors = True
            
            elif picking_colors and len(color_samples) < 2:
                # Handle color sample selection
                sample_idx = len(color_samples) + 4  # Offset by 4 corner points
                color_samples.append((x, y))
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
    return points, color_samples, water_samples

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

def generate_calibration_data(image, manual_points, config_height_cm=None):
    """Generate calibration.yaml data from analyzed scale points"""
    if len(manual_points) != 4:
        print("Cannot generate calibration data: Need exactly 4 corner points")
        return None
    
    # Calculate scale dimensions from manual points
    x_coords = [p[0] for p in manual_points]
    y_coords = [p[1] for p in manual_points]
    
    scale_height_pixels = max(y_coords) - min(y_coords)
    
    # Use provided height or prompt user
    if config_height_cm:
        scale_height_cm = config_height_cm
        print(f"\nUsing configured scale height: {scale_height_cm}cm")
    else:
        print(f"\nCalculated scale height: {scale_height_pixels} pixels")
        try:
            scale_height_cm = float(input("Enter the actual scale height in cm: "))
        except (ValueError, EOFError):
            print("Invalid input - using default 100cm")
            scale_height_cm = 100.0
    
    # Calculate pixels per cm
    pixels_per_cm = scale_height_pixels / scale_height_cm
    
    print(f"\nCalibration calculations:")
    print(f"- Scale height in pixels: {scale_height_pixels}")
    print(f"- Scale height in cm: {scale_height_cm}")
    print(f"- Pixels per cm: {pixels_per_cm:.2f}")
    
    # Generate calibration data
    from datetime import datetime
    
    calib_data = {
        'pixels_per_cm': float(round(pixels_per_cm, 2)),
        'image_path': 'data/calibration/calibration_image.jpg',
        'scale_height_cm': float(scale_height_cm),
        'calibration_date': datetime.now().isoformat(),
        'reference_points': {
            'top_of_scale': {
                'x': int(min(x_coords)),
                'y': int(min(y_coords))
            },
            'bottom_of_scale': {
                'x': int(max(x_coords)), 
                'y': int(max(y_coords))
            }
        },
        'calibration_method': 'interactive_analysis',
        'confidence': 0.95,
        'notes': 'Generated by analyze_scale_photo.py interactive analysis tool'
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
    
    # 2. Interactive coordinate and color picker
    print(f"\nStarting interactive analysis...")
    manual_points, color_samples, water_samples = interactive_coordinate_picker(image, image_path)
    
    # Analyze color samples if provided
    color_sample_data = None
    if len(color_samples) == 2:
        color_sample_data = analyze_color_samples(image, manual_points, color_samples)
    
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