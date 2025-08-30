#!/usr/bin/env python3
"""
Simple Interactive E-Template Extractor

Extracts only E-shaped templates (5cm markings) from stadia rod images:
- Black E-shaped templates (dark markings on light background)
- White E-shaped templates (light markings on dark background)
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Global variables for mouse handling
mouse_clicks = []
current_template_type = ""
image_display = None
image_original = None
extraction_active = False

def mouse_callback(event, x, y, flags, param):
    """Simple mouse callback with immediate visual feedback."""
    global mouse_clicks, image_display, extraction_active
    
    if not extraction_active:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Corner {len(mouse_clicks)+1} selected at ({x}, {y})")
        
        # Add click to list
        mouse_clicks.append((x, y))
        
        # Draw small dot immediately for feedback
        cv2.circle(image_display, (x, y), 4, (0, 255, 0), -1)  # Green dot
        cv2.circle(image_display, (x, y), 6, (255, 255, 255), 1)  # White border
        
        # Draw click number
        cv2.putText(image_display, str(len(mouse_clicks)), (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If we have 4 clicks, draw the rectangle
        if len(mouse_clicks) == 4:
            # Draw lines connecting the corners
            for i in range(4):
                start_point = mouse_clicks[i]
                end_point = mouse_clicks[(i + 1) % 4]  # Connect back to first point
                cv2.line(image_display, start_point, end_point, (0, 0, 255), 3)  # Red rectangle
            
            # Fill area with semi-transparent overlay
            overlay = image_display.copy()
            pts = np.array(mouse_clicks, np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))  # Cyan fill
            cv2.addWeighted(overlay, 0.3, image_display, 0.7, 0, image_display)
            
            print("Template region selected! Press 's' to save, 'r' to reset")
        else:
            print(f"Need {4 - len(mouse_clicks)} more corner(s). Click around the E-shaped marking.")
        
        # Update display with status
        update_display_with_status()

def update_display_with_status():
    """Update the display window with current status and instructions."""
    global image_display, current_template_type, mouse_clicks
    
    # Create a copy for overlay text
    display_copy = image_display.copy()
    
    # Get image dimensions to position overlay on right side
    height, width = display_copy.shape[:2]
    
    # Small overlay positioned on the right side
    overlay_width = 200
    overlay_height = 60
    overlay_x = width - overlay_width - 10
    overlay_y = 10
    
    # Add semi-transparent background for text (right side)
    overlay = display_copy.copy()
    cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + overlay_width, overlay_y + overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display_copy, 0.3, 0, display_copy)
    
    # Progress indicator with smaller font
    progress_text = f"Corners: {len(mouse_clicks)}/4"
    cv2.putText(display_copy, progress_text, (overlay_x + 10, overlay_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Next step instruction with smaller font
    if len(mouse_clicks) == 0:
        instruction = "Click 1st corner"
    elif len(mouse_clicks) < 4:
        instruction = f"Click corner {len(mouse_clicks)+1}"
    else:
        instruction = "Press 's' to save"
    
    cv2.putText(display_copy, instruction, (overlay_x + 10, overlay_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Update display
    cv2.imshow('E-Template Extractor', display_copy)

def reset_clicks():
    """Reset mouse clicks and restore original image."""
    global mouse_clicks, image_display, image_original
    
    mouse_clicks = []
    image_display = image_original.copy()
    print(f"Selection reset for {current_template_type.upper()} E-patterns")
    print(f"→ Click 4 corners around a {current_template_type} E-shaped marking")
    update_display_with_status()

def extract_template_from_clicks(image, clicks):
    """Extract template from 4 corner clicks."""
    if len(clicks) != 4:
        return None
    
    # Get bounding rectangle
    x_coords = [pt[0] for pt in clicks]
    y_coords = [pt[1] for pt in clicks]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    # Extract template
    template = image[y_min:y_max, x_min:x_max].copy()
    
    if template.size == 0:
        print("Error: Template region is empty")
        return None
    
    print(f"Extracted template: {template.shape[1]}x{template.shape[0]} pixels")
    return template

def save_template(template, template_type, output_dir):
    """Save extracted template with metadata."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"e_template_{template_type}_{timestamp}.png"
    filepath = output_dir / filename
    
    # Save template image
    success = cv2.imwrite(str(filepath), template)
    
    if success:
        print(f"Template saved: {filepath}")
        
        # Save metadata
        metadata = {
            'template_type': template_type,
            'size_cm': 5.0,  # E-templates represent 5cm markings
            'extraction_date': datetime.now().isoformat(),
            'dimensions': {
                'width': template.shape[1],
                'height': template.shape[0]
            },
            'corners': mouse_clicks,
            'suggested_threshold': 0.6,  # Good default for E-patterns
            'description': f'{template_type.capitalize()} E-shaped marking (5cm graduation)'
        }
        
        metadata_file = filepath.with_suffix('.yaml')
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        return True
    else:
        print(f"Failed to save template: {filepath}")
        return False

def extract_e_templates():
    """Main function to extract E-shaped templates."""
    global image_display, image_original, extraction_active, current_template_type
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("="*60)
    print("SIMPLE E-TEMPLATE EXTRACTOR FOR STADIA RODS")
    print("="*60)
    print("This tool extracts E-shaped templates representing 5cm graduations.")
    print("You'll extract 2 types:")
    print("  1. BLACK E-templates (dark markings on light background)")
    print("  2. WHITE E-templates (light markings on dark background)")
    print()
    
    # Load calibration image
    project_root = Path(__file__).parent.parent
    calib_dir = project_root / 'data' / 'calibration'
    calibration_image = calib_dir / 'calibration_image.jpg'
    
    if not calibration_image.exists():
        print(f"Calibration image not found: {calibration_image}")
        print("Available images in calibration directory:")
        if calib_dir.exists():
            for img_file in calib_dir.glob('*.jpg'):
                print(f"  - {img_file.name}")
        return
    
    # Load image
    image_original = cv2.imread(str(calibration_image))
    if image_original is None:
        print(f"Failed to load image: {calibration_image}")
        return
    
    image_display = image_original.copy()
    
    print(f"Loaded image: {calibration_image}")
    print(f"Image size: {image_original.shape[1]}x{image_original.shape[0]} pixels")
    print()
    
    # Setup output directory
    output_dir = project_root / 'data' / 'pattern_templates' / 'scale_markings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create window
    window_name = 'E-Template Extractor'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)
    cv2.imshow(window_name, image_display)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    templates_extracted = 0
    
    # Extract templates for both types
    template_types = [
        {
            'name': 'black',
            'description': 'Black E-shaped markings (dark on light background)',
            'instructions': 'Select BLACK E-shaped markings - dark patterns on light background'
        },
        {
            'name': 'white', 
            'description': 'White/light E-shaped markings (light on dark background)',
            'instructions': 'Select WHITE E-shaped markings - light patterns on dark background'
        }
    ]
    
    for template_type in template_types:
        current_template_type = template_type['name']
        extraction_active = True
        
        print("\n" + "="*70)
        print(f"STEP {template_types.index(template_type)+1}/2: EXTRACTING {template_type['name'].upper()} E-TEMPLATES")
        print("="*70)
        print(f"Target: {template_type['description']}")
        
        if template_type['name'] == 'black':
            print("\nWHAT TO LOOK FOR - BLACK E-TEMPLATES:")
            print("   - Dark E-shaped markings on light (white/yellow) background")
            print("   - Usually the most common type on stadia rods")  
            print("   - Look for clear, bold E-patterns about 5cm tall")
            print("   - These are dark horizontal lines with gaps forming an E shape")
            print("   - Found on white or light yellow sections of the rod")
            print("   - Example pattern: ===   (three dark horizontal lines)")
            print("                     =     (with gaps on the right)")
            print("                     ===   (forming an E shape)")
        else:
            print("\nWHAT TO LOOK FOR - WHITE E-TEMPLATES:")
            print("   - Light/white E-shaped markings on dark background")
            print("   - Less common, usually on colored sections of the rod")
            print("   - Look for light-colored E-patterns that stand out from dark areas")
            print("   - These are light horizontal lines with gaps forming an E shape")
            print("   - Found on blue, red, or other dark colored sections of the rod")
            print("   - Example pattern: ---   (three light horizontal lines)")
            print("                     -     (with gaps on the right)")
            print("                     ---   (forming an E shape on dark background)")
        
        print("\nEXTRACTION PROCESS:")
        print("   1. Find a clear E-shaped marking in the image")
        print("   2. Click 4 corners around it: top-left → top-right → bottom-right → bottom-left")
        print("   3. Green dots will show your clicks, red rectangle will appear when complete")
        print("   4. Press 's' to save the template, or 'r' to reset and try again")
        
        print("\nCONTROLS:")
        print("   's' = Save template    'r' = Reset selection    'n' = Skip to next type    'q' = Quit")
        
        print(f"\nReady to extract {template_type['name'].upper()} E-templates!")
        print(f"   → Look at the image window and click 4 corners around a {template_type['name']} E-pattern...")
        
        reset_clicks()
        templates_for_type = 0
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                print("Extraction cancelled by user")
                cv2.destroyAllWindows()
                return templates_extracted
            
            elif key == ord('r'):  # Reset
                print("Resetting selection...")
                reset_clicks()
            
            elif key == ord('s'):  # Save template
                if len(mouse_clicks) == 4:
                    template = extract_template_from_clicks(image_original, mouse_clicks)
                    
                    if template is not None:
                        # Show preview
                        preview = cv2.resize(template, (150, 150), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow('Template Preview', preview)
                        
                        print(f"Extracted {template_type['name'].upper()} E-template: {template.shape[1]}x{template.shape[0]} pixels")
                        print("Review the preview window. Save this template?")
                        print("   Press 'y' to SAVE and continue, 'n' to DISCARD and try again")
                        
                        # Wait for confirmation
                        while True:
                            confirm_key = cv2.waitKey(0) & 0xFF
                            if confirm_key == ord('y'):
                                if save_template(template, template_type['name'], output_dir):
                                    templates_for_type += 1
                                    templates_extracted += 1
                                    print(f"SUCCESS! {template_type['name'].upper()} E-template saved!")
                                    print(f"   Total {template_type['name']} templates: {templates_for_type}")
                                else:
                                    print("ERROR: Failed to save template")
                                break
                            elif confirm_key == ord('n'):
                                print("Template discarded - try selecting a different area")
                                break
                        
                        cv2.destroyWindow('Template Preview')
                        reset_clicks()
                        
                        # Ask if user wants to extract more of this type
                        print(f"\nExtract another {template_type['name'].upper()} E-template?")
                        print("   'y' = Extract more of the same type")
                        print("   'n' = Move to next template type")
                        print("   Choice (y/n): ", end='')
                        while True:
                            more_key = cv2.waitKey(0) & 0xFF
                            if more_key == ord('y'):
                                print("YES - Find another E-template of the same type...")
                                break
                            elif more_key == ord('n'):
                                print("NO - Moving to next template type...")
                                break
                        
                        if more_key == ord('n'):
                            break
                    else:
                        print("Failed to extract template from selection")
                else:
                    print(f"Need 4 corner clicks to save template (current: {len(mouse_clicks)})")
            
            elif key == ord('n'):  # Skip to next type
                print(f"Skipping {template_type['name']} templates...")
                break
        
        print(f"Extracted {templates_for_type} {template_type['name']} E-templates")
        extraction_active = False
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("E-TEMPLATE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total templates extracted: {templates_extracted}")
    print(f"Templates saved to: {output_dir}")
    
    if templates_extracted > 0:
        print("\nNext steps:")
        print("1. Update config.yaml to use extracted templates:")
        print("   template_matching:")
        print("     template_source: 'manager'")
        print("2. Test pattern-aware detection with your templates")
    else:
        print("No templates extracted. The built-in default templates will be used.")
    
    return templates_extracted

if __name__ == "__main__":
    try:
        extract_e_templates()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()