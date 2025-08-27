# Enhanced Waterline Detection and Calibration Guide

## Overview

This guide covers the new enhanced calibration workflow that incorporates waterline detection and real scale measurements for maximum accuracy in water level detection systems.

## Key Improvements

### 🆕 What's New

1. **Waterline-Aware Calibration**: Direct incorporation of current waterline position
2. **Real Scale Measurements**: Uses actual scale readings instead of estimates
3. **Submersion Handling**: Works with partially underwater scales
4. **Enhanced Accuracy**: Precise cm/pixel calculation from measured segments

## Enhanced Workflow Process

### Step 1: Prepare Your Calibration Image

- Take a clear photo of your measurement scale
- Ensure the scale is visible (even if partially underwater)
- Note the current water level and corresponding scale readings
- Save image as `data/calibration/calibration_image.jpg`

### Step 2: Run Enhanced Analysis

```bash
cd C:\tide-level-img-proc-backup
python src/calibration/analyze_scale_photo.py
```

### Step 3: Interactive Calibration Process

#### Phase 1: Scale Boundary Selection
1. **Click 4 corners** of the ENTIRE visible scale:
   - Top-left corner of scale
   - Top-right corner of scale  
   - Bottom-left corner of scale
   - Bottom-right corner of scale
   
   **Important**: Select the full scale boundary, even if parts are underwater

#### Phase 2: Waterline Marking
2. **Click 2 waterline points**:
   - Left edge of scale at current waterline
   - Right edge of scale at current waterline
   
   **Tip**: These points should be horizontally aligned at the water surface

#### Phase 3: Real Scale Measurements
3. **Enter actual scale readings**:
   - **Top measurement**: Scale reading at the TOP of your outlined area (e.g., 485.0 cm)
   - **Waterline measurement**: Scale reading at the current waterline (e.g., 420.0 cm)
   
   **Example**: If top shows 485cm and waterline shows 420cm, the system calculates precise pixels/cm from the 65cm difference

#### Phase 4: Optional Color Sampling
4. **Click color samples** (optional but recommended):
   - Scale background color
   - Scale marking/text color
   - Water color (if visible)

### Step 4: Apply Results

The enhanced analysis generates:

1. **Updated config.yaml suggestions** with optimal coordinates
2. **Enhanced calibration.yaml** with waterline data
3. **Precise pixels/cm ratio** from real measurements

## Benefits of Enhanced Calibration

### Accuracy Improvements

- **Real measurements**: No guessing or estimates
- **Waterline reference**: Direct water level incorporation  
- **Precise calculations**: Based on actual measured distance
- **Submersion handling**: Works with underwater scale portions

### Data Quality

- **Higher confidence**: 0.98 vs 0.95 for estimated methods
- **Waterline position**: Stored for future reference
- **Complete metadata**: Image dimensions, measurements, coordinates
- **Enhanced traceability**: Full calibration provenance

## Generated Calibration Data Structure

```yaml
pixels_per_cm: 12.546                         # Precise from real measurements
scale_measurements:
  top_measurement_cm: 485.0                  # Your entered top reading
  waterline_measurement_cm: 420.0            # Your entered waterline reading  
  measurement_difference_cm: 65.0             # Calculated difference
  current_water_level_cm: 420.0              # Current water level
reference_points:
  waterline:                                # Waterline position data
    x_left: 78                             # Left waterline point
    y_left: 245                            
    x_right: 185                           # Right waterline point
    y_right: 247
    y_average: 246                         # Average Y position
calibration_method: 'enhanced_interactive_waterline'
confidence: 0.98                              # Higher confidence
```

## Troubleshooting

### Tool Won't Start
- Ensure you're running from project root directory
- Check display is available (not headless/SSH environment)
- For WSL: Use Windows Python or enable X11 forwarding

### Scale Detection Still Failing
- Disable image resizing: Set `resize_width: null` in config.yaml
- Run calibration on original image dimensions
- Use generated percentage coordinates from enhanced analysis

### Inaccurate Measurements
- Ensure waterline points are precisely at water surface
- Double-check scale reading inputs for accuracy
- Verify scale outline covers full visible scale area

## Best Practices

### For Maximum Accuracy
1. Use the enhanced calibration workflow for all new installations
2. Take calibration photos during stable water conditions
3. Ensure good lighting and clear scale visibility
4. Enter scale measurements carefully and double-check values

### For Partially Submerged Scales
1. Outline the FULL scale boundary, not just visible portions
2. Mark waterline precisely at current water surface
3. Use actual scale readings from both visible and calculated positions
4. The system handles underwater portions automatically

### For System Updates
1. Re-run enhanced calibration when:
   - Camera position changes
   - Scale position changes  
   - Water conditions change significantly
   - Detection accuracy decreases

## Integration with Main System

After running enhanced calibration:

```bash
# Apply the enhanced calibration data
set CALIBRATION_MODE=true && python src/main.py

# Start normal processing with enhanced accuracy
python src/main.py
```

The main system will automatically use the enhanced waterline-aware calibration for all subsequent water level measurements.