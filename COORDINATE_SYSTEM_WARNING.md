# ⚠️ Image Resizing and Coordinate System Warning

## Current System Limitations

### The Problem

The current water level detection system uses **absolute pixel coordinates** for scale detection that **do not adjust** when images are resized. This causes major accuracy issues.

### Impact

- **Scale detection fails completely** when `resize_width` is set to any value other than `null`
- **Hardcoded coordinates** in config.yaml become incorrect after resizing
- **Water level measurements** become inaccurate or impossible

## Current Workarounds

### ✅ Recommended Solution

**Disable image resizing entirely:**

```yaml
processing:
  resize_width: null  # Set to null - do NOT resize images
```

### Why This Works

- Scale coordinates in config.yaml remain accurate
- No coordinate transformation needed  
- Enhanced calibration works properly
- Water detection accuracy is maintained

## Future Improvements Needed

### Relative Coordinate System (Not Implemented)

The system needs to be updated to use **percentage-based coordinates** instead of absolute pixels:

```yaml
# Future implementation needed:
scale:
  expected_position:
    x_min_pct: 0.106  # 10.6% from left edge
    x_max_pct: 0.241  # 24.1% from left edge  
    y_min_pct: 0.002  # 0.2% from top edge
    y_max_pct: 0.899  # 89.9% from top edge
```

### Required Changes

1. **Update water_level_detector.py** to calculate absolute coordinates from percentages
2. **Modify calibration.py** to save percentage coordinates
3. **Update config.yaml structure** to use relative coordinates
4. **Add coordinate conversion** in processing pipeline

## Current Status

### ✅ What Works

- **Enhanced calibration workflow**: Generates accurate data for original image sizes
- **Waterline reference system**: Now integrated with main processing
- **Standard processing**: Works correctly with `resize_width: null`

### ❌ What Doesn't Work  

- **Image resizing**: Causes coordinate mismatch
- **Relative coordinates**: Not implemented in main processing system
- **Dynamic scaling**: Coordinates don't adapt to different image sizes

## Best Practices

### For Current System

1. **Always set** `resize_width: null` in config.yaml
2. **Use enhanced calibration** for maximum accuracy
3. **Take calibration images** at full resolution
4. **Verify scale detection** in debug images before processing

### For Future Development

1. **Implement percentage-based coordinates** system-wide
2. **Add coordinate conversion** between absolute and relative
3. **Update all coordinate references** to use percentage calculations
4. **Test thoroughly** with various image sizes and resize settings

## Migration Path

### Phase 1 (Current)

- Use `resize_width: null`
- Enhanced calibration with waterline detection
- Absolute coordinate system

### Phase 2 (Future)

- Implement relative coordinate system
- Enable safe image resizing
- Backward compatibility with absolute coordinates

### Phase 3 (Advanced)

- Auto-detection of coordinate system type
- Dynamic scaling based on image analysis
- Fully resolution-independent processing

---

**Current Recommendation**: Keep `resize_width: null` and use the enhanced calibration workflow for best results.
