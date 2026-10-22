#!/bin/bash

BASE_DATA_DIR="dataset"
PREDICTION_BASE="$BASE_DATA_DIR/marigold_predictions"

echo "Base directory: $BASE_DATA_DIR"
echo "Prediction base: $PREDICTION_BASE"

# Create the main prediction directory
mkdir -p "$PREDICTION_BASE"

# Process each sequence
for sequence_dir in "$BASE_DATA_DIR"/lms_kamal_*; do
    if [ -d "$sequence_dir" ]; then
        sequence_name=$(basename "$sequence_dir")
        
        echo -e "\n=== Processing sequence: $sequence_name ==="
        
        # Check if Marigold output exists for this sequence
        MARIGOLD_OUTPUT="$sequence_dir/marigold_zero_shot/depth_npy"
        PREDICTION_TARGET="$PREDICTION_BASE/$sequence_name/sharpen_rgb/PNG"
        
        if [ ! -d "$MARIGOLD_OUTPUT" ]; then
            echo "  SKIPPING: No marigold_zero_shot/depth_npy folder found"
            echo "  Checked: $MARIGOLD_OUTPUT"
            continue
        fi
        
        echo "  Source: $MARIGOLD_OUTPUT"
        echo "  Target: $PREDICTION_TARGET"
        
        # Clean target directory if it exists
        if [ -d "$PREDICTION_TARGET" ]; then
            echo "  Cleaning existing target directory..."
            rm -f "$PREDICTION_TARGET"/*.npy 2>/dev/null
        else
            mkdir -p "$PREDICTION_TARGET"
        fi
        
        # Process each .npy file
        file_count=0
        error_count=0
        first_files=""
        
        for npy_file in "$MARIGOLD_OUTPUT"/*.npy; do
            if [ -f "$npy_file" ]; then
                filename=$(basename "$npy_file")
                
                # Extract base name (e.g., "image100" from "image100_depth.npy")
                if [[ "$filename" == *"_depth.npy" ]]; then
                    base_name="${filename%_depth.npy}"
                    new_filename="pred_${base_name}.npy"
                    
                    # Copy and rename
                    cp "$npy_file" "$PREDICTION_TARGET/$new_filename"
                    
                    # Track first 3 files for logging
                    if [ $file_count -lt 3 ]; then
                        first_files="$first_files\n    $filename -> $new_filename"
                    fi
                    
                    file_count=$((file_count + 1))
                else
                    echo "  WARNING: Unexpected filename pattern: $filename"
                    error_count=$((error_count + 1))
                fi
            fi
        done
        
        echo -e "  Processed $file_count files successfully"
        if [ $error_count -gt 0 ]; then
            echo "  WARNING: $error_count files had unexpected naming"
        fi
        
        if [ $file_count -gt 0 ]; then
            echo -e "  First 3 renames:$first_files"
            
            # Quick verification
            echo -e "  Verification - First 3 files in target:"
            ls -1 "$PREDICTION_TARGET"/pred_*.npy 2>/dev/null | head -3 | while read file; do
                echo "    $(basename "$file")"
            done
        fi
        
        # Count and show statistics
        target_count=$(ls "$PREDICTION_TARGET"/*.npy 2>/dev/null | wc -w)
        source_count=$(ls "$MARIGOLD_OUTPUT"/*.npy 2>/dev/null | wc -w)
        echo "  Source files: $source_count | Target files: $target_count"
        
        if [ $source_count -ne $target_count ]; then
            echo "  WARNING: File count mismatch!"
        fi
    fi
done
