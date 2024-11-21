#!/bin/bash

# Define the path to your specific Python interpreter
# python_command="/home/huynn/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh"
python_command="python"

# Check if at least one script path is provided
if [ $# -eq 0 ]; then
    echo "No script paths provided."
    exit 1
fi

# Loop through each provided script path and execute the Python script
for script_path in "$@"
do
    if [ -f "$script_path" ]; then
        echo "Running $script_path..."
        yes '' | "$python_command" "$script_path"  # Use 'yes' to auto-press Enter
        echo "$script_path completed."
    else
        echo "File $script_path does not exist."
    fi
done

echo "All scripts have been executed."
