#!/bin/bash
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "Running all python notebooks in notebooks/..."

for script in *.py; do
    if [ "$script" == "paths.py" ]; then
        echo "Skipping paths.py (likely a utility module)"
        continue
    fi
    
    echo "----------------------------------------------------------------"
    echo "Running $script..."
    echo "----------------------------------------------------------------"
    
    python "$script"
    
    echo "Successfully ran $script"
    echo ""
done

echo "All notebooks executed successfully."
