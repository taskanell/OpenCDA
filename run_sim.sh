#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_file_number> <destination_file_name> <use_pldm>"
    exit 1
fi

file="file_$1.yaml"
script_file="opencda/scenario_testing/config_yaml/$2.yaml"
use_pldm=$3

cp "$file" "$script_file"

if [ "$use_pldm" == "pldm" ]; then
    python3.7 opencda.py -t $2 -v 0.9.12 --apply_ml --pldm
    if [ $? -ne 0 ]; then
        echo "Error: PLDM simulation failed."
        exit 1
    fi
elif [ "$use_pldm" == "ldm" ]; then
    python3.7 opencda.py -t $2 -v 0.9.12 --apply_ml
    if [ $? -ne 0 ]; then
        echo "Error: LDM simulation failed."
        exit 1
    fi
else
    echo "Invalid value for use_pldm."
    exit 1
fi

