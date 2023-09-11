#!/bin/bash

max_iterations=$1
counter=0

# iterate over each yaml file in the current directory that starts with "background_"
for file in file_*.yaml; do

  # Break the loop if the counter reaches the max_iterations
  if [[ $counter -ge $max_iterations ]]; then
    break
  fi

  # rename the file
  cp "$file" opencda/scenario_testing/config_yaml/platoon_test_docker.yaml

  # run PLDM simulation
  python3.7 opencda.py -t platoon_test_docker -v 0.9.12 --apply_ml --pldm

  # run LDM simulation
  python3.7 opencda.py -t platoon_test_docker -v 0.9.12 --apply_ml

  # Increment the counter
  ((counter++))

done
