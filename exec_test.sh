#!/bin/bash

# iterate over each yaml file in the current directory that starts with "background_"
for file in file_*.yaml; do
  # rename the file
  cp "$file" opencda/scenario_testing/config_yaml/platoon_test_town6.yaml

  # run the Python command
  python3 opencda.py -t platoon_test_town6 -v 0.9.12 --apply_ml --pldm

  # run the Python command
  python3 opencda.py -t platoon_test_town6 -v 0.9.12 --apply_ml

done
