#!/bin/bash

# Export Python package information
pip list --format=freeze > requirements.txt

# Export Conda environment information
conda env export > environment.yml