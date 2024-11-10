#!/bin/bash

# Download required model files
pwd
tar -xf ./data/tokenizer_all_sets.tar --directory ./data/ && rm ./data/tokenizer_all_sets.tar

# Install dependencies
# pip install -r requirements.txt