#!/usr/bin/env bash
# Step 2: Print dataset download instructions.
# Datasets require manual form/registration — run this to see what's needed.
set -e
cd "$(dirname "$0")/.."
python -m src.data.download
