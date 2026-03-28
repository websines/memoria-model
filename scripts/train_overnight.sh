#!/bin/bash
# Overnight training — 8 hour budget on 2x 3090

source .venv/bin/activate
python -m memoria train --config small --time-budget 28800
