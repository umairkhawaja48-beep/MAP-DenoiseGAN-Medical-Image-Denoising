#!/bin/bash
# Evaluate estimated (blind) routing vs. oracle routing.
set -e
cd "$(dirname "$0")/../src"
python eval_estimated_routing.py
