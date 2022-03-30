#!/bin/bash
set -e
export FORCE_CUDA=1
export MAX_JOBS=16
python setup.py develop
pytest -s 