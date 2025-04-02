#!/bin/bash
pipenv run python ptl.py fit --config config.yaml 2>&1 | tee fit.log