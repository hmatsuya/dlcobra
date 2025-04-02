#/bin/bash
python setup.py clean --all
rm -rf build/*
rm -rf build dist dlshogi.egg-info
pipenv run pip install . --force-reinstall
python -c "import dlshogi; print(dlshogi.__version__)"