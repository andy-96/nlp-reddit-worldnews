#!/bin/bash

git pull
pipenv run python3 data_acquisition.py
git add .
git commit -m "new fetch"
git push -u origin
