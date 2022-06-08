#!/usr/bin/env bash
poetry export -f requirements.txt --output requirements.txt
git add requirements.txt