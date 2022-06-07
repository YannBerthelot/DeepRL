#!/usr/bin/env bash
poetry run python -m unittest tests/test_normalize.py
poetry run python -m unittest tests/test_buffer.py
poetry run python -m unittest tests/test_network.py
poetry run python -m unittest tests/test_A2C.py
