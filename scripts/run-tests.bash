#!/usr/bin/env bash
python -m unittest tests/test_normalize.py
python -m unittest tests/test_buffer.py
python -m unittest tests/test_network.py
python -m unittest tests/test_A2C.py
