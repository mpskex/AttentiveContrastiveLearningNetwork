.PHONY: install

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

install: 
	pip install -e .
	python3 setup.py
