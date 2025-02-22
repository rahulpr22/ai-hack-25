.PHONY: setup clean

# Python virtual environment variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	brew install libmagic
	# Create virtual environment if it doesn't exist
	python3 -m venv $(VENV)
	# Upgrade pip
	$(PIP) install --upgrade pip
	# Install requirements
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete! Run 'make run' to start."

# Run the script with the correct virtual environment
# TODO: set it up accordingly
run:
	streamlit run main.py

clean:
	# Remove virtual environment
	rm -rf $(VENV)
	# Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "🧹 Virtual environment removed!"
