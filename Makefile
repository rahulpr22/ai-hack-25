# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Create virtual environment and install dependencies
setup:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete! Run 'make run' to start."

# Run the script with the correct virtual environment
# TODO: set it up accordingly
run:
	$(VENV_DIR)/bin/python main.py

# Clean up (remove venv)
clean:
	rm -rf $(VENV_DIR)
	@echo "🧹 Virtual environment removed!"
