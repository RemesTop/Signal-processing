# Signal Processing

This is a python project intended for the course Industrial Project 2026

## How to use

- Create a folder named "data" at root
- Place signal csv data into the "data" folder
- Open terminal and create a virtual environment
- Install requirements if needed
- Run scripts (With venv)

## Requirements

Listed inside requirements.txt

You can install all requirements with venv:

## Use a virtual environment

- Create a virtual environment in the project root
- Activate it
- Install the project requirements

Example:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

Then run scripts from the activated venv:

```powershell
python read_data.py
```

The venv is used to keep the project packages separate from the rest of the Python installation on your computer.