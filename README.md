# Signal Processing

This is a signal processing and peak detection GUI application written in Python for the Industrial Project 2026 course.

## How to run the application

1. Create a folder named "data" at root
2. Place signal csv data into the "data" folder
3. Open terminal and create a virtual environment
4. Install requirements if needed
5. Run scripts (With venv)

### Requirements

Listed inside `requirements.txt`.

You can install all requirements with venv:

### Use a virtual environment

1. Create a virtual environment in the project root
2. Activate it
3. Install the project requirements

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

## Using the application

After launching the application you can

1. Select a `.csv` or `.dat` signal file to read using the "Browse" button.
2. Choose a downsampling method and factor.
   - "Decimation" with a factor of n only uses one in every n values of the signal.
   - "Max points" with a factor of n uses the maximum value in every n values of the signal.
   - Selecting "None" or setting the factor to 1 does no downsampling.
3. Check "Detect peaks" to run peak detection on the signal.
   - Skipping peak detection displays the signal slightly faster.
4. Click "Visualize Data" to run the signal analysis and show the results.

The graph will now display the signal in a time-amplitude graph. Found peaks are displayed as circles on the graph.