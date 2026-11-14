# Signal Processing

This is a signal processing and peak detection GUI application written in Python for the Industrial Project 2026 course.

## How to run the application

1. Open a terminal and create a virtual environment
   - Optional but recommended. The rest of these instructions assume you're using the virtual environment.
2. Install requirements if needed
3. Run scripts (with venv)

Once you have the application running, see [Using the application](#using-the-application).

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
   - A csv file will be converted to a dat file, which can take a considerable amount of time for larger files. Once you have the dat file you can always open that to avoid the conversion process.
2. Choose a downsampling method and factor.
   - "Decimation" with a factor of n only uses one in every n values of the signal.
   - "Max points" with a factor of n uses the maximum value in every n values of the signal.
   - Selecting "None" or setting the factor to 1 does no downsampling.
3. Select which channels to analyze.
   - The signal files are expected to contain two channels of data.
   - If only one is needed, you can select that from the dropdown to speed up the signal analysis.
4. Click "Run" to run the signal analysis and show the results.

The graph will now display the signal in a time-amplitude graph. Found peaks are displayed as circles on the graph. The "Peaks" label in the top bar of the application shows the number of found peaks.

### Editing the signal processing and peak finding

Check "View" > "Analysis functions and parameters" to select which functions process the signal and to edit their parameters. The values of the parameters can be `None` for no value or the default value, simple numerical values, calculations, or a tuple like `(10, None)` or `(-5, 5)` for the parameters that support that. Click "Apply" to re-process the current signal and update the graph. Doing that will also save the selected functions and the parameters so they persist after closing the app.

If you set the "distance" field to `None`, most signal analysis functions currently will use a minimum distance of 5 milliseconds between peaks (adjusted for the decimation/downsampling).