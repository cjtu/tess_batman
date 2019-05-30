# AST 520: TESS BATMAN
Welcome to the AST 520 final project! See below for some useful links, and a description of what's in this repository. This is also what you will see when you go to the folder on Monsoon (`/common/contrib/classroom/ast520/tess_batman`).

## Hypothesis
Exoplanets can be identified with similar accuracy and precision as citizen scientists by convolving TESS light curve data with modelled transit curves from BATMAN.

## Set up environment
The packages needed are in the `environment.yml` file. You can install this environment on Monsoon with:

```bash
cd /common/contrib/classroom/ast520/tess_batman
module load anaconda
conda env create -f environment.yml
```

Finally, activate the environment with

```bash
source activate tessbatman 
```

If you need to update or install any packages, activate the environment, run the following command and replace the `environment.yml` with the new one:

```bash
conda env export > environment.yml
```

## Links
Some useful links:

[OverLeaf: Write-up](https://www.overleaf.com/6962119764bvptkvfvfdxb)

[Google Drive: TESS Batman](https://drive.google.com/drive/u/1/folders/1k24tJCjtpcPBKiviM_Z97FA63OkpH_I5)

- [Google Doc: Final Project Overview](https://docs.google.com/document/d/1hIIUlYv_Pa79qHZdYeT0fcedpBuCfnFgZ9BqdtIjUvM)

- [Drive: sample_data](https://drive.google.com/drive/u/1/folders/126cUeBdO68Wx1mygxumo0xYIk0PhWFF-)

[TESS: Bulk download page](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html)

## Structure

```
tess_batman
|-- code
|   |-- downloadTESS
|       | tesscurl_sector_*_lc.sh
|       | tess_download_batch.sh
|   |-- makeSampleData
|       | real_exoplanets.csv
|       | real_exoplanets.py
|-- data
|   | batmanCurves.csv
|   | batmanParams.csv
|   | candidates.csv
|   | confCandidates.csv
|   |-- TESS
|   |   |-- Sector1
|   |   |-- Sector2
|   |   |-- Sector3
|   |   |-- Sector4
|   |   |-- Sector5
|   |   |-- Sector6
|   |   |-- Sector7
|-- sampleData
|   | sample_Sector0
|   |   |-- *.fits
|   | sample_synthTESS
|   |   |-- *.txt
|   | sample_batmanCurves.csv
|   | sample_batmanParams.csv
|   | sample_candidates.csv
| README.md
```

### code/
This is where all of the code for the project will be stored. As teams finish code Christian will update it here (teams are also welcome to open a pull request to add code directly to the repository as well, but this is not necessary).

- `downloadTESS/`: scripts needed to bulk download the TESS lightcurves and the Monsoon script to batch it (see this [link](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html) for the bulk download scripts)
- `makeSampleData/`: scripts used to generate the sample data. Orbital parameters of confirmed exoplanets are stored in `real_exoplanets.csv` and they are used to create sample batman transit curves in `real_exoplanets.py`. See [tess_batman/sampleData/](./sampleData) for generated files.

### data/
This is where the bulk of the data is stored. Because the TESS files are so big, they are excluded from this Git repository, but they will appear here in the `TESS/` folder on Monsoon.

- the .csv files: the files each team will deliver to run the analysis (currently blank)
- `TESS/`: all currently available TESS light curve data (only on Monsoon)

### sampleData/
Sample versions of the .csv files in `tess_batman/data/` are included to provide a template for input and output formats for each team. They also provide a starting point for the SCIENCE team to start investigating real transits. These data are also available on the Google Drive [here](https://drive.google.com/drive/u/1/folders/126cUeBdO68Wx1mygxumo0xYIk0PhWFF-)

- `sample_Sector0`: a subset of TESS lightcurve data stored in `.fits` files. Some of the included lightcurves have had real exoplanet transits detected which correspond to the transits in the following .csv files. Others are random lightcurves pulled from the TESS catalog [here](https://archive.stsci.edu/missions/tess/tid/) (these files begin with `tess*` and were not picked in any particular way, mileage may vary).
- `sample_batmanCurves.csv`: Batman modelled transit curves of 6 real transits observed in TESS data (five of these original light curves are in `sample_Sector0`). Column 0 is the time array and each other column is the amplitude of a batman curve normalized to 1. Rows are the time steps.
- `sample_batmanParams.csv`: Batman parameters used to generate the modelled transit curves in `batmanCurves.csv`. The columns are the curve ID followed by the 9 batman parameters (note **u** is a string with one or more space-separated values, e.g. **"0.1 0.2"** or **"0.3"** etc.). The rows correspond to the generated curves with unique IDs.
- `sample_candidates.csv`: The short list of candidates that the convolve team will generate. Columns are sector (folder of TESS data), tessFile (name of fits file which matched), matching batman curve ID, and the correlation (strength of the fit).
