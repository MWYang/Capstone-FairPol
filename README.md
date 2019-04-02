# Introduction

This repository contains the source code, Jupyter Notebooks, and LaTeX files
for producing my Capstone thesis (submitted in March 2019).

You can read my [final thesis](writeups/ThesisFinal.pdf) and
examine the source code in `src`. [This notebook](src/chicago_analysis.ipynb)
contains the main code for running the models and performing the analysis as
well as copies of the graphs.

# Replication

First, install the necessaries Python libraries:
```
cd src/
pip install -r requirements.txt
```
Then, download and clean the necessary data:
```
python download_and_clean_illinois_census_data.py
python download_and_clean_chicago_data.py
```
This step takes around an hour on my machine (MacBook Pro, Retina, 13-inch,
Early 2015), mostly because the Chicago crime data file is pretty big (\~1.5
GB).

Then, you should be able to re-run the analysis in
[chicago_analysis.ipynb](src/chicago_analysis.ipynb) without issues. This will
also take \~2 hours (at least on my machine).

# Code Structure

```
|-- src
    |-- chicago_analysis.ipynb
    |-- download_and_clean_chicago_data.py
    |-- download_and_clean_illinois_census_data.py
    |-- fairpol
    |   |-- assesspol.py
    |   |-- fairpol.py
    |   |-- helpers.py
    |   |-- predpol.py
```

`chicago_analysis.ipynb` is the main program file and produces all of the results and graphs.

`download_and_clean_chicago_data.py` and `download_and_clean_illinois_census_data.py` download the data files for replication. The first gets crime data from the Chicago police department while the second retrieves demographic information for each census tract in Chicago.

The source code for most of the functions is in `fairpol`. With a little bit of polishing, including writing test cases, `fairpol` could be published as its own Python package. As much as possible, the code in this directory attempts to be modular and follow reasonable naming conventions, where the logic for making policing predictions (`predpol.py`), assessing the results of policing prediction (`assesspol.py`), and making predictions fairer (`fairpol.py`) are separated into different files.

# Tasks
- [x] Finish migrating earlier, horrible spaghetti code for assessing fairness, implementing the fairness modification
- [x] Add final thesis document
