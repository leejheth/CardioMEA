# CardioMEA

## Overview

An open-source data pipeline to process, visualize, and analyze cardiomyocytes data recorded by HD-MEAs.

## Key features
* Efficient data processing 
  * The pipelines are coded in modular structure ([Kedro](https://kedro.org/) framework), which makes the pipeline easily understandable and modifiable.
  * The platform allows for parallel computation of recording files (using multiple CPUs) to speed up the processing.
  * The platform contains processing pipelines for both extracellular and intracellular signals obtained by HD-MEAs.
  * Processed data are stored in SQL database, preserving data history. 
* Web-based dashboard for data visualization and analysis 
  * No-code interactive dashboard offering enhanced compatibility with a broad range of users and workstations.
  * The data panel allows users to select processed data from the SQL table for visualization and feature analysis. 
* Feature analysis using automated machine learning
  * The dashboard provides visulization tools to investigate correlations, similarity, and multicollinearity between features.
  * With just a few mouse clicks, users can build the best performing model ([AUTO-SKLEARN](https://automl.github.io/auto-sklearn/master/)) for classification of diseased and healthy control cell lines.
  * Feature importance analysis to estimate predictive power of each feature. 

## CardioMEA Dashboard
![plot](https://github.com/leejheth/CardioMEA/blob/main/docs/dashboard.PNG?raw=true)

## How to setup 

Prerequisite:
- Python 3.9
- GNU Make
- Auto-sklearn is not compatible with Windows OS. To run the analysis in Windows machine, use Windows Subsystems for Linux with Ubuntu. 

Clone the repository to your working directory.
```
git clone git@github.com:leejheth/CardioMEA.git
```

Go to the directory.
```
cd CardioMEA
```

Set up the environment. This will create a new conda environment and install all dependencies in it.
```
make setup
```

Activate the environment.
```
conda activate cardio-env
```

## How to install additional packages in the environment

Add packages, which you need to install, in `src/requirements.in`.

Compile dependencies file.
```
pip-compile src/requirements.in -o src/requirements.txt
```

To install them, run:
```
python -m pip install -r src/requirements.txt
```

## Important note for data security

* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


## How to run pipelines in CardioMEA

### Data processing

Configure your PostgreSQL credentials data in `conf/local/credentials.yml' file (need to create a new file) in the following format:

```
db_credentials:
  con: postgresql://(user):(password)@(host):(port)/(dbname)
```

Also create and add credentials in `conf/local/postgresql.txt' file in the following format: host, dbname, user, password, port in each line.

Configure base_directory of your recording files, file extension, and the number CPUs to use for parallelize processing in `conf/base/parameters.yml`.

List all directories, where recording files are stored, in file `data/01_raw/catalog.csv`. 
The following commmand will create another file `data/01_raw/catalog_full.csv` and list full paths of all recording files found in the specified directories. Refer to `conf/base/catalog.yml`.

```
kedro run -p list_rec_files
```

Run parallelized (or single core) processing by running the following command.

```
python parallel_run.py
```

### Interactive data visualization and analysis

To open the CardioMEA Dashboard, run the following command.

```
kedro run -p dashboard
```

