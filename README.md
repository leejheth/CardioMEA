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


## Installation

Prerequisites:
- Python 3.9
- GNU Make
- Auto-sklearn, which is one of core modules for feature analysis, is not compatible with Windows OS. To run the analysis in Windows machine, use Windows Subsystems for Linux with Ubuntu ([How-to guide](https://canonical-ubuntu-wsl.readthedocs-hosted.com/en/latest/guides/install-ubuntu-wsl2/)). 

Clone the repository to your working directory.
```
git clone git@github.com:leejheth/CardioMEA.git
```

Go to the directory.
```
cd CardioMEA
```

Set up the environment. This will create a virtual environment and install all dependencies in it.
```
make setup
```

### (Optional) How to install additional packages in the environment

Add packages, which you need to install, in `src/requirements.in`. Then use the following command.

```
make setup_dev
```

## How to run pipelines in CardioMEA

### Data processing

1. Database

CardioMEA processes raw data and store the extracted features to [PostgreSQL](https://www.postgresql.org/) database. Create a file `conf/local/postgresql.txt` and store PostgreSQL credentials in each line as follows.

```
host 
DB-name 
user-name 
password 
port
```

2. Raw data

Specify the directory of your recording files, file extension, and the number of CPUs to use for parallelize processing in `conf/base/parameters.yml`.

List all raw recording files to be processed in `data/01_raw/catalog.csv`. Then run the following command.

```
make create_list
```

3. Run data processing

```
make process
```

### Interactive data visualization and analysis

To open the CardioMEA Dashboard, run the following command.

```
make vis
```

## Important note for data security

* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## Troubleshooting

### Installation
| Issue | Solution |
| ------- |  -------  | 
| The program 'make' is currently not installed. | For simplicity of usage, CardioMEA uses make commands. Please install GNU Make. |    
| Error while installing Auto-sklearn: "Detected unsupported operating system: win32" | Auto-sklearn is not supported in Windows machine. Use UNIX system or WSL. |
| Command 'x86_64-linux-gnu-gcc' failed | Run the following: sudo apt-get install build-essential python3.9-dev|

### Data processing
| Issue | Solution |
| ------- |  -------  | 
|   |    |  

### Dashboard
| Issue | Solution |
| ------- | ------- | 
| Port is in use | Replace the web_port number in "conf/base/parameters/visualize.yml" to another one, such as 8056, 8057, etc. |  
| psycopg2 OperationError "Could not translate host name 'X.X.X' to address" | The database host you specified in 'conf/local/postgresql.txt' could not be found. Double check if the host address is correct. |
