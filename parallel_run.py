from joblib import Parallel, delayed
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner
from pathlib import Path
import datetime
import yaml

bootstrap_project(Path.cwd())

with open("conf/base/file_count.yml", "r") as f:
    content = yaml.safe_load(f)
n_files = content['n_files']
nCPUs = content['n_CPUs']

def run_pipeline(index):    
    session = KedroSession.create(extra_params={"file_index": index})
    session.run(pipeline_name="single_pipeline", runner=SequentialRunner())

timestamp = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f"Starting parallel processing of pipelines at {timestamp}.")

Parallel(n_jobs=nCPUs, backend='multiprocessing')(delayed(run_pipeline)(i) for i in range(n_files))

timestamp = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f"Finished parallel processing of pipelines at {timestamp}.")