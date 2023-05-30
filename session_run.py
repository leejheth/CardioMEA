from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import ParallelRunner
from pathlib import Path

bootstrap_project(Path.cwd())
with KedroSession.create() as session:
    session.run(pipeline_name="auto_pipeline", runner=ParallelRunner(max_workers=10))