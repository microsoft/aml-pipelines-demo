from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.data.data_reference import DataReference
from modules.ingestion.data_ingestion_step import data_ingestion_step
from modules.preprocess.data_preprocess_step import data_preprocess_step

# Get workspace, datastores, and compute targets
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()
cpu_compute_target = workspace.compute_targets['ds3cluster']

# Get datastore reference
datastore = DataReference(datastore, mode='mount')

# Step 1: Data ingestion step
data_ingestion_step, data_ingestion_outputs = data_ingestion_step(datastore, cpu_compute_target)

# Step 2: Data preprocessing step
data_preprocess_step = data_preprocess_step(data_ingestion_outputs[0], cpu_compute_target)

# Submit pipeline
pipeline = Pipeline(workspace=workspace, steps=[data_ingestion_step, data_preprocess_step])
pipeline_run = Experiment(workspace, 'object-recognition-pipeline').submit(pipeline)
