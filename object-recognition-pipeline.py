from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.data.data_reference import DataReference
from modules.ingestion.data_ingestion_step import data_ingestion_step
from modules.preprocess.data_preprocess_step import data_preprocess_step
from modules.train.train_step import train_step
from modules.evaluate.evaluate_step import evaluate_step
from modules.deploy.deploy_step import deploy_step

# Get workspace, datastores, and compute targets
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()
cpu_compute_target = workspace.compute_targets['ds3cluster']
gpu_compute_target = workspace.compute_targets['k80cluster']

# Get datastore reference
datastore = DataReference(datastore, mode='mount')

# Step 1: Data ingestion 
data_ingestion_step, data_ingestion_outputs = data_ingestion_step(datastore, cpu_compute_target)

# Step 2: Data preprocessing 
data_preprocess_step, data_preprocess_outputs = data_preprocess_step(data_ingestion_outputs[0], cpu_compute_target)

# Step 3: Train Model
train_step, train_outputs = train_step(data_preprocess_outputs[0], data_preprocess_outputs[1], gpu_compute_target)

# Step 4: Evaluate Model
evaluate_step, evaluate_outputs = evaluate_step(train_outputs[0], data_preprocess_outputs[2], gpu_compute_target)

# Step 5: Deploy Model
deploy_step, deploy_outputs = deploy_step(train_outputs[0], evaluate_outputs[0], data_preprocess_outputs[2], cpu_compute_target)

# Submit pipeline
pipeline_parameters = {
    'num_images': 100,
    'image_dim': 200,
    'num_epochs': 10, 
    'batch_size': 16,
    'learning_rate': 0.001, 
    'momentum': 0.9
}
pipeline = Pipeline(workspace=workspace, steps=[data_ingestion_step, data_preprocess_step, train_step, evaluate_step, deploy_step])
pipeline_run = Experiment(workspace, 'object-recognition-pipeline').submit(pipeline, pipeline_parameters=pipeline_parameters)
