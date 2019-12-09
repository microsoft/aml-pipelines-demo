from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.data.data_reference import DataReference
from modules.ingestion.data_ingestion_step import data_ingestion_step
from modules.preprocess.data_preprocess_step import data_preprocess_step
from modules.train.train_step import train_step
from modules.evaluate.evaluate_step import evaluate_step
from modules.deploy.deploy_step import deploy_step
from azureml.core.compute import AmlCompute, ComputeTarget

# Get workspace, datastores, and compute targets
print('Connecting to Workspace ...')
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()

# Create CPU compute target
print('Creating CPU compute target ...')
cpu_cluster_name = 'ds3cluster'
cpu_compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_V2', 
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0, 
                                                           max_nodes=2)
cpu_compute_target = ComputeTarget.create(workspace, cpu_cluster_name, cpu_compute_config)
cpu_compute_target.wait_for_completion(show_output=True)

# Create GPU compute target
print('Creating GPU compute target ...')
gpu_cluster_name = 'k80cluster'
gpu_compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', 
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0, 
                                                           max_nodes=2)
gpu_compute_target = ComputeTarget.create(workspace, gpu_cluster_name, gpu_compute_config)
gpu_compute_target.wait_for_completion(show_output=True)

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
print('Submitting pipeline ...')
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
