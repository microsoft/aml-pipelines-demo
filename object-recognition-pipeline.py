from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.data.data_reference import DataReference
from modules.data_ingestion_step import data_ingestion_step
from modules.data_preprocess_step import data_preprocess_step

# Get workspace, datastores, and compute targets
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()
cpu_compute_target = workspace.compute_targets['ds3cluster']

# Get datastore root directory reference
datastore_root_dir = DataReference(datastore,
                                   data_reference_name='datastore_root_dir', 
                                   path_on_datastore='object_recognition_data', 
                                   mode='mount')

# Step 1: Data ingestion step
data_ingestion_step, data_ingestion_outputs = data_ingestion_step(datastore_root_dir, cpu_compute_target)

# Step 2: Data preprocessing step
data_preprocess_step = data_preprocess_step(data_ingestion_outputs[0], cpu_compute_target)

pipeline = Pipeline(workspace=workspace, steps=[data_ingestion_step, data_preprocess_step])
pipeline_run = Experiment(workspace, 'object-recognition-pipeline').submit(pipeline)

#os.path.dirname(os.path.abspath(__file__))