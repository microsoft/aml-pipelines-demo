import argparse
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice, AciWebservice
from azureml.exceptions import WebserviceException

# Define arguments
parser = argparse.ArgumentParser(description='Deploy arg parser')
parser.add_argument('--test_dir', type=str, help='Directory where testing data is stored')
parser.add_argument('--model_dir', type=str, help='Directory where model is stored')
parser.add_argument('--accuracy_file', type=str, help='File storing the evaluation accuracy')
parser.add_argument('--scoring_url', type=str, help='File storing the scoring url')
args = parser.parse_args()

# Get arguments from parser
test_dir = args.test_dir
model_dir = args.model_dir
accuracy_file = args.accuracy_file
scoring_url = args.scoring_url

# Get run context
run = Run.get_context()
workspace = run.experiment.workspace

# Read accuracy
with open(accuracy_file) as f:
    accuracy = f.read()

# Register model
model = Model.register(
    model_path = model_dir,
    model_name = 'object-recognition-pipeline',
    tags = {
        'accuracy': accuracy, 
        'test_data': test_dir
    },
    description='Object recognition classifier',
    workspace=workspace)

# Create inference config
inference_config = InferenceConfig(
    source_directory = '.',
    runtime = 'python', 
    entry_script = 'score.py',
    conda_file = 'env.yml')

# Deploy model
service_name = 'object-recognition-service'
aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 2, 
    memory_gb = 4, 
    tags = {'model': 'RESNET', 'method': 'pytorch'}, 
    description='CIFAR-Type object classifier')

try:
    service = Webservice(workspace, name=service_name)
    if service_name:
        service.delete()
except WebserviceException as e:
    print()

service = Model.deploy(workspace, service_name, [model], inference_config, aci_config)
service.wait_for_deployment(True)

# Output scoring url
print(service.scoring_uri)
with open(scoring_url, 'w+') as f:
    f.write(aci_service.scoring_uri)

