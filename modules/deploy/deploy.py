import argparse
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice, AciWebservice
from azureml.exceptions import WebserviceException

def register_model(model_dir, model_name, accuracy, test_dir, workspace):
    '''
    Registers a new model
    '''
    model = Model.register(
        model_path = model_dir,
        model_name = model_name,
        tags = {
            'accuracy': accuracy, 
            'test_data': test_dir
        },
        description='Object recognition classifier',
        workspace=workspace)
    return model

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

# Define model and service names
service_name = 'object-recognition-service'
model_name = 'object-recognition-pipeline'

# Get run context
run = Run.get_context()
workspace = run.experiment.workspace

# Read accuracy
with open(accuracy_file) as f:
    accuracy = f.read()

# Register model if accuracy is higher or if test dataset has changed
new_model = False
try:
    model = Model(workspace, model_name)
    prev_accuracy = model.tags['accuracy']
    prev_test_dir = model.tags['test_data']
    if prev_test_dir != test_dir or prev_accuracy >= accuracy:
        model = register_model(model_dir, model_name, accuracy, test_dir, workspace)
        new_model = True
except WebserviceException:
    print('Model does not exist yet')
    model = register_model(model_dir, model_name, accuracy, test_dir, workspace)
    new_model = True

# Deploy new webservice if new model was registered
if new_model:
    # Create inference config
    inference_config = InferenceConfig(
        source_directory = '.',
        runtime = 'python', 
        entry_script = 'score.py',
        conda_file = 'env.yml')

    # Deploy model
    aci_config = AciWebservice.deploy_configuration(
        cpu_cores = 2, 
        memory_gb = 4, 
        tags = {'model': 'RESNET', 'method': 'pytorch'}, 
        description='CIFAR-Type object classifier')

    try:
        service = Webservice(workspace, name=service_name)
        if service:
            service.delete()
    except WebserviceException as e:
        print()

    service = Model.deploy(workspace, service_name, [model], inference_config, aci_config)
    service.wait_for_deployment(True)
else:
    service = Webservice(workspace, name=service_name)

# Output scoring url
print(service.scoring_uri)
with open(scoring_url, 'w+') as f:
    f.write(service.scoring_uri)

