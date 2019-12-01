import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import EstimatorStep
from azureml.train.dnn import PyTorch

def train_step(train_dir, valid_dir, compute_target):

    num_epochs = PipelineParameter(name='num_epochs', default_value=25)
    batch_size = PipelineParameter(name='batch_size', default_value=16)
    learning_rate = PipelineParameter(name='learning_rate', default_value=0.001)
    momentum = PipelineParameter(name='momentum', default_value=0.9)

    output_dir = PipelineData(
        name='output_dir', 
        pipeline_output_name='output_dir',
        datastore=train_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [output_dir]

    estimator = PyTorch(
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        entry_script='train.py',
        framework_version='1.3',
        compute_target=compute_target,
        use_gpu=True)

    step = EstimatorStep(
        estimator=estimator,
        estimator_entry_script_arguments=[
            '--train_dir', train_dir, 
            '--valid_dir', valid_dir, 
            '--output_dir', output_dir, 
            '--num_epochs', num_epochs, 
            '--batch_size', batch_size,
            '--learning_rate', learning_rate, 
            '--momentum', momentum
        ],
        inputs=[train_dir, valid_dir],
        compute_target=compute_target,
        outputs=outputs,
        allow_reuse=False)

    return step, outputs
