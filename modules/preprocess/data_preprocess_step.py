import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def data_preprocess_step(raw_data_dir, compute_target):

    run_config = RunConfiguration()
    run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['opencv-python==4.1.1.26', 'imutils==0.5.3'])
    run_config.environment.docker.enabled = True

    image_dim = PipelineParameter(name='image_dim', default_value=200)

    train_data_dir = PipelineData(
        name='train_data_dir', 
        pipeline_output_name='train_data_dir',
        datastore=raw_data_dir.datastore,
        output_mode='mount',
        is_directory=True)

    valid_data_dir = PipelineData(
        name='valid_data_dir', 
        pipeline_output_name='valid_data_dir',
        datastore=raw_data_dir.datastore,
        output_mode='mount',
        is_directory=True)

    test_data_dir = PipelineData(
        name='test_data_dir', 
        pipeline_output_name='test_data_dir',
        datastore=raw_data_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [train_data_dir, valid_data_dir, test_data_dir]

    step = PythonScriptStep(
        script_name='data_preprocess.py',
        arguments=[
            '--raw_data_dir', raw_data_dir, 
            '--train_data_dir', train_data_dir, 
            '--valid_data_dir', valid_data_dir, 
            '--test_data_dir', test_data_dir, 
            '--image_dim', image_dim
        ],
        inputs=[raw_data_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=run_config,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs
