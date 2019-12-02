import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def evaluate_step(model_dir, test_dir, compute_target):

    run_config = RunConfiguration()
    run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['torch==1.3.1'])
    run_config.environment.docker.enabled = True

    accuracy_file = PipelineData(
        name='accuracy_file', 
        pipeline_output_name='accuracy_file',
        datastore=test_dir.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [accuracy_file]

    step = PythonScriptStep(
        script_name='evaluate.py',
        arguments=[
            '--test_dir', test_dir, 
            '--model_dir', model_dir, 
            '--accuracy_file', accuracy_file
        ],
        inputs=[model_dir, test_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=run_config,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs
