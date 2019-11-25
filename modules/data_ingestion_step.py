import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData

def data_ingestion_step(root_dir, compute_target):

    run_config = RunConfiguration()
    run_config.environment.environment_variables = {'COGNITIVE_SERVICES_API_KEY': os.environ['COGNITIVE_SERVICES_API_KEY']}
    run_config.environment.docker.enabled = True

    raw_data_dir = PipelineData(
        name='raw_data_dir', 
        datastore=root_dir.datastore,
        output_mode='mount')

    step = PythonScriptStep(
        script_name='data_ingestion.py',
        arguments=['--root_dir', root_dir, '--raw_data_dir', raw_data_dir, '--num_images', 5],
        inputs=[root_dir],
        outputs=[raw_data_dir],
        compute_target=compute_target,
        source_directory='src',
        runconfig=run_config,
        allow_reuse=False
    )

    return step, [raw_data_dir]
