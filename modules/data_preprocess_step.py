import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData

def data_preprocess_step(raw_data_dir, compute_target):

    step = PythonScriptStep(
        script_name='data_preprocess.py',
        arguments=['--raw_data_dir', raw_data_dir],
        inputs=[raw_data_dir],
        compute_target=compute_target,
        source_directory='src'
    )

    return step
