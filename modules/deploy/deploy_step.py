import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def deploy_step(model_dir, accuracy_file, test_dir, compute_target):

    scoring_url = PipelineData(
        name='scoring_url', 
        pipeline_output_name='scoring_url',
        datastore=accuracy_file.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [scoring_url]

    step = PythonScriptStep(
        script_name='deploy.py',
        arguments=[
            '--model_dir', model_dir, 
            '--accuracy_file', accuracy_file, 
            '--test_dir', test_dir, 
        ],
        inputs=[model_dir, accuracy_file, test_dir],
        outputs=outputs,
        compute_target=compute_target,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs
 