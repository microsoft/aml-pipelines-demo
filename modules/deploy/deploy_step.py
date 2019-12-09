import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def deploy_step(model_dir, accuracy_file, test_dir, compute_target):
    '''
    This step registers and deploys a new model on its first run. In subsequent runs, it will only register 
    and deploy a new model if the training dataset has changed or the dataset did not change, but the accuracy improved.

    :param model_dir: The reference to the directory containing the trained model
    :type model_dir: DataReference
    :param accuracy_file: The reference to the file containing the evaluation accuracy
    :type accuracy_file: DataReference
    :param test_dir: The reference to the directory containing the testing data
    :type test_dir: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The preprocess step, step outputs dictionary (keys: scoring_url)
    :rtype: PythonScriptStep, dict
    '''

    scoring_url = PipelineData(
        name='scoring_url', 
        pipeline_output_name='scoring_url',
        datastore=accuracy_file.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [scoring_url]
    outputs_map = { 'scoring_url': scoring_url }

    step = PythonScriptStep(
        script_name='deploy.py',
        arguments=[
            '--model_dir', model_dir, 
            '--accuracy_file', accuracy_file, 
            '--test_dir', test_dir, 
            '--scoring_url', scoring_url
        ],
        inputs=[model_dir, accuracy_file, test_dir],
        outputs=outputs,
        compute_target=compute_target,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=False
    )

    return step, outputs_map
 