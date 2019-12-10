import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def data_ingestion_step(datastore_reference, compute_target):
    '''
    This step will leverage Azure Cognitive Services to search the web for images 
    to create a dataset. This replicates the real-world scenario of data being 
    ingested from a constantly changing source. The same 10 classes in the CIFAR-10 dataset 
    will be used (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 

    :param datastore_reference: The reference to the datastore that will be used
    :type datastore_reference: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The ingestion step, step outputs dictionary (keys: raw_data_dir)
    :rtype: PythonScriptStep, dict
    '''

    run_config = RunConfiguration()
    run_config.environment.environment_variables = {'COGNITIVE_SERVICES_API_KEY': os.environ['COGNITIVE_SERVICES_API_KEY']}
    run_config.environment.docker.enabled = True

    num_images = PipelineParameter(name='num_images', default_value=25)

    raw_data_dir = PipelineData(
        name='raw_data_dir', 
        pipeline_output_name='raw_data_dir',
        datastore=datastore_reference.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [raw_data_dir]
    outputs_map = { 'raw_data_dir': raw_data_dir }

    step = PythonScriptStep(
        script_name='data_ingestion.py',
        arguments=['--output_dir', raw_data_dir, '--num_images', num_images],
        inputs=[datastore_reference],
        outputs=outputs,
        compute_target=compute_target,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        runconfig=run_config,
        allow_reuse=True
    )

    return step, outputs_map
