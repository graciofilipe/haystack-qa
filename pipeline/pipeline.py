import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.experimental.custom_job import utils
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component

from training_haystack.training_component import training_comp

@dsl.pipeline(name="haystack-training")
def pipeline():
    training_op = training_comp()




compiler = compiler.Compiler()

compiler.compile(
pipeline_func=pipeline, 
package_path="training_spec.json"
)


job = aip.PipelineJob(
    display_name="train-haystack",
    template_path="training_spec.json",
    pipeline_root="gs://filipegracio-haystack/pipeline",
)

job.run()