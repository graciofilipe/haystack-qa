import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.experimental.custom_job import utils
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component

from training_haystack.training_component import training_comp
from load_and_test.test_component import test_comp

@dsl.pipeline(name="haystack-training")
def pipeline():
    training_op = training_comp()
    test_op = test_comp(
    bucket_name ="filipegracio-haystack",
    artifact_path=training_op.outputs["artifact_path"],
    index_faiss_file_path=training_op.outputs["index_faiss_file_path"],
    index_json_file_path=training_op.outputs["index_json_file_path"],
    document_db_path=training_op.outputs["document_db_path"],
    pipeline_yaml_path=training_op.outputs["pipeline_yaml_path"]
    )



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