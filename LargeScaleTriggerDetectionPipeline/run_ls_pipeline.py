import os
import sys
import argparse

current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))

import large_scale_explanation_pipeline

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_file_path" , required=True, type=str)
    parser.add_argument("--pretrained_model_path" , required=True, type=str)
    parser.add_argument("--log_file_path" , required=True, type=str)
    parser.add_argument("--input_folder_path" , required=True, type=str)
    parser.add_argument("--output_folder_path" , required=True, type=str)
    args = parser.parse_args()

    work_only_on_positive_relations_in_input_ann = True
    create_output_ann_files = False

    mypipeline = large_scale_explanation_pipeline.LargeScaleExplanationPipeline_torch(
        configs_file_path = args.configs_file_path,
        pretrained_model_path = args.pretrained_model_path,
        log_file_path = args.log_file_path,
        input_folder_path = args.input_folder_path,
        output_folder_path = args.output_folder_path,
        work_only_on_positive_relations_in_input_ann=work_only_on_positive_relations_in_input_ann,  # True: look for valid relations, False: generate all possible pairs between all entities.
        create_output_ann_files=create_output_ann_files)

    mypipeline.run_large_scale_pipeline()
