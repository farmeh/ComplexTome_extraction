import os
import sys
import argparse
import subprocess

def external_eval_all_entities(pred_folder, program_halt):
    # <<<CRITICAL>>>: we should use the same command for all experiments
    cwd = os.path.dirname(os.path.realpath(__file__))
    execution_folder = os.environ['CT_REL_FOLDERPATH'] or cwd
    if execution_folder[-1] != "/":
        execution_folder += "/"
    devel_gold_folder = execution_folder + "splits/dev-set/" #space is really important at the end

    command = "python3 evalsorel.py --entities Protein --relations Complex_formation " + devel_gold_folder + " " + pred_folder

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, shell=True)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8').strip()
    stderr = stderr.decode('utf-8').strip()

    f_score = "COULD NOT CALCULATE"
    try:
        f_score = float(stdout.split(" F ")[-1].split("%")[0])
    except Exception as E:
        err_msg = "error in processing external evaluator output:\n"
        err_msg += "cmd: " +  command + "\n"
        err_msg += "stdout: " + stdout + "\n"
        err_msg += "stderr: " + stderr + "\n"
        program_halt(err_msg)
    return f_score, str({"OUT": stdout, "ERR": stderr})

def validate_parameter(param_val, param_name, range_min=None, range_max=None, map_zero_to_none=True):
    if range_min is not None:
        if param_val < range_min:
            print (param_name + " should be bigger than " + range_min)
            sys.exit(-1)
    if range_max is not None:
        if param_val > range_max:
            print (param_name + " should be smaller than " + range_max)
            sys.exit(-1)
    if map_zero_to_none:
        if param_val == 0:
            param_val = None
    return param_val

if __name__ == "__main__":
    random_seed_list = [42, 2022, 3240, 13, 8883]
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed_index"            , required=True, type=int , choices=[0,1,2,3,4])
    parser.add_argument("--model_address"                , required=True, type=str)
    parser.add_argument("--train_set_address"            , required=True, type=str)
    parser.add_argument("--devel_set_address"            , required=True, type=str)
    parser.add_argument("--preds_model_output_address"   , required=True, type=str)
    parser.add_argument("--logfile_address"              , required=True, type=str)
    args = parser.parse_args()

    # 0: turn off warning on cluster
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  #<<<CRITICAL>>> to turn off the warning

    # 1: set main random seed
    PARAM_random_seed = random_seed_list[args.random_seed_index]
    os.environ['PYTHONHASHSEED'] = str(PARAM_random_seed)

    # 2: set/check folders
    PARAM_model_folderpath = args.model_address
    PARAM_train_folderpath = args.train_set_address
    PARAM_devel_folderpath = args.devel_set_address
    PARAM_preds_model_output_address = args.preds_model_output_address

    if not os.path.isdir(PARAM_model_folderpath):
        print ("invalid path for model_address : " , PARAM_model_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_train_folderpath):
        print ("invalid path for train_set_address : " , PARAM_train_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_devel_folderpath):
        print ("invalid path for devel_set_address : " , PARAM_devel_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_preds_model_output_address):
        print ("invalid path for preds_model_output_address : " , PARAM_preds_model_output_address)
        print ("exiting ...")
        sys.exit(-1)

    # 3: delete log file if exists
    if os.path.isfile(args.logfile_address):
        print("deleting previously existed log_file and recreating: " + args.logfile_address)
        os.remove(args.logfile_address)

    # 4: set paths for import folders
    current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("/".join(current_file_path.split("/")[:-1]))
    sys.path.append("/".join(current_file_path.split("/")[:-2]))

    # 5: choose the backend to run
    from helpers import pipeline_variables
    PARAM_backend = pipeline_variables.BACKEND_Type.HUGGINGFACE_TENSORFLOW #TF2.x + a Huggingface model

    #6: read and set parameters
    PARAM_representation_strategy = pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING

    training_arguments = {
        "optimizer"        : "adam",
        "learning_rate"    : float("3e-6"),
        "num_train_epochs" : 11,
        "minibatch_size"   : 5,
        "train_fit_verbose": 0,
        "use_sample_weights"    : False,
        "sample_weight_strategy": None,
    }
    model_params = {
        "bertdr": None, #dropout_after_bert,
        "dod"   : None, #dense_dim_after_bert,
        "drad"  : None, #droupout_after_dense,
    }
    all_params = {
        "cmd_line": args._get_kwargs(),
        "training": [x for x in training_arguments.items()],
        "model"   : [x for x in model_params.items()]}

     #7: initialize project and relation extraction objects based on the selected backend
    import project
    import relation_extraction_pipeline
    prj = project.Project(args.logfile_address , "ComplexTome_configs.json")
    prj.lp ("PARAMS:\t" + str(all_params))
    pipeline = relation_extraction_pipeline.RelationExtractionPipeline(
        project= prj,
        pretrained_model_name_or_path=PARAM_model_folderpath,
        max_seq_len=128,
        backend=PARAM_backend,
        random_seed=PARAM_random_seed,
        external_evaluator=external_eval_all_entities,
        predict_devel=True,
        evaluate_devel=True,
        writeback_devel_preds=False,
        writeback_devel_preds_folder=PARAM_preds_model_output_address,
        representation_strategy=PARAM_representation_strategy,
        save_best_model_folder_path=PARAM_preds_model_output_address,
    )

    # 6: add training and devel files -----------------------------------------------
    from helpers import general_helpers
    pipeline.add_train_dev_test_brat_folders(PARAM_train_folderpath, PARAM_devel_folderpath)

    # 7: run pipeline
    pipeline.run_pipeline(training_arguments, model_params)

    # 8: safe exit
    prj.exit()