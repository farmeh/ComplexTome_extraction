import os
import json
import copy
from collections import OrderedDict
import shutil

from helpers import pipeline_variables


class RelationExtractionPipeline(object):
    def __init__(self,
                 project,
                 pretrained_model_name_or_path,
                 max_seq_len,
                 backend=pipeline_variables.BACKEND_Type.HUGGINGFACE_TENSORFLOW,
                 random_seed=112,
                 evaluation_metric="positive_f1_score",
                 external_evaluator=None,
                 predict_devel=True,
                 evaluate_devel=True,
                 writeback_devel_preds=False,
                 writeback_devel_preds_folder=None,
                 process_devel_after_epoch=0,
                 representation_strategy=pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES,
                 negative_downsampling_rate=0,
                 augment_predictions_with_regulation=False,
                 save_best_model_folder_path=None):

        # set random seeds
        import numpy as np
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
        import tensorflow as tf
        try:
            tf.set_random_seed(random_seed)
        except:
            try:
                tf.random.set_seed(random_seed)
            except:
                pass

        # init params
        self.lp = project.lp
        self.program_halt = project.program_halt
        self.configs = copy.deepcopy(project.configs)
        self.__evaluation_metric = evaluation_metric
        self.__external_evaluator = external_evaluator

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.__loading_fully_pretrained_relation_extraction_model_is_requested = False

        if max_seq_len is None:
            self.__loading_fully_pretrained_relation_extraction_model_is_requested = True

        self.max_seq_len = max_seq_len

        self.__predict_devel = predict_devel
        self.__evaluate_devel = evaluate_devel
        self.__writeback_devel_preds = writeback_devel_preds
        self.__writeback_devel_preds_folder = writeback_devel_preds_folder
        self.__process_devel_after_epoch = process_devel_after_epoch
        self.__representation_strategy = representation_strategy
        self.__backend = backend
        self.__negative_downsampling_rate = negative_downsampling_rate
        self.__augment_predictions_with_regulation = augment_predictions_with_regulation
        self.__save_best_model_folder_path = save_best_model_folder_path

        # log params:
        msg_params = ["Running pipeline:", "-" * 80,
                      "\t- backend                             : " + str(backend),
                      "\t- random_seed                         : " + str(random_seed),
                      "\t- pretrained_model_name_or_path       : " + str(pretrained_model_name_or_path),
                      "\t- max_seq_len                         : " + str(max_seq_len),
                      "\t- representation_strategy             : " + str(representation_strategy),
                      "\t- evaluation_metric                   : " + str(evaluation_metric),
                      "\t- external_evaluator                  : " + str(external_evaluator),
                      "\t- predict_devel                       : " + str(predict_devel),
                      "\t- evaluate_devel                      : " + str(evaluate_devel),
                      "\t- writeback_devel_preds               : " + str(writeback_devel_preds),
                      "\t- writeback_devel_preds_folder        : " + str(writeback_devel_preds_folder),
                      "\t- process_devel_after_epoch           : " + str(process_devel_after_epoch),
                      "\t- negative_downsampling_rate          : " + str(negative_downsampling_rate),
                      "\t- augment_predictions_with_regulation : " + str(augment_predictions_with_regulation),
                      "\t- save_best_model_folder_path         : " + str(save_best_model_folder_path),
                      "-" * 80]
        self.lp(msg_params)

        # evaluate params
        self.__evaluate_init_params()

        if self.__loading_fully_pretrained_relation_extraction_model_is_requested:
            self.__load_best_model_from_folder()
        else:
            # init needed helpers
            if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
                from helpers import bert_helper
                from helpers import architecture_builder
                from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator
                self.__bert_helper = bert_helper.BERT_helper(self.lp, self.program_halt, pretrained_model_name_or_path, max_seq_len)
                self.__ANN_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt, self.configs, self.__bert_helper)
                self.__architecture_builder = architecture_builder.Architecture_Builder(self.lp, self.program_halt, self.configs, self.__bert_helper)
            else:
                from helpers import tf_model_helper
                from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator
                self.__bert_helper = tf_model_helper.TF_helper(self.lp, self.program_halt, self.configs, pretrained_model_name_or_path, max_seq_len=max_seq_len)
                self.__ANN_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt, self.configs, self.__bert_helper)

                # init variables
                self.training_set = {"items": [], "x": None, "y_true": None, "pair_tracking": None}  # each item is a json example file (input/output file name and json content), whereas x, y_true, pair_tracking are created by ann_io_generation
                self.development_set = {"items": [], "x": None, "y_true": None, "pair_tracking": None}  # each item is a json example file (input/output file name and json content), whereas x, y_true, pair_tracking are created by ann_io_generation
                self.model = None
                self.__best_model_weights = None
                self.__all_epochs_evaluation_results = []
                self.__epoch_counter = 0

                if self.__augment_predictions_with_regulation:
                    self.lp("[WARNING]: augmenting predictions with regulation is activated!")
                    from helpers import preds_regulation_augmentation
                    self.__pred_conf_modifier = preds_regulation_augmentation.augment_pred_confs_with_regulation
                else:
                    self.__pred_conf_modifier = None

    def __evaluate_init_params(self):
        def chack_is_instance(param_val, param_name, tp):
            if not isinstance(param_val, tp):
                self.program_halt("invalid type for argument " + param_name + " . Type should be " + str(tp))

        chack_is_instance(self.__predict_devel, "predict_devel", bool)
        chack_is_instance(self.__evaluate_devel, "evaluate_devel", bool)
        chack_is_instance(self.__writeback_devel_preds, "writeback_devel_preds", bool)
        chack_is_instance(self.__process_devel_after_epoch, "process_devel_after_epoch", int)

        if (self.__evaluate_devel or self.__writeback_devel_preds) and (not self.__predict_devel):
            self.program_halt("evaluation and/or write_back are requested while predict_devel is set to False.")

        if ((self.__external_evaluator is not None) or self.__writeback_devel_preds) and (self.__writeback_devel_preds_folder is None):
            self.program_halt("external_evaluator is given and/or writeback_devel_preds is True, but no folder has been defined for writeback_devel_preds_folder")

        if self.__writeback_devel_preds_folder is not None:
            if not isinstance(self.__writeback_devel_preds_folder, str):
                self.program_halt("writeback_devel_preds_folder should be either None or a valid folder path.")
            if not os.path.isdir(self.__writeback_devel_preds_folder):
                self.program_halt("invalid folder address for writeback_devel_preds_folder: " + self.__writeback_devel_preds_folder)
            if len(os.listdir(self.__writeback_devel_preds_folder)) > 1:
                self.program_halt("writeback_devel_preds_folder is not empty: " + self.__writeback_devel_preds_folder)
            if self.__writeback_devel_preds_folder[-1] != "/":
                self.__writeback_devel_preds_folder += "/"

        if self.__save_best_model_folder_path is not None:
            if not isinstance(self.__save_best_model_folder_path, str):
                self.program_halt("save_best_model_folder_path should be either None or a valid folder path.")
            if not os.path.isdir(self.__save_best_model_folder_path):
                self.program_halt("invalid folder address for save_best_model_folder_path: " + self.__save_best_model_folder_path)
            if len(os.listdir(self.__save_best_model_folder_path)) > 1:
                self.program_halt("save_best_model_folder_path folder is not empty: " + self.__save_best_model_folder_path)
            if self.__save_best_model_folder_path[-1] != "/":
                self.__save_best_model_folder_path += "/"

    def add_training_or_validation_files(self, list_of_json_files, train_or_devel):
        # reading json EXAMPLE files (which has pairs sections in the json content, and adding them as items to train or devel set.
        if train_or_devel not in (1, 2):
            self.program_halt("train_or_devel should be either 1 for training_file or 2 for devel file.")
        if train_or_devel == 1:
            self.lp("Adding files to training set...")
        elif train_or_devel == 2:
            self.lp("Adding files to development set ...")

        # loading json EXAMPLE files ...
        for json_file_address in list_of_json_files:
            try:
                with open(json_file_address, "r", encoding='utf-8') as jsonfile:
                    json_data = json.load(jsonfile)

                item = {
                    "input_json_file_address": json_file_address,
                    "output_brat_file_name": None,
                    "json_data": json_data,
                }

                if train_or_devel == 1:  # since we don't evaluate training files with an external evaluator, we don't need to create output files for the training set files.
                    self.training_set['items'].append(item)

                elif train_or_devel == 2:
                    # should i create corresponding prediction output files for devel predictions?
                    if self.__writeback_devel_preds or (self.__external_evaluator is not None):
                        brat_file_name = json_file_address.split("/")[-1].split(".json")[0] + ".ann"
                        item["output_brat_file_name"] = brat_file_name
                    self.development_set['items'].append(item)

            except Exception as E:
                err_msg = "Error loading json file: " + json_file_address + "\nError: " + str(E)
                self.program_halt(err_msg)
        self.lp("done.")

    def __assign_network_inputs_outputs(self, train_or_devel):
        # runs ann_io_generator for the training or devel set, and finds and assigns the input(s) and output(s) for the neural network.
        if train_or_devel not in (1, 2):
            self.program_halt("train_or_devel should be either 1 for training_file or 2 for devel file.")
        if train_or_devel == 1:
            selected_set = self.training_set
        elif train_or_devel == 2:
            selected_set = self.development_set

        if len(selected_set['items']) < 1:
            self.program_halt("selected set has no items yet. Aborting.")

        import numpy as np
        x, y_true = [], []  # vertically stacked outputs of the ann_io_generator for each json EXAMPLE file.
        all_ann_inputs = []
        all_ann_outputs = []
        all_pair_tracking = []

        for item in selected_set['items']:
            json_data = item['json_data']
            output_brat_file_name = item['output_brat_file_name']  # This can be None. For example, training set files don't have any output file because we don't run any external evaluator for the training set files.
            _pair_tracking, ann_inputs, ann_outputs = self.__ANN_input_output_generator.generate_ANN_Input_Outputs_pairs(json_data, generate_output=True, strategy=self.__representation_strategy, print_results=False)
            if _pair_tracking is None:  # <<<CRITICAL>>>: This means that the json example file does not have any pairs at all, or the ann_io_generator could not generate ann_input/ann_output for any pairs maybe because of max_seq_len
                continue
            _pair_tracking = [[output_brat_file_name] + list(pt) for pt in _pair_tracking]
            all_ann_inputs.append(ann_inputs)  # the input for a json example file is a list like [x1, x2, x3], hence we append to preserve the data structure
            all_ann_outputs.append(ann_outputs)  # the output for a json example file is a like like [y1, y2], hence we append to preserve the data structure
            all_pair_tracking.extend(_pair_tracking)  # the pair_tracking is a simple list, hence we extend.

        for i in range(len(all_ann_inputs[0])):  # look at the first item in the list which is all ANN_INPUT for the first file to know how many inputs (e.g. x1, x2, x3) the network has ...
            x.append(np.vstack([item[i] for item in all_ann_inputs]))
        for i in range(len(all_ann_outputs[0])):
            y_true.append(np.vstack([item[i] for item in all_ann_outputs]))

        selected_set['x'] = x
        selected_set['y_true'] = y_true
        selected_set['pair_tracking'] = all_pair_tracking

    def __evaluate_training_params(self, training_params):
        if "optimizer" not in training_params:
            self.program_halt("optimizer is missing in training_params")
        if training_params["optimizer"].lower() not in ["adam", "nadam", "adamwarmup", "sgd"]:
            self.program_halt("optimizer should be either of adam , nadam, adamwarmup or sgd")
        if "learning_rate" not in training_params:
            self.program_halt("learning_rate is missing in training_params")
        if "train_fit_verbose" not in training_params:
            self.program_halt("train_fit_verbose is missing in training_params")
        if "minibatch_size" not in training_params:
            self.program_halt("minibatch_size is missing in training_params")
        if "num_train_epochs" not in training_params:
            self.program_halt("num_train_epochs is missing in training_params")

        # use_sample_weights
        if "use_sample_weights" not in training_params:
            self.program_halt("use_sample_weights is missing in training_params")
        if training_params['use_sample_weights'] not in (True, False):
            self.program_halt('use_sample_weights should be eighter True or False')

        # sample_weight_strategy
        if training_params['use_sample_weights']:
            if not "sample_weight_strategy" in training_params:
                self.program_halt("sample_weight_strategy is missing in training_params")
            else:
                if (not isinstance(training_params['sample_weight_strategy'], str)) and (not isinstance(training_params['sample_weight_strategy'], list)):
                    self.program_halt("sample_weight_strategy should be either 'balanced' or a list like [1,6] for negative/positive.")
                else:
                    self.lp("sample_weight_strategy : " + str(training_params['sample_weight_strategy']))

                if isinstance(training_params['sample_weight_strategy'], str):
                    if not training_params['sample_weight_strategy'].lower() == "balanced":
                        self.program_halt("sample_weight_strategy should be either 'balanced' or a list like [1,6] for negative/positive.")

                if isinstance(training_params['sample_weight_strategy'], list):
                    if not len(training_params['sample_weight_strategy']) == 2:
                        self.program_halt("sample_weight_strategy is list, hence its length should be 2, like [1,6] for negative/positive.")

    def run_pipeline(self, training_params, model_params):
        if self.__loading_fully_pretrained_relation_extraction_model_is_requested:
            self.lp ("you had requested to load a fully pretrained relation extraction from a folder. That model cannot be further retrained. Exiting.")
            return

        import numpy as np
        # check training and devel sets not to be empty ...
        if len(self.training_set['items']) < 1:
            self.program_halt("you have to first give training set files. It is empty now.")
        if len(self.development_set['items']) < 1:
            self.program_halt("you have to first give development set files. It is empty now.")

        # reset the counter
        self.__epoch_counter = 0

        # evaluate and get training_params
        self.__evaluate_training_params(training_params)
        selected_optimizer = training_params["optimizer"].lower()
        selected_learning_rate = training_params["learning_rate"]
        selected_minibatch_size = training_params["minibatch_size"]
        selected_num_train_epochs = training_params["num_train_epochs"]
        selected_fit_verbose = training_params["train_fit_verbose"]

        if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
            should_use_sample_weights = training_params["use_sample_weights"]
        else:
            self.lp("HF backend is selected, hence no sample_weight_strategy is going to be used. Not implemented yet.")
            should_use_sample_weights = False

        if should_use_sample_weights:
            selected_sample_weight_strategy = training_params["sample_weight_strategy"]

        msg = ["---- training params -----"]
        for i, j in sorted(training_params.items(), key=lambda x: x[0]):
            msg += [i + "\t" + str(j)]
        self.lp(msg)
        msg = ["---- model hyper params --------"]
        for i, j in sorted(model_params.items(), key=lambda x: x[0]):
            msg += [i + "\t" + str(j)]
        self.lp(msg)

        # run ANN input/output generation and assemble all matrices into one
        self.__assign_network_inputs_outputs(train_or_devel=1)
        self.__assign_network_inputs_outputs(train_or_devel=2)
        train_set_x = self.training_set['x']
        train_set_y_true = self.training_set['y_true']

        try:
            train_dim_res_msg = ["=" * 80]
            if isinstance(train_set_x, list) or isinstance(train_set_x, tuple):
                dim_msg_x = "Training set X : " + str([str(x.shape) for x in train_set_x])
                train_dim_res_msg.append(dim_msg_x)
            if isinstance(train_set_y_true, list) or isinstance(train_set_y_true, tuple):
                dim_msg_x = "Training set Y : " + str([str(x.shape) for x in train_set_y_true])
                train_dim_res_msg.append(dim_msg_x)
            train_dim_res_msg.append("=" * 80)
            self.lp(train_dim_res_msg)
        except Exception as E:
            pass

        if self.__negative_downsampling_rate > 0:
            from helpers import negative_downsampler
            negative_sp = negative_downsampler.Negative_Downsampler(self.lp, self.program_halt, self.configs, self.__negative_downsampling_rate, self.training_set)
            train_set_x, train_set_y_true = negative_sp.get_new_sample()

        # build model
        if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
            self.lp("building model ...")
            self.model = self.__architecture_builder.generate_BERT_pair_prediction(model_params)

            # initialize optimizer with based on the selected optimizer
            from keras.optimizers import Adam, Nadam, SGD
            if selected_optimizer == "adam":
                my_optimizer = Adam(lr=selected_learning_rate)
            elif selected_optimizer == "nadam":
                my_optimizer = Nadam(lr=selected_learning_rate)
            elif selected_optimizer == "sgd":
                my_optimizer = SGD(lr=selected_learning_rate)
            elif selected_optimizer == "adamwarmup":
                from keras_bert import AdamWarmup, calc_train_steps
                param_total_training_example_count = len(train_set_x[0])  # remember there can be multiple items in x: first one is bert tokens
                total_steps, warmup_steps = calc_train_steps(
                    num_example=param_total_training_example_count,
                    batch_size=selected_minibatch_size,
                    epochs=selected_num_train_epochs,
                    warmup_proportion=0.1,
                )
                my_optimizer = AdamWarmup(total_steps, warmup_steps, lr=selected_learning_rate)

            # which loss-function to use based on type of the task ...
            if not self.configs['classification_type'] in ['binary', 'multi-class', 'multi-label']:
                self.program_halt("invalid classification type in the config file: " + str(self.configs['classification_type']))

            if self.configs['classification_type'] in ['binary', 'multi-class']:
                loss_function_name = "categorical_crossentropy"
            else:
                loss_function_name = "binary_crossentropy"  # multi-label

            # compile model
            self.lp("compiling model ...")
            self.model.compile(loss=loss_function_name, optimizer=my_optimizer)

            # sample-weights (to balance class distributions)
            if should_use_sample_weights:
                # TODO: multi-class and multi-label ...
                if self.configs['classification_type'] in ['multi-class', 'multi-label']:
                    self.program_halt('sample/class weighting for multi-class and multi-label is requested. but they are not implemented yet.')

                import numpy as np
                if isinstance(selected_sample_weight_strategy, str):
                    import numpy as np
                    from sklearn.utils.class_weight import compute_sample_weight
                    y_labels_index = [np.argmax(y) for y in train_set_y_true[0]]
                    y_labels_sample_weights = compute_sample_weight(class_weight='balanced', y=y_labels_index)
                    self.lp('using sample weights based on scikit balanced approach ...')
                else:
                    # negative positive sample weights
                    assert isinstance(selected_sample_weight_strategy, list)
                    assert len(selected_sample_weight_strategy) == 2
                    negative_weight = selected_sample_weight_strategy[0]
                    positive_weight = selected_sample_weight_strategy[1]
                    y_labels_index = [np.argmax(y) for y in train_set_y_true[0]]
                    y_labels_sample_weights = []
                    for y in y_labels_index:
                        if y == 0:
                            y_labels_sample_weights.append(negative_weight)  # add weight for negatives
                        else:
                            y_labels_sample_weights.append(positive_weight)  # add weight for positives (min,max, abs)
                    y_labels_sample_weights = np.array(y_labels_sample_weights)
                    self.lp('using sample weights based on provided class weights:' + str(selected_sample_weight_strategy))
            else:
                y_labels_sample_weights = None

        else:
            if not self.configs['classification_type'] in ['binary', 'multi-class', 'multi-label']:
                self.program_halt("invalid classification type in the config file: " + str(self.configs['classification_type']))

            num_train_examples = len(train_set_x[0])
            self.__bert_helper.build_model(selected_optimizer, selected_learning_rate, num_train_examples, selected_minibatch_size, selected_num_train_epochs)
            self.model = self.__bert_helper.model

        for epoch_counter in range(selected_num_train_epochs):
            if self.__negative_downsampling_rate > 0:
                train_set_x, train_set_y_true = negative_sp.get_new_sample()

            self.__epoch_counter = epoch_counter
            self.lp("=" * 80)
            self.lp("Training ... Epoch: " + str(epoch_counter))

            if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
                history = self.model.fit(train_set_x, train_set_y_true, epochs=1, batch_size=selected_minibatch_size, shuffle=True, verbose=selected_fit_verbose, sample_weight=y_labels_sample_weights)
            else:
                history = self.model.fit(train_set_x, train_set_y_true, epochs=1, batch_size=selected_minibatch_size, shuffle=True, verbose=selected_fit_verbose)

            self.lp("training loss : " + str(history.history['loss']))
            evaluation_results = self.__predict_evaluate_writeback(train_or_devel_or_test=2)

            if epoch_counter == 0:
                self.__best_model_weights = self.model.get_weights()
            else:
                this_epoch_evaluation_score = evaluation_results['evaluation_score']
                highest_evaluation_score = np.max([x['evaluation_score'] for x in self.__all_epochs_evaluation_results])
                if this_epoch_evaluation_score > highest_evaluation_score:
                    self.__best_model_weights = self.model.get_weights()

            self.__all_epochs_evaluation_results.append(evaluation_results)

        # getting and printing best scores (i.e., which epoch gave the best score)
        best_evaluation_scores = sorted(self.__all_epochs_evaluation_results, key=lambda x: x['evaluation_score'], reverse=True)[0]
        self.lp("BEST_RESULTS:\t" + str(best_evaluation_scores))

        if self.__save_best_model_folder_path is not None:
            self.__save_best_model(training_params, model_params, best_evaluation_scores)

    def __predict_evaluate_writeback(self, train_or_devel_or_test=2):
        from helpers import general_helpers  # safe to import here instead of the top, because in the future this module MAY include numpy or keras or tensorflow things, and we want random seed init to take place before importing anything.
        if train_or_devel_or_test not in (1, 2, 3):
            self.program_halt("train_or_devel should be either 1 for training_file or 2 for devel file.")

        if train_or_devel_or_test in (1, 3):
            self.program_halt("not implemented yet.")  # TODO: consider writing output files for training set if you're going to run an extrenal evaluator.

        elif train_or_devel_or_test == 2:
            set_name = "development"
            param_process_after_epoch = self.__process_devel_after_epoch
            param_should_predict = self.__predict_devel
            param_should_evaluate = self.__evaluate_devel
            param_should_write_back = self.__writeback_devel_preds
            param_write_back_folder = self.__writeback_devel_preds_folder
            x = self.development_set['x']
            y_true = self.development_set['y_true']
            pair_tracking = self.development_set['pair_tracking']
            file_items = self.development_set['items']

        if (param_process_after_epoch < 0) or (self.__epoch_counter < param_process_after_epoch) or (not param_should_predict):
            return

        # initialization
        # TODO: consider changing in the future for multi output neural network experiments:
        y_true = y_true[0]  # <<<CRITICAL>>> currently, the all neural network models have only 1 output for the main decision layer, but it is in the list e.g.: y_true = [decision_layer_out]

        # prediction
        # TODO: consider changing in the future for multi output neural network experiments:
        y_pred_confs = self.model.predict(x, verbose=0)

        if self.__augment_predictions_with_regulation:
            self.lp("[WARNING] : modifying y_pred_confs ...")
            y_pred_confs = self.__pred_conf_modifier(self.configs, self.lp, self.program_halt, y_pred_confs)

        # evaluation
        if not param_should_evaluate:
            return None

        internal_evaluation_scores = self.__low_level_internal_evaluation(y_true, y_pred_confs)
        external_evaluator_score, external_evaluator_message = "", ""

        # write-back and/or external evaluation:
        if (self.__external_evaluator is not None) or param_should_write_back:
            # write-back
            preds_sub_folder_name = "Epoch_" + str(self.__epoch_counter)
            preds_sub_folder_path = param_write_back_folder + preds_sub_folder_name
            try:
                os.mkdir(preds_sub_folder_path)
            except Exception as E:
                self.program_halt("Error in creating folder :" + preds_sub_folder_path + "\nError: " + str(E))

            for item in file_items:
                json_data = item['json_data']
                brat_ann_filepath = preds_sub_folder_path + "/" + item['output_brat_file_name']
                self.__write_back_json_entities_into_brat_ann(json_data, brat_ann_filepath)

            self.__write_back_interaction_preds_to_ann_files(pair_tracking, y_pred_confs, preds_sub_folder_path)

            # run external evaluator
            if self.__external_evaluator is not None:
                external_evaluator_score, external_evaluator_message = self.__external_evaluator(preds_sub_folder_path, self.program_halt)

            # remove mess if no write-back has been requested, otherwise keep the folder content.
            if not param_should_write_back:
                shutil.rmtree(preds_sub_folder_path)

        # select evaluation score ...
        if self.__external_evaluator is None:
            if self.__evaluation_metric not in internal_evaluation_scores:
                self.program_halt("invalid evaluation_metric: " + str(self.__evaluation_metric))
            evaluation_score = internal_evaluation_scores[self.__evaluation_metric]
        else:
            evaluation_score = external_evaluator_score

        # print eval results
        eval_msg = ["Evaluation results: " + set_name, "-" * 80]
        eval_msg += ["Internal evaluation results:", "-" * 40]
        for i, j in internal_evaluation_scores.items():
            eval_msg += [general_helpers.nlvr(i, 60) + " : " + str(j)]
        eval_msg += ["-" * 40]

        if self.__external_evaluator is None:
            eval_msg += ["External evaluation results: None"]
        else:
            eval_msg += ["External evaluation results:", "-" * 40, external_evaluator_message, "-" * 40]
        eval_msg += ["Evaluation Score: " + str(evaluation_score), "-" * 80]
        self.lp(eval_msg)

        # return results
        results = {
            "evaluation_score": evaluation_score,
            "internal_evaluation_scores": internal_evaluation_scores,
            "external_evaluator_score": external_evaluator_score,
            "external_evaluator_message": external_evaluator_message,
            "epoch": self.__epoch_counter,
        }
        return results

    def __low_level_internal_evaluation(self, y_true, y_pred_confs):
        import numpy as np
        import sklearn.metrics as sklearn_eval_metrics

        if self.configs['classification_type'] in ["binary", "multi-class"]:
            """
            Notes:
                  - For binary and multi-class setups, there IS a neg dimension. 
                  - Output for each example is something like [0.1, 0.8, 0.1], because decision layer has softmax activation. 
                  - Therefore, we take the argmax with axis=1, to find which dimension in the example array has the maximum probability.
                  - We do this for both y_true and y_pred_confs, because even in y_true (in binary and multi-class) we need to know which dimension has the maximum prob. 
                  - For example if y_true[0,:] == [1, 0, 0] then argmax gives 0 (the index of the element with maximum value). 
            
            Example:
                    a = np.array([[0.6 , 0.2 , 0.2 ],
                                  [0.1 , 0.8 , 0.1 ],
                                  [0.05, 0.05, 0.9 ]])
                    np.argmax(a, axis=1) --> array([0, 1, 2])
            """
            y_true_labels_indices = np.argmax(y_true, axis=1)
            y_pred_labels_indices = np.argmax(y_pred_confs, axis=1)
            if not (len(y_true_labels_indices) == len(y_pred_labels_indices)):
                self.program_halt("Length of y_true and y_pred should be the same!")

        else:
            """
            Multi-label Notes:
                  - For multi-label, there is NO dimension for the negative class in the decision layer.
                  - In addition, we have used sigmoid activation in the decision layer.
                  - Example output for one example pair:  [0.4, 0.6, 0.6]  ... this should be converted to --> [0,1,1]
                  - We don't need to do this for y_true, because y_true in its original shape is like [0,1,1], so no need to transform anything really.  
            Example:
                   a = np.array([[0.6, 0.6, 0.2],
                                 [0.1, 0.8, 0.8],
                                 [0.2, 0.3, 0.4],
                                 [0.5, 0.5, 0.5]])
                   
                   np.where(a >= 0.5 , 1 , 0) --> 
                   array([[1, 1, 0],
                          [0, 1, 1],
                          [0, 0, 0],
                          [1, 1, 1]])
            """
            y_true_labels_indices = y_true
            y_pred_labels_indices = np.where(y_pred_confs >= 0.5, 1, 0)
            if y_true_labels_indices.shape != y_pred_labels_indices.shape:
                self.program_halt("shape of y_true and y_pred should be the same!")

        if self.configs['classification_type'] == 'binary':
            precision, recall, f_score, support = sklearn_eval_metrics.precision_recall_fscore_support(y_true_labels_indices, y_pred_labels_indices)
            _counts = sklearn_eval_metrics.confusion_matrix(y_true_labels_indices, y_pred_labels_indices)
            tp = _counts[1, 1]
            tn = _counts[0, 0]
            fp = _counts[0, 1]
            fn = _counts[1, 0]
            auc_score = sklearn_eval_metrics.roc_auc_score(y_true_labels_indices, y_pred_confs[:, 1])
            accuracy = sklearn_eval_metrics.accuracy_score(y_true_labels_indices, y_pred_labels_indices)
            return OrderedDict([
                ("confusion_TP", tp),
                ("confusion_TN", tn),
                ("confusion_FP", fp),
                ("confusion_FN", fn),
                ("negative_precision", precision[0]),
                ("negative_recall", recall[0]),
                ("negative_f1_score", f_score[0]),
                ("negative_support", support[0]),
                ("positive_precision", precision[1]),
                ("positive_recall", recall[1]),
                ("positive_f1_score", f_score[1]),
                ("positive_support", support[1]),
                ("total_f1_macro", np.mean(f_score)),
                ("total_f1_weighted", (((f_score[0] * support[0]) + (f_score[1] * support[1])) / float(support[0] + support[1]))),
                ("accuracy", accuracy),
                ("AUC_score", auc_score),
            ])

        elif self.configs['classification_type'] in ["multi-class", "multi-label"]:
            evaluation_results = OrderedDict()
            number_of_classes = self.configs['RelationTypeEncoding'].number_of_classes
            integer_labels_list = list(range(number_of_classes))  # --> e.g. , [0,1,2,3,4]
            inv_class_lbls = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping

            per_class_precision, per_class_recall, per_class_f_score, per_class_support = sklearn_eval_metrics.precision_recall_fscore_support(y_true_labels_indices, y_pred_labels_indices, labels=integer_labels_list)
            for i in range(number_of_classes):
                this_class_lbl = inv_class_lbls[i]
                evaluation_results[this_class_lbl + "_precision"] = per_class_precision[i]
                evaluation_results[this_class_lbl + "_recall"] = per_class_recall[i]
                evaluation_results[this_class_lbl + "_f1_score"] = per_class_f_score[i]
                evaluation_results[this_class_lbl + "_support"] = per_class_support[i]

            # <<<CRITICAL>>>
            #   - multi-label does not have a negative dimension, hence when calculating total_micro_f1, we are concenterating on positive labels, excluding negatives in evaluation
            #   - multi-class does have a negative dimension, hence we have to exclude the negative dimension (0) ourselves manually, in order to get f1_micro for positive classes.
            #     example: y_true = [0,1,4,3,2,3,2,2,3] , y_pred = [0,1,3,3,2,3,2,2,3] ...
            if self.configs['classification_type'] == "multi-label":
                positive_integer_labels_list = integer_labels_list
            else:
                positive_integer_labels_list = list(range(1, number_of_classes))
                # e.g: if we have 10 positive class and a negative class, number_of_classes = 11. Hence, list(range(1, 11)) --> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            evaluation_results["total_f1_macro"] = sklearn_eval_metrics.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=positive_integer_labels_list, average="macro")
            evaluation_results["total_f1_micro"] = sklearn_eval_metrics.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=positive_integer_labels_list, average="micro")
            evaluation_results["total_f1_weighted"] = sklearn_eval_metrics.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=positive_integer_labels_list, average="weighted")

            """
            task_positive_integer_labels = [3,4,5,6,9]
            evaluation_results["task_f1_macro"]    = METRICS.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=task_positive_integer_labels, average="macro")
            evaluation_results["task_f1_micro"]    = METRICS.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=task_positive_integer_labels, average="micro")
            evaluation_results["task_f1_weighted"] = METRICS.f1_score(y_true_labels_indices, y_pred_labels_indices, labels=task_positive_integer_labels, average="weighted")
            """

            # TODO: implement AUC and Accuracy later ...
            # TODO: implement ignoring one or more classes later ...
            """
            ignore_class_labels_for_evaluation = self.__PRJ.Configs["EvaluationParameters"]["ExcludeClassLabelsList"]
            if len(ignore_class_labels_for_evaluation) > 0:
                
                #The following could be nice, but does not work for the "negative" class(ES)!!!
                #Because negative classes are ALWAYS represented as 'negative' in OneHotEncodingForMultiClass.
                #     IG_IDX = [class_lbls[i.lower()] for i in ignore_class_labels_for_evaluation]
                #     IG_LBL = "total_MINUS_" + "_".join ([inv_class_lbls[i] for i in IG_IDX]) ;
                
                # <<<CRITICAL>>>
                IG_IDX = []
                for igc in ignore_class_labels_for_evaluation:
                    igc = igc.lower()
                    if igc in class_lbls:
                        IG_IDX.append(class_lbls[igc])
                    else:
                        if igc in self.__PRJ.Configs["CLASSES"]["Negative"]:
                            if not 0 in IG_IDX:
                                IG_IDX.append(0)

                ignore_title = "total_MINUS_" + "_".join([inv_class_lbls[i] for i in IG_IDX]) + "_"
                REMAIN_IDX = [i for i in range(len(class_lbls)) if not i in IG_IDX]
                evaluation_results[ignore_title + "f1_macro"]    = METRICS.f1_score(y_true, y_pred, labels=REMAIN_IDX, average="macro")
                evaluation_results[ignore_title + "f1_micro"]    = METRICS.f1_score(y_true, y_pred, labels=REMAIN_IDX, average="micro")
                evaluation_results[ignore_title + "f1_weighted"] = METRICS.f1_score(y_true, y_pred, labels=REMAIN_IDX, average="weighted")
            """
            return evaluation_results

    def __write_back_json_entities_into_brat_ann(self, json_data, brat_ann_filepath):
        try:
            file_handle = open(brat_ann_filepath, "wt", encoding="utf-8")
        except Exception as E:
            self.program_halt("error in opening file to write: " + brat_ann_filepath + "\nError:" + str(E))

        for document in json_data['documents']:
            for entity_id in document['entities'].keys():
                entity_info = document['entities'][entity_id]  # dictionary
                output_string = entity_id + "\t" + entity_info['tag'] + " "
                for span in entity_info['orig_spans']:
                    output_string += str(span[0]) + " " + str(span[1]) + ";"
                output_string = output_string[:-1]
                output_string += "\t" + entity_info['text'] + "\n"
                file_handle.write(output_string)
        file_handle.close()

    def __write_back_interaction_preds_to_ann_files(self, pair_tracking, y_pred_confs, pred_folder):
        import numpy as np
        relation_id_trackers = {}  # just to control R0, R1, R2, ... generation in each .ann output file.

        # for binary and multi-class, we get the index of the element with the maximum number, for example [0.2, 0.8, 0.0] --> [1]
        # but for multi-label there maybe zero or more elements which are >= 0.5 and all of those need to be written to the file.
        if self.configs['classification_type'] in ["binary", "multi-class"]:
            y_pred_labels_indices = np.argmax(y_pred_confs, axis=1)

        if self.configs['classification_type'] in ["binary", "multi-class"]:
            for index, item in enumerate(pair_tracking):
                brat_file_name, document_id, example_id, example_e1, example_e2, this_example_tokens_indices , this_example_tokens_offsets = item
                predicted_relation_type = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[
                    y_pred_labels_indices[index]]
                if predicted_relation_type == 'neg':
                    continue

                # <<<CRITICALL>>>:
                # 1) For non-directional relation types (e.g. 'Complex_formation'), we don't need to look at the order of e1 and e2.
                # 2) For directional relation types (e.g., Regulation), if the direction is > (e.g., "Regulation>"), we don't need to change the ordger of e1 and e2.
                #    It is predicted that direction is from e1 to e2. So we can write in the .ann file : R108 Regulation Arg1:e1 Arg2:e2
                # 3) If direction is < , we need to swap e1 and e2, and write R108 Regulation Arg1:e2 Arg1:e1
                if predicted_relation_type[-1] == ">":
                    predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation>' --> 'Regulation'
                elif predicted_relation_type[-1] == "<":
                    predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation<'  --> 'Regulation'
                    example_e1, example_e2 = example_e2, example_e1

                with open(pred_folder + "/" + brat_file_name, "at") as file_handle:
                    if brat_file_name in relation_id_trackers:
                        relation_id_trackers[brat_file_name] += 1
                    else:
                        relation_id_trackers[brat_file_name] = 1
                    relation_id = "R" + str(relation_id_trackers[brat_file_name])
                    output_string = relation_id + "\t" + predicted_relation_type + " Arg1:" + example_e1 + " Arg2:" + example_e2 + "\n"
                    file_handle.write(output_string)
        else:
            """
            For multi-label: 
                   a = np.array([[0.6, 0.6, 0.2],
                                 [0.1, 0.8, 0.8],
                                 [0.2, 0.3, 0.4],
                                 [0.5, 0.5, 0.5]])
                   
                   list(np.nonzero(a[0,:]>=0.5)[0]) --> [0, 1]
                   list(np.nonzero(a[1,:]>=0.5)[0]) --> [1, 2]
                   list(np.nonzero(a[2,:]>=0.5)[0]) --> []
                   list(np.nonzero(a[3,:]>=0.5)[0]) --> [0, 1, 2]
            """
            for index, item in enumerate(pair_tracking):
                brat_file_name, document_id, example_id, example_e1, example_e2 , this_example_tokens_indices, this_example_tokens_offsets = item
                positive_label_indices = list(
                    np.nonzero(y_pred_confs[index, :] >= 0.5)[0])  # --> for example something like [0,4,5] or [4] or []
                for posivie_label_index in positive_label_indices:
                    predicted_relation_type = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[
                        posivie_label_index]
                    if predicted_relation_type[-1] == ">":
                        writeback_e1, writeback_e2 = example_e1, example_e2
                        predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation>' --> 'Regulation'

                    elif predicted_relation_type[-1] == "<":
                        writeback_e1, writeback_e2 = example_e2, example_e1  # <<<CRITICAL>>> SWAP e1 and e2
                        predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation<'  --> 'Regulation'
                    else:
                        writeback_e1, writeback_e2 = example_e1, example_e2

                    with open(pred_folder + "/" + brat_file_name, "at") as file_handle:
                        if brat_file_name in relation_id_trackers:
                            relation_id_trackers[brat_file_name] += 1
                        else:
                            relation_id_trackers[brat_file_name] = 1
                        relation_id = "R" + str(relation_id_trackers[brat_file_name])
                        output_string = relation_id + "\t" + predicted_relation_type + " Arg1:" + writeback_e1 + " Arg2:" + writeback_e2 + "\n"
                        file_handle.write(output_string)

    def __save_best_model(self, training_params, model_params, best_evaluation_scores):
        if self.__save_best_model_folder_path is None:
            return

        folder_path = self.__save_best_model_folder_path
        info = {
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "max_seq_len": self.max_seq_len,
            "representation_strategy": str(self.__representation_strategy),
            "training_params": training_params,
            "model_params": model_params,
            "best_evaluation_scores": eval(str(best_evaluation_scores)),
            "configs": str(self.configs),
        }

        try:
            self.lp("Saving best model to " + folder_path)
            with open(folder_path + "info.json", "wt", encoding='utf-8') as f:
                json.dump(info, f)
            self.model.set_weights(self.__best_model_weights)
            self.model.save_weights(folder_path + "best_model_weights.h5")
            if self.__backend == pipeline_variables.BACKEND_Type.HUGGINGFACE_TENSORFLOW:
                self.__bert_helper.tokenizer.save_pretrained(folder_path)
        except Exception as E:
            msg = "[WARNING]: could not proceed with saving the best model.\n"
            msg += "folder :" + str(folder_path) + "\n"
            msg += "erorr  :" + str(E)
            self.lp(msg)
            return

    def __load_best_model_from_folder(self):
        self.__loading_fully_pretrained_relation_extraction_model_is_requested = True
        try:
            msg = ["[WARNING]: No max_seq_len is given."]
            msg += ["Assumming you're providing a fully pretrained relation extraction model in a folder."]
            msg += ["Trying to get information from info.json from : " + self.pretrained_model_name_or_path]
            self.lp(msg)
            if self.pretrained_model_name_or_path[-1] != "/":
                self.pretrained_model_name_or_path += "/"
            with open(self.pretrained_model_name_or_path + "info.json", "rt", encoding='utf-8') as f:
                json_content = json.load(f)
            #self.lp(["info.json file abs-path: " + os.path.abspath(self.pretrained_model_path + "info.json"), "info.json file contents:", "-" * 20, json.dumps(json_content, indent=4, sort_keys=False), "-" * 80])
            fetched_pretrained_model_name_or_path = json_content['pretrained_model_name_or_path']
            max_seq_len = int(json_content['max_seq_len'])
            representation_strategy = eval("pipeline_variables." + json_content['representation_strategy'])
            training_params = json_content['training_params']
            model_params = json_content['model_params']
            msg = ["pretrained_model_name_or_path : " + fetched_pretrained_model_name_or_path]
            msg+= ["max_seq_len                   : " + str(max_seq_len)]
            msg+= ["representation_strategy       : " + str(representation_strategy)]
            msg+= ["training_params               : " + str(training_params)]
            msg+= ["model_params                  : " + str(model_params)]
            self.lp(msg)
        except Exception as E:
            self.program_halt("error loading file. Error:" + str(E))

        self.__representation_strategy = representation_strategy # <<<CRITICAL>>> very important
        if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
            from helpers import bert_helper
            from helpers import architecture_builder
            from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator
            self.__bert_helper = bert_helper.BERT_helper(self.lp, self.program_halt, fetched_pretrained_model_name_or_path, max_seq_len)
            self.__ANN_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt, self.configs, self.__bert_helper)
            self.__architecture_builder = architecture_builder.Architecture_Builder(self.lp, self.program_halt, self.configs, self.__bert_helper)
        else:
            from helpers import tf_model_helper
            from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator
            self.__bert_helper = tf_model_helper.TF_helper(self.lp, self.program_halt, self.configs, fetched_pretrained_model_name_or_path, max_seq_len=max_seq_len, tokenizer_folder_path=self.pretrained_model_name_or_path)
            self.__ANN_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt, self.configs, self.__bert_helper)

        # BUILD MODEL FOR PREDICTION HERE ...
        # evaluate and get training_params
        self.__evaluate_training_params(training_params)
        selected_optimizer = training_params["optimizer"].lower()
        selected_learning_rate = training_params["learning_rate"]
        selected_minibatch_size = training_params["minibatch_size"]
        selected_num_train_epochs = training_params["num_train_epochs"]

        if self.__backend == pipeline_variables.BACKEND_Type.KERAS_BERT:
            self.lp("building model ...")
            self.model = self.__architecture_builder.generate_BERT_pair_prediction(model_params)

            # initialize optimizer with based on the selected optimizer
            from keras.optimizers import Adam, Nadam, SGD
            if selected_optimizer == "adam":
                my_optimizer = Adam(lr=selected_learning_rate)
            elif selected_optimizer == "nadam":
                my_optimizer = Nadam(lr=selected_learning_rate)
            elif selected_optimizer == "sgd":
                my_optimizer = SGD(lr=selected_learning_rate)
            elif selected_optimizer == "adamwarmup":
                from keras_bert import AdamWarmup, calc_train_steps
                param_total_training_example_count = 100000 # just a number, because we're not going to retrain this model, but we need a number
                total_steps, warmup_steps = calc_train_steps(
                    num_example=param_total_training_example_count,
                    batch_size=selected_minibatch_size,
                    epochs=selected_num_train_epochs,
                    warmup_proportion=0.1,
                )
                my_optimizer = AdamWarmup(total_steps, warmup_steps, lr=selected_learning_rate)

            # which loss-function to use based on type of the task ...
            if not self.configs['classification_type'] in ['binary', 'multi-class', 'multi-label']:
                self.program_halt(
                    "invalid classification type in the config file: " + str(self.configs['classification_type']))

            if self.configs['classification_type'] in ['binary', 'multi-class']:
                loss_function_name = "categorical_crossentropy"
            else:
                loss_function_name = "binary_crossentropy"  # multi-label

            # compile model
            self.lp("compiling model ...")
            self.model.compile(loss=loss_function_name, optimizer=my_optimizer)

        else:
            if not self.configs['classification_type'] in ['binary', 'multi-class', 'multi-label']:
                self.program_halt("invalid classification type in the config file: " + str(self.configs['classification_type']))

            num_train_examples = 100000  # just a number ...
            self.__bert_helper.build_model(selected_optimizer, selected_learning_rate, num_train_examples, selected_minibatch_size, selected_num_train_epochs)
            self.model = self.__bert_helper.model

        # LOAD BEST MODEL WEIGHTS ...
        try:
            self.lp ("loading best model weights from " + self.pretrained_model_name_or_path + "best_model_weights.h5 ...")
            self.model.load_weights(self.pretrained_model_name_or_path + "best_model_weights.h5")
        except Exception as E:
            self.program_halt("could not load best_model_weights.h5 from " + self.pretrained_model_name_or_path + "\nError: " + str(E))

        self.__best_model_weights = self.model.get_weights()
        self.__all_epochs_evaluation_results = [json_content["best_evaluation_scores"]]
        self.__epoch_counter = json_content["best_evaluation_scores"]["epoch"]
        self.development_set = {"items": [], "x": None, "y_true": None, "pair_tracking": None}  # each item is a json example file (input/output file name and json content), whereas x, y_true, pair_tracking are created by ann_io_generation

    def only_predict_devel_and_exit(self):
        if not self.__loading_fully_pretrained_relation_extraction_model_is_requested:
            self.lp("you have not requested to load a fully pretrained relation extraction model. exiting.")
            return
        self.__assign_network_inputs_outputs(train_or_devel=2)
        self.__predict_evaluate_writeback(train_or_devel_or_test=2)


    def add_train_dev_test_brat_folders(self, train_path, devel_path, test_path=None, dont_generate_negatives_if_sentence_distance_ge=None):
        from helpers import general_helpers as ge
        from helpers import brat_json_converter
        from helpers import example_generation_cross_sentence_MD

        #check training set ...
        if not isinstance(train_path, str):
            self.program_halt("train_path should be a valid folder path.")
        if not os.path.isdir(train_path):
            self.program_halt("invalid folder address for train_path: " + train_path)
        if len(ge.get_all_files_with_extension(train_path , "ann")) < 1:
            self.program_halt("no .ann files were found for the given train_path in the folder : " + train_path)

        #check devel set ...
        if not isinstance(devel_path, str):
            self.program_halt("devel_path should be a valid folder path.")
        if not os.path.isdir(devel_path):
            self.program_halt("invalid folder address for devel_path: " + devel_path)
        if len(ge.get_all_files_with_extension(devel_path , "ann")) < 1:
            self.program_halt("no .ann files were found for the given devel_path in the folder : " + devel_path)

        #check test set ...
        if test_path is not None:
            if not isinstance(test_path, str):
                self.program_halt("test_path should be either None or a valid folder path.")
            if not os.path.isdir(test_path):
                self.program_halt("test_path should be either None or a valid folder path. Given: " + test_path)
            if len(ge.get_all_files_with_extension(test_path, "ann")) < 1:
                self.program_halt("test_path should be either None or a valid folder path containing .ann files. No .ann files were found for the given test_path in the folder : " + test_path)

        #create needed objects ...
        my_brat_json_converter = brat_json_converter.brat_json_Converter(self.lp, self.program_halt, self.configs)
        my_example_generator = example_generation_cross_sentence_MD.example_generator(self.lp, self.program_halt, self.configs)

        train_rel_info , devel_rel_info , test_rel_info = dict(), dict(), dict()
        train_pair_info, devel_pair_info, test_pair_info = dict(), dict(), dict()

        folders = [("train" , train_path , train_rel_info, train_pair_info),
                   ("devel" , devel_path , devel_rel_info, devel_pair_info)]

        if test_path is not None:
            folders.append(("test" , test_path , test_rel_info, test_pair_info))

        #process train, devel, test input brat folders
        for folder_name, folder_path, set_rel_info, set_pair_info in folders:
            self.lp ("processing " + folder_name + " set, from folder :" + folder_path)
            file_count = 0
            for input_ann_file_path in ge.get_all_files_with_extension(folder_path , "ann"):
                input_txt_file_path = input_ann_file_path.replace(".ann", ".txt")

                #0. init
                file_count+= 1

                #1.1 brat to json conversion
                json_content = my_brat_json_converter.convert_brat_to_json(input_txt_file_path, input_ann_file_path, output_file_json=None)

                #1.2 get relations info from the json object in RAM
                for document in json_content['documents']:
                    for relation_id in document['relations']:
                        rel_tp = document['relations'][relation_id]['type']
                        if rel_tp in set_rel_info:
                            set_rel_info[rel_tp] += 1
                        else:
                            set_rel_info[rel_tp] = 1

                #2.1 generate examples ...
                json_content, counter_annotated_positives, counter_generated_positives, counter_generated_negatives =  my_example_generator.generate_examples(
                    input_json_fileaddress= None,
                    output_json_fileaddress = None,
                    input_json_conent=json_content,
                    dont_generate_negatives_if_sentence_distance_ge=dont_generate_negatives_if_sentence_distance_ge)

                #2.2 get pairs info from the json object in RAM
                for document in json_content['documents']:
                    for pair in document['pairs']:
                        labels = pair['labels']
                        if len(labels) == 0:
                            if 'neg' in set_pair_info:
                                set_pair_info['neg'] += 1
                            else:
                                set_pair_info['neg'] = 1
                        else:
                            for lbl in labels:
                                if lbl in set_pair_info:
                                    set_pair_info[lbl] += 1
                                else:
                                    set_pair_info[lbl] = 1


                #3. add item
                item = {
                    "output_brat_file_name": None,
                    "json_data": json_content,
                }

                if folder_name == "train":  # since we don't evaluate training files with an external evaluator, we don't need to create output files for the training set files.
                    self.training_set['items'].append(item)

                elif folder_name == "devel":
                    # should i create corresponding prediction output files for devel predictions?
                    if self.__writeback_devel_preds or (self.__external_evaluator is not None):
                        brat_file_name = input_ann_file_path.split("/")[-1] #basically we'll only need the "pmid.ann"
                        item["output_brat_file_name"] = brat_file_name
                    self.development_set['items'].append(item)

                elif folder_name == "test":
                    pass #there is no support to pred test at the moment in the pipeline.



        #reports: report with "\t" for later eval() if needed
        all_rel_types = sorted(set(train_rel_info.keys()) | set(devel_rel_info.keys()) | set(test_rel_info.keys()))
        all_pair_types = set(train_pair_info.keys()) | set(devel_pair_info.keys()) | set(test_pair_info.keys())
        all_pair_types = all_pair_types - set(['neg'])
        all_pair_types = ['neg'] + sorted(all_pair_types)

        for rel_info in [train_rel_info , devel_rel_info , test_rel_info]:
            for key in all_rel_types:
                if not key in rel_info.keys():
                    rel_info[key] = 0

        for pair_info in [train_pair_info , devel_pair_info , test_pair_info]:
            for key in all_pair_types:
                if not key in pair_info.keys():
                    pair_info[key] = 0

        msg = [
            "[TRAIN_INFO]\t" + str(train_path) + "\t" + str(train_rel_info) + "\t" + str(train_pair_info) ,
            "[DEVEL_INFO]\t" + str(devel_path) + "\t" + str(devel_rel_info) + "\t" + str(devel_pair_info) ,
            "[TEST_INFO]\t"  + str(test_path)  + "\t" + str(test_rel_info)  + "\t" + str(test_pair_info)  ,
        ]
        self.lp (msg)

        # reports: pretty print relations
        msg = []
        msg.append("="*100)
        msg.append ("RELATIONS INFORMATION (.ann files):")
        msg.append("-"*100)
        msg.append (ge.nlvr("relation_type", 60) + ge.nlvl("#train", 10) + ge.nlvl("#devel", 10) + ge.nlvl("#test", 10))
        for key in all_rel_types:
            msg.append(ge.nlvr(key, 60) + ge.nlvl(train_rel_info[key], 10) + ge.nlvl(devel_rel_info[key], 10) + ge.nlvl(test_rel_info[key], 10))
        msg.append("="*100)
        self.lp(msg)

        # reports: pretty print pairs
        msg = []
        msg.append("="*100)
        msg.append ("PAIRS INFORMATION (.ann files):")
        msg.append("-"*100)
        msg.append (ge.nlvr("pair_type", 60) + ge.nlvl("#train", 10) + ge.nlvl("#devel", 10) + ge.nlvl("#test", 10))
        for key in all_pair_types:
            msg.append(ge.nlvr(key, 60) + ge.nlvl(train_pair_info[key], 10) + ge.nlvl(devel_pair_info[key], 10) + ge.nlvl(test_pair_info[key], 10))
        msg.append("="*100)
        self.lp(msg)
