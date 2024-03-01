"""
This is the code for large-scale prediction on a cluster of gpu-s like puhti.
    Input: a folder, containing .tar.gz bundles, each bundle, contains a one or more .ann+.txt files for relation extraction.
    For example: 1.tar.gz , 2.tar.gz , ... , 100.tar.gz

    Outputs:
    1) extracted relations: in the form of 1.tar.gz, 2.tar.gz , ... , 100.tar.gz
    2) error logs: in the form of 1.tar.gz, 2.tar.gz, ... , 100.tar.gz

"""
import os
import sys
import json
import gzip
import shutil
import tarfile
import datetime
import numpy as np
import argparse

from helpers import logger
from helpers import configs_manager
from helpers import pipeline_variables
from helpers import general_helpers as ge

from helpers import brat_json_converter
from helpers import example_generation_cross_sentence_MD
from helpers import tf_model_helper
from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator

class Large_Scale_Prediction_Pipeline_tensorflow(object):

    def __init__(self,
                 configs_file_path,
                 log_file_path,
                 pretrained_model_path,
                 input_folder_path,
                 output_folder_path,
                 dont_generate_negatives_if_sentence_distance_ge=7,
                 create_output_ann_files=False):

        self.configs_file_path = configs_file_path
        self.log_file_path = log_file_path
        self.pretrained_model_path = pretrained_model_path

        if os.path.isfile(log_file_path):
            print ("log file already exists. deleting and recreating the file :" + log_file_path)
            os.remove(log_file_path)
        if not os.path.isfile(configs_file_path):
            print("[ERROR] configs file not found: " + configs_file_path + "\nHALTING.")
            sys.exit(0)

        self.logger = logger.Logger(log_file_path)
        self.lp = self.logger.lp
        self.configs = configs_manager.ConfigsManager(configs_file_path, self.lp, self.program_halt).configs
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.temp_folder_path = None
        self.dont_generate_negatives_if_sentence_distance_ge = dont_generate_negatives_if_sentence_distance_ge
        self.create_output_ann_files = create_output_ann_files

        self.__check_input_output_folders()

        # log params:
        msg_params = ["Running pipeline:", "-" * 80,
                      "\t- configs_file_path              : " + str(self.configs_file_path),
                      "\t- log_file_path                  : " + str(self.log_file_path),
                      "\t- pretrained_model_name_or_path  : " + str(self.pretrained_model_path),
                      "\t- input_folder_path              : " + self.input_folder_path,
                      "\t- output_folder_path             : " + self.output_folder_path,
                      "\t- temp_folder_path               : " + self.temp_folder_path,
                      "\t- create_output_ann_files        : " + str(self.create_output_ann_files),
                      "\t- dont_generate_negatives_if_sentence_distance_ge : " + str(self.dont_generate_negatives_if_sentence_distance_ge),
                      "-" * 80]
        self.lp(msg_params)

        self.__load_pretrained_relation_extraction_model_tensorflow()

    def program_halt(self, message):
        self.logger.lp_halt(message)
        self.exit()

    def exit(self):
        if self.logger.is_open():
            self.lp ("EXITING PROGRAM ... ")
            self.logger.close()
        sys.exit(0)

    def program_halt_raise_exception_do_not_exit(self, message):
        raise Exception(message)

    def __mkdir(self, folder_address, log_folder_creation_with_message=False):
        if not shutil.os.path.exists(folder_address):
            try:
                if log_folder_creation_with_message:
                    self.lp("creating folder : " + folder_address)
                shutil.os.makedirs(folder_address)
            except Exception as E:
                self.program_halt("could not create folder : " + folder_address + "\nerror: " + str(E))
        else:
            self.lp ("folder already exists: " + folder_address)

    def __rm_directory_with_content(self,folder_address, log_folder_creation_with_message=False):
        if shutil.os.path.exists(folder_address):
            try:
                if log_folder_creation_with_message:
                    self.lp("removing folder with all contents : " + folder_address)
                shutil.rmtree(folder_address)
            except Exception as E:
                self.program_halt("could not remove folder : " + folder_address + "\nerror: " + str(E))

    def __check_input_output_folders(self):
        if not isinstance(self.input_folder_path, str):
            self.program_halt("invalid path for input_folder_path :" + str(self.input_folder_path))
        if not isinstance(self.output_folder_path, str):
            self.program_halt("invalid path for output_folder_path :" + str(self.output_folder_path))
        if self.input_folder_path[-1] != "/":
            self.input_folder_path += "/"
        if self.output_folder_path[-1] != "/":
            self.output_folder_path += "/"
        if not os.path.isdir(self.input_folder_path):
            self.program_halt("invalid path for input_folder_path :" + str(self.input_folder_path))
        if not os.path.isdir(self.output_folder_path):
            self.program_halt("invalid path for output_folder_path :" + str(self.output_folder_path))

        self.temp_folder_path = self.output_folder_path + "tmp" + "_" + datetime.datetime.now().strftime("%Y%M%d_%H%M%S") + "/"
        self.__rm_directory_with_content(self.temp_folder_path)
        self.__mkdir(self.temp_folder_path)

    def __load_pretrained_relation_extraction_model_tensorflow(self):
        try:
            msg = ["Trying to fetch all information from info.json from folder : " + self.pretrained_model_path]
            self.lp(msg)
            if self.pretrained_model_path[-1] != "/":
                self.pretrained_model_path += "/"
            with open(self.pretrained_model_path + "info.json", "rt", encoding='utf-8') as f:
                json_content = json.load(f)
            self.lp(["info.json file abs-path: " + os.path.abspath(self.pretrained_model_path + "info.json"), "info.json file contents:", "-" * 20, json.dumps(json_content, indent=4, sort_keys=False), "-" * 80])
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

        #CREATE THE MOST IMPORTANT OBJECTS ...
        self.brat_json_converter = brat_json_converter.brat_json_Converter(self.lp, self.program_halt, self.configs)
        self.example_generator = example_generation_cross_sentence_MD.example_generator(self.lp, self.program_halt, self.configs)
        self.tf_model_helper = tf_model_helper.TF_helper(self.lp, self.program_halt, self.configs, fetched_pretrained_model_name_or_path, max_seq_len=max_seq_len, tokenizer_folder_path=self.pretrained_model_path)
        self.ann_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt, self.configs, self.tf_model_helper)

        # BUILD MODEL FOR PREDICTION HERE ...
        # evaluate and get training_params
        self.__evaluate_training_params(training_params)
        selected_optimizer = training_params["optimizer"].lower()
        selected_learning_rate = training_params["learning_rate"]
        selected_minibatch_size = training_params["minibatch_size"]
        selected_num_train_epochs = training_params["num_train_epochs"]
        num_train_examples = 100000  # just a number ...
        self.tf_model_helper.build_model(selected_optimizer, selected_learning_rate, num_train_examples, selected_minibatch_size, selected_num_train_epochs)
        self.model = self.tf_model_helper.model

        # LOAD MODEL_WEIGHTS ...
        try:
            self.lp ("loading best model weights from " + self.pretrained_model_path + "best_model_weights.h5 ...")
            self.model.load_weights(self.pretrained_model_path + "best_model_weights.h5")
        except Exception as E:
            self.program_halt("could not load best_model_weights.h5 from " + self.pretrained_model_path + "\nError: " + str(E))

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

    def run_large_scale_pipeline(self):
        self.brat_json_converter.program_halt = self.program_halt_raise_exception_do_not_exit
        self.example_generator.program_halt = self.program_halt_raise_exception_do_not_exit
        self.tf_model_helper.program_halt = self.program_halt_raise_exception_do_not_exit
        self.ann_input_output_generator.program_halt = self.program_halt_raise_exception_do_not_exit

        for input_tar_gz_file_path in sorted(ge.get_all_files_with_extension(self.input_folder_path , "tar.gz")):
            date_time_start = datetime.datetime.now()
            self.lp ("input tar.gz: " + input_tar_gz_file_path)
            input_tar_gz_file_name = input_tar_gz_file_path.split("/")[-1]
            output_gzip_file_path = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + ".gz"
            output_errlog_file_path = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + ".err.log"

            # 1: create output ann folder if requested
            try:
                if self.create_output_ann_files:
                    output_ann_folder = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + "/"
                    self.__rm_directory_with_content(output_ann_folder)
                    self.__mkdir(output_ann_folder)
                else:
                    output_ann_folder = None
            except Exception as E:
                errmsg = ["[ERROR]: input tar.gz : " + input_tar_gz_file_path]
                errmsg += ["TYPE: could not output ann folder :" + output_errlog_file_path]
                errmsg += ["Error details: " + str(E)]
                self.lp (errmsg)
                date_time_end = datetime.datetime.now()
                self.lp ("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 2: create error_log handler
            try:
                if os.path.isfile(output_errlog_file_path):
                    os.remove(output_errlog_file_path)
                this_input_logger = logger.Logger(output_errlog_file_path, write_header_to_file=False)
                errlp = this_input_logger.lp
            except Exception as E:
                errmsg = ["[ERROR]: input tar.gz : " + input_tar_gz_file_path]
                errmsg += ["TYPE: could not create log file :" + output_errlog_file_path]
                errmsg += ["Error details: " + str(E)]
                self.lp (errmsg)
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 3: create output_gz file handler
            try:
                output_gz_file_handler = None
                output_gz_file_handler = gzip.open(output_gzip_file_path, "wt", encoding='utf-8')
                #write confidence score indices of the task , e.g. : {0: 'neg', 1: 'Complex_formation'}
                output_gz_file_handler.write("# confidence indices:\t" + str(self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping) + "\n")
                output_gz_file_handler.flush()
            except Exception as E:
                errmsg = ["[ERROR]: input tar.gz : " + input_tar_gz_file_path]
                errmsg += ["TYPE  : could not create output file :" + output_errlog_file_path]
                errmsg += ["Error details: " + str(E)]
                self.lp(errmsg)
                errlp(errmsg)
                this_input_logger.close()
                if output_gz_file_handler is not None:
                    try:
                        output_gz_file_handler.close()
                    except:
                        pass
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 4: process input tar.gz, write into output .gz and possibly into .ann files
            try:
                self.__process_one_input_tar_gz(input_tar_gz_file_path, output_gz_file_handler, output_ann_folder, errlp)
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                errlp("<<<END_OF_PROCESSING>>>")
                try:
                    output_gz_file_handler.close()
                    this_input_logger.close()
                except:
                    pass

            except Exception as E:
                #dont use finally here to wait it out until the loop is done ...
                errmsg = ["[ERROR]: input tar.gz : " + input_tar_gz_file_path]
                errmsg += ["Error details: " + str(E)]
                self.lp(errmsg)
                errlp(errmsg)
                try: #don't wait this out with finally till the loop ends, do it now here!
                    output_gz_file_handler.close()
                    this_input_logger.close()
                except:
                    pass
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

        #finalize work ...
        self.__rm_directory_with_content(self.temp_folder_path)
        self.lp("<<<END_OF_PROCESSING_ALL_BATCHES>>>")


    def __process_one_input_tar_gz(self, input_tar_gz_file_path, output_gz_file_handler, output_ann_folder, errlp):
        try:
            anns = []
            txts = []
            tar = None
            tar = tarfile.open(input_tar_gz_file_path , "r:gz", encoding='utf-8')
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                file_name = member.name.split("/")[-1]
                if file_name.endswith(".ann"):
                    anns.append(file_name.split(".ann")[0])
                elif file_name.endswith(".txt"):
                    txts.append(file_name.split(".txt")[0])

            anns = set(anns)
            txts = set(txts)
            if len(anns) != len(txts):
                anns_minues_txts = anns - txts
                anns_minues_txts_len = len(anns_minues_txts)
                if anns_minues_txts_len > 0:
                    raise Exception("There are " + str(anns_minues_txts_len) + " .ann files with no corresponding .txt :" + str(sorted(anns_minues_txts)))

                txts_minues_anns = txts - anns
                txts_minues_anns_len = len(txts_minues_anns)
                if txts_minues_anns_len > 0:
                    raise Exception("There are " + str(txts_minues_anns_len) + " .txt files with no corresponding .ann :" + str(sorted(txts_minues_anns)))

            if len(anns) < 1:
                raise Exception("There are zero ann files in the input file.")

            for brat_file in sorted(anns):
                try:
                    # 1: extract .ann+.txt from .tar.gz into temp
                    ann_file_path = self.temp_folder_path + brat_file + ".ann"
                    txt_file_path = self.temp_folder_path + brat_file + ".txt"
                    tar.extract("./" + brat_file + ".ann" , self.temp_folder_path)
                    tar.extract("./" + brat_file + ".txt" , self.temp_folder_path)

                    # 2: convert brat to json (json will be in memory)
                    json_data = self.brat_json_converter.convert_brat_to_json(txt_file_path,
                                                                              ann_file_path,
                                                                              output_file_json=None,
                                                                              all_event_types=[],
                                                                              encoding="utf-8")

                    # 2-2: remove physical temporary files
                    os.remove(txt_file_path)
                    os.remove(ann_file_path)

                    # 3: writeback entities into output .ann file if requested
                    if self.create_output_ann_files:
                        output_ann_file_path = output_ann_folder + brat_file + ".ann"
                        output_ann_file_handler = self.__write_back_json_entities_into_brat_ann(json_data, output_ann_file_path)

                    # 4: convert json to json_examples (in memory operation with considering maximum sentence distance)
                    json_data, counter_annotated_positives, counter_generated_positives, counter_generated_negatives = self.example_generator.generate_examples(
                        input_json_fileaddress=None,
                        output_json_fileaddress=None,
                        input_json_conent=json_data,
                        dont_generate_negatives_if_sentence_distance_ge=self.dont_generate_negatives_if_sentence_distance_ge)

                    # 5: generate ann_inputs
                    pair_tracking , ANN_inputs , ANN_outputs = self.ann_input_output_generator.generate_ANN_Input_Outputs_pairs(
                        json_data,
                        generate_output=True,
                        strategy=self.__representation_strategy)

                    # 6: skip file if no candidate relation is found
                    if pair_tracking is None:
                        if self.create_output_ann_files:
                            output_ann_file_handler.close()
                        continue

                    """
                    # 7: temp ... delete me later ...
                    print (ann_file_path , len(pair_tracking))
    
                    if brat_file != "19140000":
                        output_ann_file_handler.close()
                        continue
                    """

                    # 7: predict outputs with ANN
                    pred_confidence_scores = self.model.predict(ANN_inputs)
                    #self.results = pred_confidence_scores
                    #self.pair_tracking = pair_tracking

                    # 8: write predictions to file(s)
                    if self.configs['classification_type'] in ["binary", "multi-class"]:
                        relation_id_tracker = 0
                        for index , item in enumerate(zip(pair_tracking, pred_confidence_scores)):
                            document_id, example_id, example_e1, example_e2, _this_example_tokens_indices , _this_example_tokens_offsets = item[0]
                            pred_confidence = item[1]
                            predicted_relation_type = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[np.argmax(pred_confidence)]

                            # Take directionality of predicted relation into account
                            # 1) For non-directional relation types (e.g. 'Complex_formation'), we don't need to look at the order of e1 and e2.
                            # 2) For directional relation types (e.g., Regulation), if the direction is > (e.g., "Regulation>"), we don't need to change the ordger of e1 and e2.
                            #    It is predicted that direction is from e1 to e2. So we can write in the .ann file : R108 Regulation Arg1:e1 Arg2:e2
                            # 3) If direction is < , we need to swap e1 and e2, and write R108 Regulation Arg1:e2 Arg1:e1
                            if predicted_relation_type[-1] == ">":
                                predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation>' --> 'Regulation'
                            elif predicted_relation_type[-1] == "<":
                                predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation<'  --> 'Regulation'
                                example_e1, example_e2 = example_e2, example_e1

                            # write the predictions to .gz file, regardless of being positive or negative ...
                            gz_output_str = "\t".join([brat_file, example_e1 , example_e2, predicted_relation_type, str(pred_confidence.tolist())]) + "\n"
                            output_gz_file_handler.write(gz_output_str)
                            output_gz_file_handler.flush()

                            # skip writing negatives to .ann file
                            if predicted_relation_type == 'neg':
                                continue

                            # only write a positive prediction to output .ann if it is requested
                            if self.create_output_ann_files:
                                relation_id_tracker += 1
                                ann_output_str = 'R' + str(relation_id_tracker) + "\t" + predicted_relation_type + " Arg1:" + example_e1 + " Arg2:" + example_e2 + "\n"
                                output_ann_file_handler.write(ann_output_str)

                    if self.create_output_ann_files:
                        output_ann_file_handler.close()

                except Exception as E:
                    msg = ["--------- [ERROR] --------"]
                    msg += ["ann : " + brat_file + ".ann"]
                    msg += ["txt : " + brat_file + ".txt"]
                    msg += ["Error : " + str(E)]
                    errlp(msg)
                    continue

            if tar is not None:
                if not tar.closed:
                    tar.close()

        except Exception as E:
            if tar is not None:
                if not tar.closed:
                    tar.close()
            raise Exception("error in __process_one_input_tar_gz: " + str(E))

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
                file_handle.flush()
        return file_handle

