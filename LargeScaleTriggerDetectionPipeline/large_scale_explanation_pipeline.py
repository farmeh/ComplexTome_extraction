"""
This is the code for large-scale automatic trigger detection on a cluster of gpu-s like puhti.
"""

import os
import sys
import shutil
import gzip
import tarfile
import datetime
import numpy as np
from collections import OrderedDict

import large_scale_pt_rel_model_loader_helper
import large_scale_explanation_helper

# Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))
# -------------------------------------------------------------------------
from helpers import logger
from helpers import configs_manager
from helpers import general_helpers as ge

class LargeScaleExplanationPipeline_torch(object):
    def __init__(self,
                 configs_file_path,
                 pretrained_model_path,
                 log_file_path,
                 input_folder_path,
                 output_folder_path,
                 work_only_on_positive_relations_in_input_ann, # True: look for valid relations, False: generate all possible pairs between all entities.
                 create_output_ann_files=False):

        #1. init local variables
        self.DEFAULT_REF_POSITION_ID_APPROACH = large_scale_pt_rel_model_loader_helper.RefPositionIdsApproach.MODEL_DEFAULT #based on the best approach, evaluated on devel
        self.DEFAULT_POSITIVE_CLASS_NAME = "Complex_formation" #needed for explanation
        self.DEFAULT_LAYER_IDX = 14 #based on the best approach, evaluated on devel <<<CRITICAL>>> don't change! this is based on the best approach.
        self.configs_file_path = configs_file_path
        self.pretrained_model_path = pretrained_model_path
        self.log_file_path = log_file_path
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.work_only_on_positive_relations_in_input_ann = work_only_on_positive_relations_in_input_ann
        self.create_output_ann_files = create_output_ann_files

        #2. check the path for log and configs files
        if os.path.isfile(log_file_path):
            print ("log file already exists. deleting and recreating the file :" + log_file_path)
            os.remove(log_file_path)
        if not os.path.isfile(configs_file_path):
            print("[ERROR] configs file not found: " + configs_file_path + "\nHALTING.")
            sys.exit(0)

        #3. create lp and configs objects
        self.logger = logger.Logger(log_file_path)
        self.lp = self.logger.lp
        self.configs = configs_manager.ConfigsManager(configs_file_path, self.lp, self.program_halt).configs

        #4. check input/output folders, add '/' to the end if needed
        self.__check_input_output_folders()

        #5. set temp folder path here
        self.temp_folder_path = self.output_folder_path + "tmp" + "_" + datetime.datetime.now().strftime("%Y%M%d_%H%M%S") + "/"
        self.__recreate_temp_folder()

        #6. log parameters
        msg_params = ["Running pipeline:", "-" * 80,
                      "\t- configs_file_path                            : " + str(self.configs_file_path),
                      "\t- pretrained_model_name_or_path                : " + str(self.pretrained_model_path),
                      "\t- log_file_path                                : " + str(self.log_file_path),
                      "\t- input_folder_path                            : " + self.input_folder_path,
                      "\t- output_folder_path                           : " + self.output_folder_path,
                      "\t- temp_folder_path                             : " + self.temp_folder_path,
                      "\t- work_only_on_positive_relations_in_input_ann : " + str(self.work_only_on_positive_relations_in_input_ann),
                      "\t- create_output_ann_files                      : " + str(self.create_output_ann_files),
                      "-" * 80]
        self.lp(msg_params)

        #7. load pretrained relation extraction model ...
        self.ls_pt_rel_model_helper = large_scale_pt_rel_model_loader_helper.LargeScaleTorchRelModelLoaderHelper(
            self.lp,
            self.configs,
            self.program_halt,
            self.program_halt_raise_exception_do_not_exit,
            self.exit,
            self.pretrained_model_path)

        #8. log other important params ...
        more_params_msg = ["Other important pipeline params:" ,
            "\t- DEFAULT_REF_POSITION_ID_APPROACH : " + str(self.DEFAULT_REF_POSITION_ID_APPROACH),
            "\t- DEFAULT_POSITIVE_CLASS_NAME      : " + str(self.DEFAULT_POSITIVE_CLASS_NAME),
            "\t- DEFAULT_LAYER_IDX                : " + str(self.DEFAULT_LAYER_IDX),
        ]
        self.lp(more_params_msg)

    def program_halt(self, message):
        #THIS WILL HALT and CRASH (runs self.exit function) THE WHOLE PIPELINE
        self.logger.lp_halt(message)
        self.exit()

    def program_halt_raise_exception_do_not_exit(self, message):
        #THIS WILL NOT HALT, but just raises exception, to be caught, and then we will proceed to the next input file ...
        #THIS WILL BE PASSED FOR EXAMPLE to brat_json_converter etc so that bad .ann+.txt files don't crash the whole pipeline, and we skip to the next file
        raise Exception(message)

    def exit(self):
        if self.logger.is_open():
            self.lp ("EXITING PROGRAM ... ")
            self.logger.close()
        sys.exit(0)

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

    def __recreate_temp_folder(self):
        try:
            if shutil.os.path.exists(self.temp_folder_path):
                shutil.rmtree(self.temp_folder_path)
            shutil.os.makedirs(self.temp_folder_path)
        except Exception as E:
            self.program_halt('unable to create temp folder here :' + self.temp_folder_path + "\nError: " + str(E))

    def __mkdir(self, folder_address):
        if not shutil.os.path.exists(folder_address):
            try:
                shutil.os.makedirs(folder_address)
            except Exception as E:
                self.program_halt("could not create folder : " + folder_address + "\nerror: " + str(E))

    def __rm_directory_with_content(self,folder_address):
        if shutil.os.path.exists(folder_address):
            try:
                shutil.rmtree(folder_address)
            except Exception as E:
                self.program_halt("could not remove folder : " + folder_address + "\nerror: " + str(E))

    def run_large_scale_pipeline(self):
        #PROCESS ALL tar.gz files in a given input folder ...
        for input_tar_gz_file_path in sorted(ge.get_all_files_with_extension(self.input_folder_path , "tar.gz")):
            self.__recreate_temp_folder()
            date_time_start = datetime.datetime.now()
            self.lp ("input tar.gz: " + input_tar_gz_file_path)
            input_tar_gz_file_name = input_tar_gz_file_path.split("/")[-1]
            output_gzip_file_path = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + ".outtsv.gz"
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
                errmsg = "[ERROR#1]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create output ann folder :" + output_ann_folder + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
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
                errmsg = "[ERROR#2]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create log file :" + output_errlog_file_path + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
                self.lp (errmsg)
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 3: create output_gz file handler
            try:
                output_gz_file_handler = None
                output_gz_file_handler = gzip.open(output_gzip_file_path, "wt", encoding='utf-8')
                # %TODO: write file header here ....
                #write confidence score indices of the task , e.g. : {0: 'neg', 1: 'Complex_formation'}
                #output_gz_file_handler.write("# confidence indices:\t" + str(self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping) + "\n")
                #output_gz_file_handler.flush()
            except Exception as E:
                errmsg = "[ERROR#3]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create output file :" + output_gzip_file_path + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
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
                errmsg = "[ERROR#4]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: processing input tar.gz file.\t" + \
                         "[MORE_DETAILS]: " + str(E)
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
        self.lp("<<<END_OF_PROCESSING_ALL_BATCHES>>>")
        self.__rm_directory_with_content(self.temp_folder_path) #delete tmp folder ...

    def __process_one_input_tar_gz(self, input_tar_gz_file_path, output_gz_file_handler, output_ann_folder, errlp):
        #check integrity of the input .tar.gz file: check .ann+.txt
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

        except Exception as E:
            if tar is not None:
                if not tar.closed:
                    tar.close()
            raise Exception(str(E))

        #now let's unpack each brat doc (.ann+.txt pair) from THIS INPUT tar.gz file and process one-by-one ...
        for brat_file in sorted(anns):
            try: #if an exception happens in this try, we don't raise an exception for the WHOLE .tat.gz, but we simply skip this brat.
                # We log into corresponding log file of the .tar.gz and continue with the next brat document ...

                # 1: extract .ann+.txt from .tar.gz into temp
                ann_file_path = self.temp_folder_path + brat_file + ".ann" #to hold file-name for later use after extraction from .tar
                txt_file_path = self.temp_folder_path + brat_file + ".txt" #to hold file-name for later use after extraction from .tar
                tar.extract("./" + brat_file + ".ann", self.temp_folder_path)
                tar.extract("./" + brat_file + ".txt", self.temp_folder_path)
                txt_file_content = self.__lowlevel_read_txt_file_content(txt_file_path)

                #2: encode document: brat -> json -> json_examples -> pair_tracking ...
                pair_tracking, ann_inputs, ann_outputs = self.ls_pt_rel_model_helper.encode_brat_document(
                       txt_file_path,
                       ann_file_path,
                       return_only_positive_examples=self.work_only_on_positive_relations_in_input_ann)

                os.remove(ann_file_path)
                os.remove(txt_file_path)

                if pair_tracking is None:
                    raise Exception ("empty pair_tracking.")

                #3: create empty .ann file if requested ...
                if output_ann_folder is not None:
                    output_ann_file_handler = open(output_ann_folder + brat_file + ".ann", "wt", encoding='utf-8')
                else:
                    output_ann_file_handler = None

                #4: decode (preparing for Captum here)
                all_candidate_examples_in_ann = []
                for idx, item_info in enumerate(pair_tracking):
                    doc_id, rel_id, e1_id, e2_id, this_example_tokens_indices, this_example_tokens_offsets = item_info
                    input_token_ids = ann_inputs[0][idx].tolist()
                    while input_token_ids[-1] == self.ls_pt_rel_model_helper.torch_helper.tokenizer.pad_token_id:
                        input_token_ids = input_token_ids[:-1]

                    # remove CLS and SEP tokens, cause they will be added later:
                    if input_token_ids[0] == self.ls_pt_rel_model_helper.torch_helper.tokenizer.cls_token_id:
                        input_token_ids = input_token_ids[1:]

                    if input_token_ids[-1] == self.ls_pt_rel_model_helper.torch_helper.tokenizer.sep_token_id:
                        input_token_ids = input_token_ids[:-1]

                    decoded_input_text = self.ls_pt_rel_model_helper.torch_helper.tokenizer.decode(input_token_ids, clean_up_tokenization_spaces=False) #<<<CRITICAL>>>

                    """
                    clean_up_tokenization_spaces=False is very needed, because if we don't do that, there encode-decode will not be correct. example:
                    >>> t.decode(t('. .').input_ids)
                        '..'
                        t.decode(t('. .').input_ids ,  clean_up_tokenization_spaces=False)
                        '. .' 
                    """

                    item = OrderedDict({
                        'pmid': ann_file_path.split("/")[-1].split(".ann")[0],
                        'rel_id': rel_id,
                        'e1_id': e1_id,
                        'e2_id': e2_id,
                        'decoded_input_text': decoded_input_text,
                        'orig_tokens_indices': this_example_tokens_indices,
                        'orig_tokens_offsets': this_example_tokens_offsets,
                        'orig_tokens_values': self.ls_pt_rel_model_helper.torch_helper.tokenizer.convert_ids_to_tokens(this_example_tokens_indices),
                    })
                    all_candidate_examples_in_ann.append(item)

                #<<<CRITICAL>>> this is for debugging the inputs ...
                #for item in all_candidate_examples_in_ann:
                #    self.lp ("[DECODED_TEXT]\t" + item['pmid']+"_"+item['e1_id']+"_"+item['e2_id']+"\t"+str([item['decoded_input_text']]))

                #5: predict labels for all of them here
                #<<<CRITICAL>>> this is batch prediction
                #all_input_texts = [e['decoded_input_text']for e in all_candidate_examples_in_ann]
                #all_predictions_for_inputs = self.ls_pt_rel_model_helper.predict_list_of_str_calling_model(all_input_texts, return_logits=True).tolist()

                #<<<CRITICAL>>> this is one-by-one prediction
                all_predictions_for_inputs = []
                for item in all_candidate_examples_in_ann:
                    tmp_scores = self.ls_pt_rel_model_helper.old_pipeline_predict_one_example(item['decoded_input_text'], self.DEFAULT_REF_POSITION_ID_APPROACH)
                    all_predictions_for_inputs.append(tmp_scores)

                #6: process each example:
                # 6.1 negatives --> write to tsv (but not into ANN)
                # 6.2 positives --> detect trigger AND write to tsv/ANN
                this_file_trigger_index = 0
                for index, item in enumerate(zip(all_candidate_examples_in_ann, all_predictions_for_inputs)):
                    this_example_info = item[0]
                    this_example_confidence_scores = item[1]
                    neg_score , pos_score = this_example_confidence_scores[0] , this_example_confidence_scores[1]

                    #self.__write_into_tsv_gz(output_gz_file_handler,
                    #                         this_example_info['pmid'],
                    #                         this_example_info['e1_id'],
                    #                         this_example_info['e2_id'],
                    #                         neg_score, pos_score)
                    #continue

                    if neg_score > pos_score: #NEGATIVE
                        #just write into .tsv.gz , there is no trigger to write into .ann, even if requested
                        self.__write_into_tsv_gz(output_gz_file_handler,
                                                 this_example_info['pmid'],
                                                 this_example_info['e1_id'],
                                                 this_example_info['e2_id'],
                                                 neg_score, pos_score)
                    else: #POSITIVE
                        #1. explain
                        explanation_results = self.ls_pt_rel_model_helper.explain(
                            this_example_info['decoded_input_text'],
                            focus_class_label = self.DEFAULT_POSITIVE_CLASS_NAME ,
                            layer_idx = self.DEFAULT_LAYER_IDX ,
                            ref_position_ids_approach = self.DEFAULT_REF_POSITION_ID_APPROACH
                        )
                        this_example_info["tokens_values"] = explanation_results["tokens"]
                        this_example_info["pred_value"] = this_example_confidence_scores
                        this_example_info["output_attrs_sum"] = explanation_results["output_attrs_sum"]

                        #<<<DEBUG>>> this was for checking to see if we're getting the same tokens and attributions score from Captum. We do as before.
                        #self.lp ("[DECODED_INFO]\t" +
                        #         str(this_example_info['pmid'])+"_"+this_example_info['e1_id']+"_"+this_example_info['e2_id'] + "\t" +
                        #         str([this_example_info["tokens_values"]]) + "\t"+
                        #         str([this_example_info["output_attrs_sum"]]))

                        #2. check if answer is valid (do tokens match):
                        # if not valid, do not skip the whole brat doc, but just skip this example and proceed to the next ...
                        this_example_explanation, convertion_error = self.ls_pt_rel_model_helper.check_example_validity_convert_to_ExplanationExample(this_example_info)

                        if this_example_explanation is None:
                            #1. write as a negative into the output .tsv file
                            self.__write_into_tsv_gz(output_gz_file_handler,
                                                     this_example_info['pmid'],
                                                     this_example_info['e1_id'],
                                                     this_example_info['e2_id'],
                                                     neg_score, pos_score)

                            #2. log into the local file
                            errmsg = "[WARN]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                                     "[WARN_ANN] : " + brat_file + ".ann \t" + \
                                     "[WARN_DETAILS]: token_mismatch_after_conversion\t" + \
                                     "[e1_e2]:(" + this_example_info['e1_id'] + "," +this_example_info['e2_id'] + ")\t" + \
                                     convertion_error
                            errlp(errmsg)
                            continue

                        #3. find the best trigger word here ...
                        best_trigger_selector = large_scale_explanation_helper.FilterJunkSelectBestTriggers(this_example_explanation, 0 , txt_file_content)
                        selected_best_triggers_info, selection_error = best_trigger_selector.get_best_triggers()
                        if selected_best_triggers_info is None:
                            #1. write as a negative into the output .tsv file
                            self.__write_into_tsv_gz(output_gz_file_handler,
                                                     this_example_info['pmid'],
                                                     this_example_info['e1_id'],
                                                     this_example_info['e2_id'],
                                                     neg_score, pos_score)

                            #2. log into the local file
                            errmsg = "[ERROR]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                                     "[ERROR_ANN] : " + brat_file + ".ann \t" + \
                                     "[ERROR_DETAILS]: error in best trigger finding.\t" + \
                                     "[e1_e2]:(" + this_example_info['e1_id'] + "," +this_example_info['e2_id'] + ")\t" + \
                                     selection_error
                            errlp(errmsg)
                            continue

                        #4. found everything, now let's write them ...
                        for item in selected_best_triggers_info:
                            span_bgn, span_end, span_txt, span_score = item
                            # 1. write as a negative into the output .tsv file
                            self.__write_into_tsv_gz(output_gz_file_handler,
                                                     this_example_info['pmid'],
                                                     this_example_info['e1_id'],
                                                     this_example_info['e2_id'],
                                                     neg_score, pos_score,
                                                     span_score, span_bgn, span_end, span_txt)

                            # 2. write in output .ann file ...
                            if output_ann_file_handler is not None:
                                this_file_trigger_index += 1
                                self.__lowlevel_write_trigger_to_ann(output_ann_file_handler, this_file_trigger_index, span_bgn, span_end, span_txt)
                                output_ann_file_handler.flush()

                #7. close output_ann_file_handler
                if output_ann_file_handler is not None:
                    try:
                        output_ann_file_handler.close()
                    except:
                        pass

            except Exception as E:
                #e1: log into local .tar.gz log-file
                errmsg = "[ERROR]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_ANN] : " + brat_file + ".ann \t" + \
                         "[ERROR_DETAILS] :" + str(E)
                errlp(errmsg)

                #e2: close output ann_file_handler if is open
                if output_ann_file_handler is not None:
                    try:
                        output_ann_file_handler.flush()
                        output_ann_file_handler.close()
                    except:
                        pass

                continue

    def __write_into_tsv_gz(self, output_gz_file_handler, pmid, e1_id, e2_id, negative_pred_score, positive_pred_score, trigger_detection_score = '',trigger_bgn_offset = '', trigger_end_offset = '',trigger_text=''):
        try:
            gz_output_str = "\t".join([pmid , e1_id , e2_id ,
                                       str(negative_pred_score) ,
                                       str(positive_pred_score),
                                       str(trigger_detection_score) ,
                                       str(trigger_bgn_offset) ,
                                       str(trigger_end_offset),
                                       str(trigger_text)]) + "\n"
            output_gz_file_handler.write(gz_output_str)
            output_gz_file_handler.flush()
        except Exception as E:
            raise Exception ("cannot write into tsv.gz file: " + str(E)) #this will be caught in the main for loop and written into the logs, and then we proceed into the next .ann/.txt file

    #for explanation, when actual writing of trigger word into the .ann or .tsv file ...
    def __lowlevel_read_txt_file_content(self, txt_filepath):
        # read .txt file for example to get different information ...
        try:
            with open(txt_filepath, "rt" , encoding='utf-8') as f:
                file_content = f.read()
                return file_content
        except Exception as E:
            raise Exception ("cannot read .txt file " + txt_filepath , " [READ_ERROR]: " + str(E))

    def __lowlevel_write_trigger_to_ann(self, output_ann_file_handler , trigger_index , bgn_offset, end_offset, trigger_text):
        #<<<CRITICAL>>> there is no check for example if there is a "\n" in the trigger_text or if trigger_text is just a '\t' or ' '
        # because those are for sure filtered prior to finding best triggers ...
        output = 'T' + str(trigger_index) + "\tTrigger " + str(bgn_offset) + " " + str(end_offset) + "\t" + trigger_text + "\n"
        output_ann_file_handler.write(output)
        output_ann_file_handler.flush() #<<<CRITICAL>>> don't remove if you need a solid evaluation.
