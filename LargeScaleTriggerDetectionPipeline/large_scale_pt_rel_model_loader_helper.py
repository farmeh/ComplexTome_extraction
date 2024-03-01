import json
import os
import sys
from enum import Enum
import numpy as np
import torch
import transformers
from datasets import Dataset
from captum.attr import LayerIntegratedGradients
import transformers.models.roberta.modeling_roberta as RobertaModeling

import large_scale_explanation_helper
# Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))
# -------------------------------------------------------------------------

from helpers import pipeline_variables #needed for eval
from helpers import pt_model_helper
from helpers import brat_json_converter
from helpers import example_generation_cross_sentence_MD
from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator

class RefPositionIdsApproach(Enum):
    ROBERTA_IF_POSSIBLE = 1
    ALL_ZEROS = 2
    RANDPERM = 3
    MODEL_DEFAULT = 4

class LargeScaleTorchRelModelLoaderHelper:
    def __init__(self,
                 lp,
                 configs,
                 program_halt,
                 program_halt_raise_exception_do_not_exit,
                 exit,
                 pretrained_re_model_folder_path,
                ):

        #1. init variables
        self.lp = lp
        self.configs = configs
        self.program_halt = program_halt
        self.program_halt_raise_exception_do_not_exit = program_halt_raise_exception_do_not_exit
        self.exit = exit
        self.pretrained_re_model_folder_path = pretrained_re_model_folder_path

        #2. create other variables, load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lp ("torch device: " + str(self.device))
        self.__load_re_model_from_folder()
        self.is_roberta_model = "roberta" in str(self.torch_helper.pretrained_model).lower()  #for explanation
        self.model_padding_idx = self.torch_helper.pretrained_model.base_model.embeddings.padding_idx   #for explanation

    def __load_re_model_from_folder(self):
        if self.pretrained_re_model_folder_path[-1] != "/":
            self.pretrained_re_model_folder_path += "/"

        self.lp("-"*40 + " LOADING MODEL " + "-"*40)
        self.lp("\t1. Checking __pretrained_re_model_folder_path.")
        if not os.path.isdir(self.pretrained_re_model_folder_path):
            self.program_halt("invalid path given for pretrained_re_model_folder_path :" + self.pretrained_re_model_folder_path)

        self.lp("\t2. loading info.json from the folder.")
        try:
            info_json_file_path = self.pretrained_re_model_folder_path + "info.json"
            with open(info_json_file_path, "rt", encoding='utf-8') as f:
                self.__info_json_file_content = json.load(f)
            self.lp(["\tinfo.json: " + os.path.abspath(info_json_file_path),
                     "\tfile contents:", "-" * 20,
                     json.dumps(self.__info_json_file_content, indent=4, sort_keys=True), "-" * 80])
            self.max_seq_len = int(self.__info_json_file_content['max_seq_len'])
            self.representation_strategy = eval("pipeline_variables." + self.__info_json_file_content['representation_strategy'])
        except Exception as E:
            self.program_halt("Error: " + str(E))

        self.lp("\t3. loading and building model.")
        #<<<CRITICAL>>> passing program_halt_raise_exception_do_not_exit instead of program_halt, so that the pipeline just raises an exception and not totally crash
        self.torch_helper = pt_model_helper.PT_helper(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs, self.pretrained_re_model_folder_path, max_seq_len=self.max_seq_len)
        self.torch_helper.build_load_trainer_for_prediction()
        self.brat_json_converter = brat_json_converter.brat_json_Converter(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs)
        self.example_generator = example_generation_cross_sentence_MD.example_generator(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs)
        self.ann_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs, self.torch_helper)

        self.torch_helper.pretrained_model.eval() #<<<CRITICAL>>> THIS IS NEEDED TO TURN OFF DROPOUT and OTHER REGULARIZATIONS
        #self.torch_helper.pretrained_model.zero_grad() #<<<CRITICAL>>> DO NOT DO THIS.. SHAP/CAPTUM REQUIRE GRAD

    def encode_brat_document(self, txt_file_path, ann_file_path, return_only_positive_examples):
        try:
            json_data = self.brat_json_converter.convert_brat_to_json(txt_file_path, ann_file_path, output_file_json=None, all_event_types=[], encoding="utf-8")
            json_data, counter_annotated_positives, counter_generated_positives, counter_generated_negatives = self.example_generator.generate_examples(
                input_json_fileaddress=None,
                output_json_fileaddress=None,
                input_json_conent=json_data,
                dont_generate_negatives_if_sentence_distance_ge=None) #<<<CRITICAL>>> set this to None to always try to generate anything. in our case, it is not important cause seq_len=128 and we're just processing those examples which are predicted to be positive ...

            if return_only_positive_examples:
                if counter_generated_positives < 1:
                    return None, None, None

            if counter_generated_positives + counter_generated_negatives < 1:
                return None, None, None

            pair_tracking, ann_inputs, ann_outputs = self.ann_input_output_generator.generate_ANN_Input_Outputs_pairs(
                json_data,
                generate_output=True,
                strategy=self.representation_strategy)

            if pair_tracking is None:  # because of max_seq_len, maybe some examples are discarded. Then, a situation may arise that pair_tracking is None.
                return None, None, None

            if return_only_positive_examples:
                return self.__return_positives(pair_tracking, ann_inputs, ann_outputs)
            else:
                return pair_tracking, ann_inputs, ann_outputs

        except Exception as E:
            the_ann_file_name = ann_file_path.split("/")[-1]
            errmsg = "error in encode_brat_document. ann: " + the_ann_file_name + " .[INTERNAL_DETAIL] :" + str(E)
            raise Exception(errmsg)

    def __return_positives(self, pair_tracking, ann_inputs, ann_outputs):
        classification_type = self.configs['classification_type']
        if (not isinstance(ann_inputs, list)) or (not isinstance(ann_outputs, list)):
            self.program_halt('error in return_positives. ann_inputs and ann_outputs should be lists.')

        y = ann_outputs[0]
        if classification_type in ['binary', 'multi-class']:  # there is a negative dimension in the decision layer for neg class.
            neg_decision_layer_index = self.configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['neg']  # which index in decision_layer (i.e., ann_output) denotes the negative.
            all_positive_indices_in_y = np.where(y[:, neg_decision_layer_index] == 0)[0]  # find rows in which the neg_element is zero!
        else:  # multi-label
            all_positive_indices_in_y = np.where(y.any(axis=1))[0]  # find rows in which at least there is a non-zero element

        if len(all_positive_indices_in_y) == 0:
            return None, None, None

        _p = [pair_tracking[i] for i in all_positive_indices_in_y]
        _ann_inputs, _ann_outputs = [], []
        for item in ann_inputs:
            _ann_inputs.append(item[all_positive_indices_in_y])
        for item in ann_outputs:
            _ann_outputs.append(item[all_positive_indices_in_y])
        return _p, _ann_inputs, _ann_outputs

    def predict_ann_file(self, txt_file_path, ann_file_path):
        pair_tracking, ann_inputs, ann_outputs = self.encode_brat_document(txt_file_path, ann_file_path)
        if pair_tracking is None:
            return None, None, None, None

        y = self.torch_helper.predict(ann_inputs)
        return y, pair_tracking, ann_inputs, ann_outputs

    def predict_list_of_str_calling_model(self, input_sentences, return_logits=False, **tokenizer_kwargs):
        if tokenizer_kwargs == {}:
            tokenizer_kwargs['padding']=True
        inputs_pt_tensors = self.torch_helper.tokenizer(input_sentences, return_tensors='pt', **tokenizer_kwargs).to(self.device)
        with torch.no_grad():
             logits = self.torch_helper.pretrained_model(**inputs_pt_tensors).logits
        if return_logits:
            return logits.cpu().detach().numpy() #<<<CRITICAL>>> added .cpu() before detach to run pn GPU
        if self.configs['classification_type'] in ('binary' , 'multi-class'):
            return torch.softmax(logits, axis=1).cpu().detach().numpy()
        else: #multi-label
            return torch.sigmoid(logits).cpu().detach().numpy()

    def predict_list_of_str_calling_trainer(self, input_sentences, return_logits=False, **tokenizer_kwargs):
        if tokenizer_kwargs == {}:
            tokenizer_kwargs['padding']=True
        inputs_pt_tensors = self.torch_helper.tokenizer(input_sentences, return_tensors='pt', **tokenizer_kwargs).to(self.device)
        inputs_hf_dataset = Dataset.from_dict(inputs_pt_tensors)
        np_logits = self.torch_helper.trainer.predict(inputs_hf_dataset).predictions #this is numpy, not a cpu/gpu tensor
        if return_logits:
            return np_logits
        if self.configs['classification_type'] in ('binary', 'multi-class'):
            return torch.softmax(torch.tensor(np_logits), axis=1).cpu().detach().numpy()
        else: #multi-label
            torch.sigmoid(torch.tensor(np_logits)).cpu().detach().numpy()

    def predict_list_of_str_calling_pipeline(self, input_sentences, return_logits=False, return_all_scores=True):
        pred_pipeline = transformers.pipeline("text-classification", model=self.torch_helper.pretrained_model, tokenizer=self.torch_helper.tokenizer, return_all_scores=True)
        if return_logits:
            return pred_pipeline(input_sentences, return_all_scores, function_to_apply='none') #scalar values
        else:
            return pred_pipeline(input_sentences, return_all_scores) #scalar values


    #------------------ MODEL EXPLANATION FUNCTIONS -------------------------------------------------------------------#
    # Forward on the model: data in, prediction out
    def lowlevel_forward_full_embeddings(self, inputs_embeds, attention_mask):
        pred = self.torch_helper.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # pred --> #SequenceClassifierOutput(loss=None, logits=tensor([[-8.0835,  7.3970]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
        # pred[0] --> tensor([[ -8.0835,  7.3970]], grad_fn=<AddmmBackward>)
        return pred[0]

    # Given input text, construct a pair of (text input, blank reference input as long as the text itself)
    def construct_input_ref_indices_tensors(self, text, ref_position_ids_approach:RefPositionIdsApproach):
        """
        For 1 input, model() needs the inputs to be like a 2D-tensor with shape [1, seq_length]
        e.g,  input_ids = tensor([[1,2,3]]) , token_type_ids = tensor([[0,0,0]]) , attention_mask = tensor([[1,1,1]]) , position_ids = tensor([[0,1,2]])
        This is exactly matching with the tokenizer(text, return_tensors='pt') output
        """
        results = {
            "input": {"input_ids": None, "token_type_ids": None, "attention_mask": None, "position_ids": None},
            "ref"  : {"input_ids": None, "token_type_ids": None, "attention_mask": None, "position_ids": None},
        }
        #text

        #input tensors
        tokenizer_call_outputs_pt = self.torch_helper.tokenizer(text, return_token_type_ids=True, return_attention_mask=True, return_tensors='pt')
        seq_len_with_added_cls_sep_integer = tokenizer_call_outputs_pt['input_ids'].shape[1]
        results["input"]["input_ids"] = tokenizer_call_outputs_pt['input_ids'].to(self.device)
        results["input"]["token_type_ids"] = tokenizer_call_outputs_pt['token_type_ids'].to(self.device)
        results["input"]["attention_mask"] = tokenizer_call_outputs_pt['attention_mask'].to(self.device)
        if self.is_roberta_model:
            results["input"]["position_ids"] = RobertaModeling.create_position_ids_from_input_ids(results['input']['input_ids'], self.model_padding_idx).to(self.device)
        else:
            results["input"]["position_ids"] = torch.arange(seq_len_with_added_cls_sep_integer, dtype=torch.long).unsqueeze(0).to(self.device)

        #reference (blank) tensors
        ref_text = self.torch_helper.tokenizer.pad_token * (seq_len_with_added_cls_sep_integer - 2) # '<pad><pad>...<pad>' without CLS or SEP tokens
        ref_tokenizer_call_outputs_pt = self.torch_helper.tokenizer(ref_text, return_token_type_ids=True, return_attention_mask=True, return_tensors='pt') #--> '[CLS]<pad><pad>...<pad>[SEP]'
        results["ref"]["input_ids"] = ref_tokenizer_call_outputs_pt['input_ids'].to(self.device)
        results["ref"]["token_type_ids"] = ref_tokenizer_call_outputs_pt['token_type_ids'].to(self.device)
        results["ref"]["attention_mask"] = ref_tokenizer_call_outputs_pt["attention_mask"].to(self.device)
        if (self.is_roberta_model) and (ref_position_ids_approach == RefPositionIdsApproach.ROBERTA_IF_POSSIBLE):
            results["ref"]["position_ids"] = RobertaModeling.create_position_ids_from_input_ids(results['ref']['input_ids'], self.model_padding_idx).to(self.device)
        elif ref_position_ids_approach == RefPositionIdsApproach.ALL_ZEROS:
            results["ref"]["position_ids"] = torch.zeros(seq_len_with_added_cls_sep_integer, dtype=torch.long).unsqueeze(0).to(self.device)
        elif ref_position_ids_approach == RefPositionIdsApproach.RANDPERM:
            results["ref"]["position_ids"] = torch.randperm(seq_len_with_added_cls_sep_integer, dtype=torch.long).unsqueeze(0).to(self.device)
        elif ref_position_ids_approach == RefPositionIdsApproach.MODEL_DEFAULT:
            results["ref"]["position_ids"] = None #let the model build whatever is needed (defaults to something anyways)
        else: #model is not roberta but roberta was requested --> default to all_zeros
            results["ref"]["position_ids"] = torch.zeros(seq_len_with_added_cls_sep_integer, dtype=torch.long).unsqueeze(0).to(self.device)

        return results

    def construct_whole_bert_embeddings(self, text, ref_position_ids_approach:RefPositionIdsApproach):
        input_ref_indices_tensors = self.construct_input_ref_indices_tensors(text, ref_position_ids_approach)

        input_embd = self.torch_helper.pretrained_model.base_model.embeddings.forward(
                         input_ids=input_ref_indices_tensors['input']['input_ids'],
                         token_type_ids=input_ref_indices_tensors['input']['token_type_ids'],
                         position_ids=input_ref_indices_tensors['input']['position_ids'])

        if ref_position_ids_approach != RefPositionIdsApproach.MODEL_DEFAULT:
            ref_position_ids = input_ref_indices_tensors['ref']['position_ids']
        else:
            ref_position_ids = None

        ref_embd = self.torch_helper.pretrained_model.base_model.embeddings.forward(
                         input_ids=input_ref_indices_tensors['ref']['input_ids'],
                         token_type_ids=input_ref_indices_tensors['ref']['token_type_ids'],
                         position_ids=ref_position_ids)

        return{
            "input_embd" : input_embd,
            "ref_embd"    : ref_embd,
            "input_attention_mask" : input_ref_indices_tensors['input']['attention_mask'],
            "ref_attention_mask"   : input_ref_indices_tensors['ref']['attention_mask'],
        }


    def summarize_attributions(self, attributions):
        """
        1. attibution has a shape of [1, sequence_length, dimensionality_of_embedding_layer_for_each_token] , where sequence_length is basically number of tokens in the input.
           for example it can be [1, 71, 1024], assuming seq_len=71 tokens.
           therefore, sum(dim=-1) = sum(dim=2). This sums all the (1024) values for each token.

        2. squeeze(0) basically removes the extra dimension in the first dimentions
           so : attr.sum(dim=-1).shape --> [1,71]
                attr.sum(dim=-1).squeeze(0).shape --> [71]
        """
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def explain(self, text, focus_class_label, layer_idx, ref_position_ids_approach:RefPositionIdsApproach):
        # 1. get embeddings for input text and ref_text
        input_ref_indices_tensors = self.construct_input_ref_indices_tensors(text, ref_position_ids_approach) #TODO: do not call twice .. fix construct_whole_bert to get info from this
        input_ref_embeddings = self.construct_whole_bert_embeddings(text, ref_position_ids_approach)

        # 2. explain
        output_attrs_sum = []
        target_cls_idx = self.configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping[focus_class_label]
        lig = LayerIntegratedGradients(self.lowlevel_forward_full_embeddings, self.torch_helper.pretrained_model.base_model.encoder.layer[layer_idx])
        attrs, delta = lig.attribute(inputs=input_ref_embeddings['input_embd'],
                                     baselines=input_ref_embeddings['ref_embd'],
                                     additional_forward_args=(input_ref_embeddings['input_attention_mask'],),
                                     return_convergence_delta=True,
                                     target=target_cls_idx,
                                     n_steps=100)
        attrs_sum = self.summarize_attributions(attrs)
        output_attrs_sum.append({"layer_name": str(layer_idx), "attrs_sum": attrs_sum.cpu().detach().tolist()})

        # 3. get tokens
        list_of_integer_token_ids = input_ref_indices_tensors['input']['input_ids'][0].detach().tolist()
        list_of_str_tokens = self.torch_helper.tokenizer.convert_ids_to_tokens(list_of_integer_token_ids)
        # 4. return
        return {"tokens": list_of_str_tokens, "output_attrs_sum": output_attrs_sum}

    # ------------------ MODEL EXPLANATION UTILITIES-------------------------------------------------------------------#
    def check_example_validity_convert_to_ExplanationExample(self, src_example):
        try:
            assert len(src_example['orig_tokens_indices']) == len(src_example['orig_tokens_offsets']) == len(src_example['orig_tokens_values']) == len(src_example['tokens_values']), "assert_error_1"
            assert src_example['orig_tokens_values'] == src_example['tokens_values'], "assert_error_2"
            assert len(src_example['tokens_values']) == len(src_example['output_attrs_sum'][0]['attrs_sum']), "assert_error_3"
        except Exception as E:
            return None, str(E)

        example = large_scale_explanation_helper.ExplanationExample()
        example.type = large_scale_explanation_helper.ExplanationFileType.Captum
        example.ann = src_example['pmid']
        example.e1_id = src_example['e1_id']
        example.e2_id = src_example['e2_id']
        example.decoded_input_text = src_example['decoded_input_text']
        example.pred_value = src_example['pred_value']

        # 1. Get rid of CLS and SEP
        example.orig_tokens_values = src_example['orig_tokens_values'][1:-1]
        example.orig_tokens_offsets = src_example['orig_tokens_offsets'][1:-1]
        example.tokens_values = src_example['tokens_values'][1:-1]

        # 2.1 delete all_zero layers, i.e., add layer if it is NOT all_zero
        # 2.2 at the same time, delete non-relevant values, i.e., CLS and SEP from the begining and end of attr_sum
        layers_names, layers_np_vectors = [], []
        for item in src_example['output_attrs_sum']:
            layer_name = item['layer_name']
            layer_vector = item['attrs_sum'][1:-1]  # point 1
            layer_vector = np.array(layer_vector, dtype=np.double)
            if np.count_nonzero(layer_vector) > 0:
                layers_names.append(layer_name)
                layers_np_vectors.append(layer_vector)
        example.layers_names = layers_names
        example.layers_np_vectors = np.vstack(layers_np_vectors)

        try:
            number_of_layers, number_of_attributed_tokens = example.layers_np_vectors.shape
            assert len(example.orig_tokens_offsets) == number_of_attributed_tokens , "assert_error_4"
        except Exception as E:
            return None, str(E)

        return example, None


    # ------------------ MODEL EXPLANATION UTILITIES-------------------------------------------------------------------#
    def old_pipeline_predict_one_example(self, text, ref_position_ids_approach:RefPositionIdsApproach):
        input_ref_embeddings = self.construct_whole_bert_embeddings(text, ref_position_ids_approach)
        prediction_results = self.lowlevel_forward_full_embeddings(inputs_embeds=input_ref_embeddings['input_embd'], attention_mask=input_ref_embeddings['input_attention_mask'])
        prediction_results = prediction_results.tolist()
        return prediction_results[0]


