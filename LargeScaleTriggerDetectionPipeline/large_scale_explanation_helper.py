from enum import Enum
import string
import numpy as np
from collections import OrderedDict

CCWords = {'&', "'cause", "'n", "'n'", "'til", 'I', 'a', 'aboard', 'about', 'above', 'across', 'after', 'against', 'ago',
           'albeit', 'all', 'along', 'alongside', 'although', 'always', 'am', 'amid', 'among', 'amongst', 'an', 'and', 'any',
           'anybody', 'anyhow', 'anyone', 'anything', 'anytime', 'anyway', 'anywhere', 'are', 'around', 'as', 'astride', 'at',
           'atop', 'be', 'because', 'been', 'before', 'behind', 'being', 'below', 'beneath', 'beside', 'besides', 'between',
           'beyond', 'billion', 'billionth', 'both', 'but', 'by', 'can', 'cannot', 'could', 'de', 'despite', 'did', 'do',
           'does', 'doing', 'done', 'down', 'during', 'each', 'eight', 'eighteen', 'eighteenth', 'eighth', 'eightieth', 'eighty',
           'either', 'eleven', 'eleventh', 'en', 'enough', 'et', 'every', 'everybody', 'everyone', 'everything', 'everywhere',
           'except', 'few', 'fewer', 'fifteen', 'fifteenth', 'fifth', 'fiftieth', 'fifty', 'first', 'five', 'for', 'fortieth',
           'forty', 'four', 'fourteen', 'fourteenth', 'fourth', 'from', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
           'hers', 'herself', 'him', 'himself', 'his', 'how', 'hundred', 'hundredth', 'if', 'in', 'inside', 'into', 'is',
           'it', 'its', 'itself', 'least', 'less', 'lest', 'like', 'little', 'many', 'may', 'me', 'might', 'million',
           'millionth', 'mine', 'minus', 'more', 'most', 'much', 'must', 'my', 'myself', 'near', 'neither', 'never',
           'next', 'nine', 'nineteen', 'nineteenth', 'ninetieth', 'ninety', 'ninth', 'no', 'nobody', 'none', 'nor', 'not',
           'nothing', 'notwithstanding', 'now', 'nowhere', 'of', 'off', 'on', 'one', 'oneself', 'onto', 'opposite', 'or',
           'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'par', 'past', 'per', 'plus', 'post', 'second', 'seven',
           'seventeen', 'seventeenth', 'seventh', 'seventieth', 'seventy', 'shall', 'she', 'should', 'since', 'six',
           'sixteen', 'sixteenth', 'sixth', 'sixtieth', 'sixty', 'so', 'some', 'somebody', 'somehow', 'someone', 'something',
           'sometime', 'somewhere', 'ten', 'tenth', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
           'then', 'there', 'these', 'they', 'third', 'thirteen', 'thirteenth', 'thirtieth', 'thirty', 'this', 'those',
           'though', 'thousand', 'thousandth', 'three', 'through', 'throughout', 'till', 'times', 'to', 'too', 'toward',
           'towards', 'twelfth', 'twelve', 'twentieth', 'twenty', 'two', 'under', 'underneath', 'unless', 'unlike', 'until',
           'unto', 'up', 'upon', 'us', 'v.', 'versus', 'via', 'vs.', 'was', 'we', 'were', 'what', 'when', 'where', 'whereas',
           'whether', 'which', 'while', 'who', 'whom', 'whose', 'why', 'will', 'willing', 'with', 'within', 'without', 'worth',
           'would', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'zero'}

devel_trigger_words = {'sos recruitment system (srs) screen', 'direct protein-protein interaction', 'histone deacetylase components',
                       'tandem affinity purification', 'protein-protein interaction', 'direct physical interaction', 'binding regions between',
                       'co-immunoprecipitation', 'coimmunoprecipitation', 'two-hybrid screening', 'dimerization partner', 'physical interaction',
                       'direct interaction', 'heterodimerization', 'immunoprecipitates', 'fraction purified', 'yeast two-hybrid',
                       'counterreceptor', 'coprecipitated', 'heterotetramer', 'directly binds', 'three subunits', 'heterocomplex',
                       'dissociation', 'heterodimers', 'interactions', 'binding site', 'heterodimer', 'association', 'interaction',
                       'scaffolding', 'interacting', 'recruitment', 'recruiting', 'interacted', 'associates', 'associated', 'coactivate',
                       'component', 'recruited', 'complexes', 'same time', 'interacts', 'associate', 'fraction', 'contacts', 'isolated',
                       'receptor', 'interact', 'delivers', 'detected', 'subunit', 'complex', 'binding', 'engages', 'ligand', 'client',
                       'binds', 'bound', 'pair', 'bait', 'bind', '-', '.', '/'}

class ExplanationFileType(Enum):
    Captum = 1
    SHAP_WITH_CLS_SEP = 2
    SHAP_NO_CLS_SEP = 3

class ExplanationExample(object):
    pass

class FilterJunkSelectBestTriggers:
    def __init__(self, example, vector_index, corresponding_gold_txt_file_content):
        self.example = example
        self.vector_index = vector_index
        self.corresponding_gold_txt_file_content = corresponding_gold_txt_file_content
        self.vector = self.example.layers_np_vectors[self.vector_index].flatten()

        # define invalid punctuations: every punctuation that is not in devel
        self.invalid_punctuations = set(string.punctuation) - devel_trigger_words
        self.translate_punc_space_to_None_pattern = str.maketrans('', '', string.punctuation+ " ")

    def create_token_info_per_vector_index(self):
        res = OrderedDict()
        number_of_tokens = self.vector.shape[0]
        for this_token_index in range(0, number_of_tokens):
            span_bgn, span_end = self.example.orig_tokens_offsets[this_token_index]
            span_txt = self.corresponding_gold_txt_file_content[span_bgn:span_end]
            orig_token_value = self.example.orig_tokens_values[this_token_index]
            res[this_token_index] = (span_bgn , span_end, span_txt, orig_token_value)
        return res

    def discard_junk_tokens(self):
        all_tokens_indices = set(range(self.vector.shape[0])) #how many tokens in total
        junk_tokens_indices = set()
        last_token_index = self.vector.shape[0] - 1

        for this_token_index in self.token_info_per_vector_index.keys():
            span_bgn, span_end, span_txt, orig_token_value = self.token_info_per_vector_index[this_token_index]

            #discard spaces or other non-text tokens
            if span_bgn == span_end: # i.e, len(span_txt) == 0
                junk_tokens_indices.add(this_token_index)
                continue

            # discard pure spaces
            if len(span_txt) == 1:
                if (span_txt in self.invalid_punctuations) or span_txt == ' ':
                    junk_tokens_indices.add(this_token_index)
                    continue

            if ("\n" in span_txt) or ("\t" in span_txt):
                junk_tokens_indices.add(this_token_index)
                continue

            # discard entities
            if orig_token_value in ('[unused1]' , '[unused2]'):
                junk_tokens_indices.add(this_token_index)
                continue

            # discard non-CCW words
            if span_txt.lower() in CCWords:
                junk_tokens_indices.add(this_token_index)
                continue

            # discard all punc/space tokens
            if len(span_txt) in {2,3,4}:
                if len(span_txt.translate(self.translate_punc_space_to_None_pattern)) == 0: #meaning the whole token is compopsed of punctuations or space
                    junk_tokens_indices.add(this_token_index)
                    continue

            # discard every non-relevant dot. relevant dot has this pattern [unused1].[unused1]
            # any other pattern for dot is invalid
            if span_txt == ".":
                if (this_token_index > 0) and (this_token_index < last_token_index):
                    previous_token_orig_token_value = self.token_info_per_vector_index[this_token_index-1][-1]
                    next_token_orig_token_value = self.token_info_per_vector_index[this_token_index+1][-1]
                    if (previous_token_orig_token_value != '[unused1]') or (next_token_orig_token_value != '[unused1]'):
                        junk_tokens_indices.add(this_token_index)
                        continue
                else:
                    junk_tokens_indices.add(this_token_index)
                    continue

        valid_tokens_indices = all_tokens_indices - junk_tokens_indices
        return all_tokens_indices, valid_tokens_indices, junk_tokens_indices

    def BTSM_filter_junk_get_max_withoutparam(self):
        exclude_indices = self.junk_tokens_indices
        mask = np.ones(self.vector.shape[0], dtype=bool)
        mask[sorted(exclude_indices)] = False
        max_value = np.max(self.vector[mask])
        return ([index for index, value in enumerate(self.vector) if value == max_value and index not in exclude_indices])

    def get_best_triggers(self):
        try:
            # 1. find token text for all token index in the vector ...
            self.token_info_per_vector_index = self.create_token_info_per_vector_index()

            # 2. discard junk tokens
            self.all_tokens_indices, self.valid_tokens_indices, self.junk_tokens_indices = self.discard_junk_tokens()

            # 3. check if anything remains after junk elimination
            if len(self.valid_tokens_indices) == 0:
                raise Exception('no valid tokens indices.')

            # 4. get best triggers
            best_triggers_indices = self.BTSM_filter_junk_get_max_withoutparam()

            # 5. final checks on the best triggers
            selected_best_triggers_info = []
            for trigger_index in best_triggers_indices:
                if trigger_index in self.junk_tokens_indices:
                    continue
                span_bgn, span_end, span_txt, orig_token_value = self.token_info_per_vector_index[trigger_index]
                span_score = self.vector[trigger_index]
                selected_best_triggers_info.append((span_bgn, span_end, span_txt, span_score))

            if len(selected_best_triggers_info)<1:
                raise Exception("no trigger found.")

            return selected_best_triggers_info, None
        except Exception as E:
            return None, "error in best trigger detection: " + str(E)


