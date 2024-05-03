import os
import json
import argparse

def read_json_file(filepath):
    with open(filepath, "rt", encoding="utf-8") as f:
        content = json.load(f)
    return content


def save_as_json(data, filepath):
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=lambda o: o.__dict__,  # to be able to serialize nested objects https://stackoverflow.com/a/15538391
            )
        return True, None
    except Exception as E:
        return False, str(E)


if __name__=="__main__":
    content = read_json_file("the_best_model/info.json")
    new_moodel_path = os.path.dirname(os.path.realpath(__file__))
    if new_moodel_path[-1] != "/":
        new_moodel_path+="/"
    new_moodel_path += "original_model/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf/"
    content["pretrained_model_name_or_path"] = new_moodel_path

    print ("updating the_best_model/info.json file ...")
    print ("setting pretrained_model_name_or_path = " , new_moodel_path)
    save_as_json(content, "the_best_model/info.json")
    print ("Done.")
