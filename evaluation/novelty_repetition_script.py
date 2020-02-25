import os
import json
import shutil

root = "/remote-home/yrchen/tasks/fastnlp-relevant/summarization/my-pnt-sum/log"

with open("result_paths.json", 'r') as f:
    result_paths = json.load(f)

if os.path.exists("result"):
    shutil.rmtree("result")

for key, value in result_paths.items():
    for path in value["pred_paths"]:
        print("python novelty_repetition.py -pred_path {} -raw_path {} -n_grams 1,2,3,4,-1".format(
            os.path.join(root, path), value["raw_path"]))
        os.system("python novelty_repetition.py -pred_path {} -raw_path {} -n_grams 1,2,3,4,-1".format(
            os.path.join(root, path), value["raw_path"]))
