import shutil
import os
from os.path import join

try:
    shutil.rmtree("corpus")
except:
    pass
finally:
    os.mkdir("corpus")
    os.mkdir("corpus/train")
    os.mkdir("corpus/test")
LABEL_CHOICES = "0123456789abcdefghijklmnopqrstuvwxyz"
def remove_invalid_images(files):
    output = []
    for file in files:
        label = file.split(".")[0]
        if len(label) != 5:
            continue
        if len(set(label.lower()).difference(set(LABEL_CHOICES))) > 0:
            continue
        output.append(file)
    return output

TRAIN_SAMPLES = 8000

RAW_FOLDER = "raw"
files = os.listdir(RAW_FOLDER)
files = remove_invalid_images(files)
train_files = files[:TRAIN_SAMPLES]
test_files = files[TRAIN_SAMPLES:]

for file in train_files:
    name = file.split(".")[0].lower() + "_0.png"
    shutil.copyfile(join(RAW_FOLDER, file), join("corpus", "train", name))

for file in test_files:
    name = file.split(".")[0].lower() + "_0.png"
    shutil.copyfile(join(RAW_FOLDER, file), join("corpus", "test", name))

meta = {
    "num_per_image": 5,
    "label_size": 36,
    "label_choices": "0123456789abcdefghijklmnopqrstuvwxyz",
    "n_epoch": 1,
    "width": 200,
    "height": 45
}
import json
f = open(join("corpus", "meta.json"), "w")
f.write(json.dumps(meta))


