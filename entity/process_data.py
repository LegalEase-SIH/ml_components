import json
import re
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm


# define arguement parser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src", type=str, default="../data/ner/NER_TRAIN/NER_TRAIN_PREAMBLE.json"
    )
    parser.add_argument("--dst", type=str, default="../data/ner/ner_preamble_train.csv")
    args = parser.parse_args()
    return args


# define preprocessor
class Preprocessor(object):
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.data = json.load(open(self.args.src))
        self.classes = []

        print(f"[{datetime.now()}] Extractin Unique Entities...")
        # extract different entities
        for ix in tqdm(range(len(self.data))):
            for res in self.data[ix]["annotations"]:
                tmp = res["result"]
                for r in tmp:
                    self.classes.extend(r["value"]["labels"])
        self.classes = list(set(self.classes))
        self.classes.append("OBJECT")

        # print unique classes
        print(f"[{datetime.now()}]: Unique Classes: {self.classes}")

    def _process_data(self):
        self.df = {"sentence #": [], "words": [], "tag": []}

        print(f"[{datetime.now()}]: Starting to preprocess the JSON data...")

        skip = 0
        # Start the preprocessing step
        for ix in tqdm(range(len(self.data))):
            texts, labels, starts, ends = [], [], [], []

            for res in self.data[ix]["annotations"][0]["result"]:
                texts.append(res["value"]["text"])
                labels.extend(res["value"]["labels"])
                starts.append(res["value"]["start"])
                ends.append(res["value"]["end"])

            replaced_texts = []
            text = self.data[ix]["data"]["text"]
            for i in range(len(texts)):
                replaced_texts.append(text[starts[i] : ends[i]].replace(" ", "_"))

            err = False
            for j in range(len(replaced_texts)):
                try:
                    text = re.sub(texts[j], replaced_texts[j], text)
                except Exception as e:
                    cnt = self.df["sentence #"].count(f"Sentence: {ix - skip}")
                    skip += 1

                    # self.df["sentence #"] = self.df["sentence #"][:-cnt]
                    # self.df["words"] = self.df["words"][:-cnt]
                    # self.df["tag"] = self.df["tag"][:-cnt]

            if err == True:
                continue

            mapping_dict = dict(zip(replaced_texts, labels))
            text = text.split(" ")

            for w in text:
                if w in mapping_dict:
                    self.df["sentence #"].append(f"Sentence: {ix - skip}")
                    self.df["words"].append(w)
                    self.df["tag"].append(mapping_dict[w])
                else:
                    self.df["sentence #"].append(f"Sentence: {ix - skip}")
                    self.df["words"].append(w)
                    self.df["tag"].append("OBJECT")

        self.df = pd.DataFrame(self.df)
        self.df['words'] = self.df['words'].apply(lambda x: x.strip())
        self.df.to_csv(self.args.dst, index=False)

        print(
            f"[{datetime.now()}]: Finished Preprocessing Data... Saved File to {self.args.dst}"
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._process_data()


if __name__ == "__main__":
    args = parse_args()
    processor = Preprocessor(args=args)
    processor()
