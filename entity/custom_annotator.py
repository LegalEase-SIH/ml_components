import json
from typing import Any
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class Annotator(object):
    def __init__(self, file_path: str) -> None:
        self.data = json.load(open(file_path, 'r'))
    
    def process(self):
        data = defaultdict(list)

        for ix in tqdm(range(len(self.data))):
            text = self.data[ix]['data']['text']
            annot = self.data[ix]['annotations'][0]['result']

            tags = [0 for _ in range(len(text))]

            for i, _ in enumerate(text):
                for x in annot:
                    start, end, label = x['value']['start'], x['value']['end'], x['value']['labels'][0]
                    if i >= start and i <= end:
                        tags[i] = label
                        break
            
            words, word_tags = [], []
            i = 0

            while i < len(text):
                if tags[i] == 0:
                    w = ''
                    while i < len(text) and text[i] != ' ' and tags[i] == 0:
                        w += text[i]
                        i += 1
                    
                    # print(1, w)
                    words.append(w)
                    word_tags.append('OBJECT')
                    # print(words, word_tags)
                    i += 1 # because this will now point to blank space and would leave a infinite loop
                else:
                    w = ''
                    curr_tag = tags[i]
                    while (i < len(tags) - 1) and (tags[i] == tags[i + 1]) and tags[i] != 0:
                        w += text[i]
                        i += 1
                    
                    # print(2, w)
                    words.append(w)
                    word_tags.append(curr_tag)

                    # print(words, word_tags)
                    
                    if curr_tag != tags[i]:
                        pass
                    else:
                        i += 1 # will be blank space -> infinite loop

                data['sentence'].extend([f'sentence #{ix}']*len(word_tags))
                data['words'].extend(words)
                data['tags'].extend(word_tags)

        print({k:len(v) for k,v in data.items()})
        # print(data)

        self.df = pd.DataFrame(data=data)

    def __call__(self) -> Any:
        self.process()
        self.df = self.df[self.df['words'] != '']
        self.df['words'] = self.df['words'].apply(lambda x: x.strip())
        self.df = self.df[self.df['words'] != '']
        print(self.df.shape)
        self.df.to_csv('../data/ner/ner_train_preamble.csv', index=False)


if __name__ == '__main__':
    p = Annotator(file_path='../data/ner/NER_TRAIN/NER_TRAIN_PREAMBLE.json')
    p()
