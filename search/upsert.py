import json
import os

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()

import pinecone

pinecone.init(      
	api_key=os.environ['PINECONE_API_KEY'],      
	environment=os.environ['PINECONE_ENVIRONMENT']
)      
index = pinecone.Index(os.environ['PINECONE_INDEX'])

idx = 0

for ix, file in enumerate(tqdm(os.listdir('../data/ner/annotatedCentralActs/'))):
    data = json.load(open(f'../data/ner/annotatedCentralActs/{file}'))
    # print(data.keys())
    # DEFINITIONS
    print(f'--------- Act: {data["Act Title"]}')
    try:
        definitions = {
            'Act Title': data['Act Title'],
            'Act ID': data['Act ID'],
            'text': '\n'.join([v for k,v in data['Act Definition'].items()])
        }
    except: 
        definitions = None
        
    # FOOTNOTES
    tmps = []
    for k,v in data['Footnotes'].items():
        tmps.extend([x for y, x in v.items()])
    
    foot_notes = {
        'Act Title': data['Act Title'],
        'Act ID': data['Act ID'],
        'text': '\n'.join(tmps)
    }
    
    ### PARTS
    parts = []
    # print(data.keys())
    try:
        for _, v in data['Parts'].items():
            if 'ID' in v.keys():
                sections = v['Sections']
                # __import__('pprint').pprint(sections)
                # data['']
                try:
                    for n, p in sections.items():
                        # print(p)
                        t = '\n'.join([v for k,v in p['paragraphs'].items()])
                        para = [f"{p['heading']}\n{t}"]
                        # print([p['heading']])
                        para = [p['heading']] + para
                        parts.append({'section': data['Act Title'] + f' - {n}', 'text': '\n'.join(para)})
                except:
                    pass
    except:
        try:
            for n, p in sections.items():
                # print(p)
                t = '\n'.join([v for k,v in p['paragraphs'].items()])
                para = [f"{p['heading']}\n{t}"]
                # print([p['heading']])
                para = [p['heading']] + para
                parts.append({'section': data['Act Title'] + f' - {n}', 'text': '\n'.join(para)})
        except:
            pass

    if definitions is not None:
        chunks = []
        for i in range(0, len(definitions['text']), 1000):
            mx = min(len(definitions['text']), i+1000)
            chunks.append(definitions['text'][i:mx])
        
        for chunk in tqdm(chunks):
            index.upsert(vectors=[(str(idx), model.encode(chunk).tolist(), {'Act Title': data['Act Title'], 'Act ID': data['Act ID'], 'text': chunk})])
            idx += 1

    if foot_notes is not None:
        chunks = []
        for i in range(0, len(foot_notes['text']), 1000):
            mx = min(len(foot_notes['text']), i+1000)
            chunks.append(foot_notes['text'][i:mx])
        
        for chunk in tqdm(chunks):
            index.upsert(vectors=[(str(idx), model.encode(chunk).tolist(), {'Act Title': data['Act Title'], 'Act ID': data['Act ID'], 'text': chunk})])
            idx += 1

    for part in parts:

        chunks = []

        for i in range(0, len(part['text']), 1000):
            mx = min(len(part['text']), i+1000)
            chunks.append(part['text'][i:mx])

        for chunk in tqdm(chunks):
            index.upsert(vectors=[(str(idx), model.encode(chunk).tolist(), part)])
            idx += 1
