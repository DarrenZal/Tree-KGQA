#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import json
from faiss_util import DenseHNSWFlatIndexer
from datetime import datetime


def get_args():
    parser = ArgumentParser(description="Entity indexer")
    parser.add_argument("--output_path", required=True, type=str, help="output path")
    parser.add_argument("--faiss_index", type=str, default="hnsw", help='hnsw index')
    parser.add_argument('--index_buffer', type=int, default=50000)
    parser.add_argument("--save_index", action='store_true', help='save indexed file')
    parsed_args = parser.parse_args()
    return parsed_args

def main(args):

    data, idx2entity, idx2id = list(), dict(), dict()
    start_time = datetime.now()
    
    # Load your specific JSON files here
    with open("utils/combined_data.json") as f:
        json_data = json.load(f)["profiles_json_ld"]
        
        for i, item in enumerate(json_data):
            
            # Index name and description for all, with some exceptions
            if "name" in item:
                data.append(item["name"])
                idx2entity[i] = item["name"]
                
            if "description" in item:
                data.append(item["description"])
                idx2entity[i] = item["description"]
            
            # For specific object types
            if "@type" in item and item["@type"] == "events_JSON_ld":
                if "about" in item:
                    data.append(item["about"])
                    idx2entity[i] = item["about"]
            
            if "@type" in item and item["@type"] == "content_JSON_ld":
                for field in ["keywords", "url", "imageURL", "excerpt"]:
                    if field in item:
                        data.append(item[field])
                        idx2entity[i] = item[field]
            
            # Add @type indexing here
            if "@type" in item:
                data.append(item["@type"])
                idx2entity[i] = item["@type"]

            # Capturing the "@id" value for idx2id
            if "@id" in item:
                idx2id[i] = item["@id"]
            else:
                idx2id[i] = None

    print('Data loading Duration: {}'.format(datetime.now() - start_time))
    start_time = datetime.now()
    json.dump(idx2entity,open("data/wikidata/i2e.json","w"), ensure_ascii=False)
    json.dump(idx2id, open("data/wikidata/i2id.json", "w"), ensure_ascii=False)
    print('Data saving Duration: {}'.format(datetime.now() - start_time))

    start_time = datetime.now()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    print('Sentence Transformer loading Duration: {}'.format(datetime.now() - start_time))

    start_time = datetime.now()
    encoded_data = model.encode(data)
    print('Encoding Duration: {}'.format(datetime.now() - start_time))

    print("Using HNSW index in FAISS")
    vector_size = 768
    index = DenseHNSWFlatIndexer(vector_size, len(data))
    print("Building index.")
    start_time = datetime.now()
    index.index_data(encoded_data)
    print("Done indexing data.")
    print(f'Number of vectors indexed: {index.index.ntotal}')
    print('Indexing Duration: {}'.format(datetime.now() - start_time))

    if args.save_index:                                                 # saving index
        print("Saving index file")
        index.serialize(args.output_path)
        print("Done")

if __name__ == '__main__':

    args = get_args()
    main(args)


# %%




