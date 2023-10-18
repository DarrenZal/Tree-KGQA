#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import json
from faiss_util import DenseHNSWFlatIndexer
from datetime import datetime
import numpy as np

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
    global_index = 0

    with open("utils/combined_data.json") as f:
        all_data = json.load(f)

        for entity_type, entity_list in all_data.items():
            for i, item in enumerate(entity_list):

                if item is None:
                    continue

                entity_value = None
                # List to store the indices of all entities extracted from the current item
                current_item_indices = []

                if "name" in item:
                    entity_value = item["name"]
                    data.append(entity_value)
                    idx2entity[global_index] = entity_value
                    current_item_indices.append(global_index)
                    global_index += 1

                if "description" in item:
                    entity_value = item["description"]
                    data.append(entity_value)
                    idx2entity[global_index] = entity_value
                    current_item_indices.append(global_index)
                    global_index += 1

                if "@type" in item:
                    entity_value = item["@type"]
                    data.append(entity_value)
                    idx2entity[global_index] = entity_value
                    current_item_indices.append(global_index)
                    global_index += 1

                    if item["@type"] == "events_JSON_ld" and "about" in item:
                        data.append(item["about"])
                        idx2entity[global_index] = item["about"]
                        current_item_indices.append(global_index)
                        global_index += 1

                    elif item["@type"] == "content_JSON_ld":
                        for field in ["keywords", "url", "imageURL", "excerpt"]:
                            if field in item:
                                data.append(item[field])
                                idx2entity[global_index] = item[field]
                                current_item_indices.append(global_index)
                                global_index += 1

                # Associate the @id with all entities extracted from the current item
                if "@id" in item:
                    for idx in current_item_indices:
                        idx2id[idx] = item["@id"]
                else:
                    for idx in current_item_indices:
                        idx2id[idx] = None


                    
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




