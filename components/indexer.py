import json
from sentence_transformers import SentenceTransformer
from utils.faiss_util import DenseHNSWFlatIndexer


class Indexer:
    def __init__(self):
        self.indexer = self.load_indexer()

    @staticmethod
    def load_indexer():
        """
        loads required files for indexing
        :return: required mapping and encoder for indexing
        """
        i2e = json.load(open("data/wikidata/i2e.json"))
        with open("data/wikidata/i2e.json", "r") as file:
            i2e_data = json.load(file)
        i2id = json.load(open("data/wikidata/i2id.json"))
        encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')

        indexer = DenseHNSWFlatIndexer(1)
        indexer.deserialize_from("data/ReFi/indexed_ReFi_entities.pkl")

        with open("data/wikidata/index_to_entity_id.json", "r") as file:
            index_to_entity_id = json.load(file)

        return {
            "i2e": i2e,
            "i2id": i2id,
            "index_to_entity_id": index_to_entity_id,
            "indexer": indexer,
            "encoder": encoder
        }

    def lookup(self, text, topk=1):
        """
        Perform faiss_hnsw lookup
        :param topk: number of candidate entities
        :param text: text chunk to be looked up
        :return:  [labels],[ids] --> list of entity labels and entity ids
        """
        query_vector = self.indexer["encoder"].encode([text])
        sc, e_id = self.indexer["indexer"].search_knn(query_vector, topk)  # 1 -> means top entity
        print("Entity IDs:", e_id)
        return [self.indexer["i2e"].get(str(self.indexer["index_to_entity_id"][e_id[0][i]]), "Unknown") for i in range(topk)],\
               [self.indexer["i2id"].get(str(self.indexer["index_to_entity_id"][e_id[0][i]]), "Unknown") for i in range(topk)]


