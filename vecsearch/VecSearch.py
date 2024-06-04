import os
import numpy as np
import csv
import sys

def vecsearch_l2_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def vecsearch_cos_sim(vector1, vector2):
    return -np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class vectorEngine:
    def __init__(self, source_data_path, source_embedding_path, metadata_path):
        self.source_data_path = source_data_path
        self.source_embedding_path = source_embedding_path
        self.metadata_path = metadata_path
        self.vector_dict = {}

    def load_vectors(self, vector_path):
        self.vector_dict = {}
        for file in os.listdir(vector_path):
            if file.endswith(".npy"):
                local_vectors = np.load(vector_path + file)
                key_prefix = file.split(".")[0].replace("_vectors", "")
                chunk_id = 0
                for vector in local_vectors:
                    chunk_id += 1
                    vector_key = key_prefix + "_" + str(chunk_id)
                    self.vector_dict[vector_key] = vector
    
    def get_closest_vector(self, target_vector, top_k=3):
        # find the closest vector
        top_kv = []
        for key, vector in self.vector_dict.items():
            # print(key, L2_distance(vector, target_vector))
            top_kv.append((key, vecsearch_cos_sim(vector, target_vector)))
            top_kv.sort(key=lambda x: x[1])
            if len(top_kv) > top_k:
                top_kv.pop()
        return top_kv
    
    def get_info_from_metadata(vector):
        for 