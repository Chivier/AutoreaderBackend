import os
import json
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
        self.index = None

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
    
    def get_closest_vector_dummy(self, target_vector, top_k=3):
        # find the closest vector
        top_kv = []
        for key, vector in self.vector_dict.items():
            # print(key, L2_distance(vector, target_vector))
            top_kv.append((key, vecsearch_cos_sim(vector, target_vector)))
            top_kv.sort(key=lambda x: x[1])
            if len(top_kv) > top_k:
                top_kv.pop()
        return top_kv

def read_source_from_key(source_path, key):
    chunk_id = key.split("_")[-1]
    # file_id = key remove "_{chunk_id}"
    file_id = key[:-(len(chunk_id) + 1)]
    # print(chunk_id, file_id)
    with open(source_path + file_id + "_chunks.csv", "r") as f:
        # read #chunk_id line from the file
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == int(chunk_id):
                return file_id, row[1]

def read_metadata_from_key(metadata_path, key):
    chunk_id = key.split("_")[-1]
    # file_id = key remove "_{chunk_id}"
    file_id = key[:-(len(chunk_id) + 1)]
    file_id = file_id.replace("-", ".").replace("_", "v")
    with open(metadata_path, "r") as f:
        metadata_json = json.load(f)
        for item in metadata_json:
            if file_id in item["guid"]:
                return item
    return None
    

def get_info_from_metadata(top_kv_list, metadata_path):
    for key, distance in top_kv_list:
        print(key)