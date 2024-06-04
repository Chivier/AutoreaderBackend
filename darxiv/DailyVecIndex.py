import os
import csv
import torch
import json
import numpy as np
import openai

def L2_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def cos_sim(vector1, vector2):
    return -np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_closest_vector(vectors, target_vector, top_k=3):
    # find the closest vector
    top_kv = []
    for key, vector in vectors.items():
        # print(key, L2_distance(vector, target_vector))
        top_kv.append((key, cos_sim(vector, target_vector)))
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
            

def generate_report(source_path, top_kv):
    for key, sim in top_kv:
        file_id, chunk_id = read_source_from_key(source_path, key)
        

        