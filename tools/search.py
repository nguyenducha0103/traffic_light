import faiss
import numpy as np
from numpy.linalg import norm
import time
import sys

def compute_sim(emb1, emb2):
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

class VectorWarehouse(object):
    def __init__(self, vector_list, dimension = 512):
        self.dim = dimension
        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vector_list)

    def search(self, vector):
        if vector.shape==(1,self.dim):
            D, I = self.index.search(vector, 1)
        else:
            print(f'[E] Error Shape is not match between ur shape {vector.shape} vs (1,{self.dim})')
            sys.exit()
        return I