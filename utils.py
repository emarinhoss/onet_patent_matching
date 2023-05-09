import numpy as np
import pandas as pd

# Embeddings
from sentence_transformers import SentenceTransformer

# analysis
import scann

## Class that takes as input two lists and performs similarity analysis using hugging face sentence transformers
class Patent_ONET_tasks_matching:

    def __init__(self, onet_tasks:pd.DataFrame, patents:pd.DataFrame, device:str ="cuda", llm_model:str ='anferico/bert-for-patents') -> None:
        self.device = device
        self.useLLM = llm_model
        self.onet_tasks = onet_tasks
        self.patents = patents
        self.patent_matches_by_year = pd.DataFrame()
        self.model = SentenceTransformer(self.useLLM, 
                                         device=self.device)
        
        # Generate the ONET-embeddings
        self.onet_tasks.embeddings = self.model.encode(self.onet_tasks.Task.values.tolist(), show_progress_bar=True)

        # Normalize the ONET-embeddings
        self.onet_tasks.normalized_embeddings = self.onet_tasks.embeddings / np.linalg.norm(self.onet_tasks.embeddings, axis=1)[:, np.newaxis]
        pass

    # Function that takes in a list of patent titles, generates the needed embeddings, and determines patents that closely match a task 
    # using the scann library with dot product as a metric
    def compareByYear(self, depth:int=500, metric:str="dot_product"):

        for year in np.sort(self.patents['year'].unique()):
            print(year)
            # generate input data
            pat_year = self.patents[self.patents['year']==year]
            input = pat_year.title.values.tolist()
            results = self.model.encode(input, show_progress_bar=True)

            # Normalize 
            normalized_results = results / np.linalg.norm(results, axis=1)[:, np.newaxis]

            # use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
            searcher = scann.scann_ops_pybind.builder(normalized_results, depth, metric).tree(
                num_leaves=200, num_leaves_to_search=100, training_sample_size=250000).score_ah(2,anisotropic_quantization_threshold=0.2).reorder(100).build()
    
            # perform the search
            neighbors, distances = searcher.search_batched(normalized_queries,
                                                           leaves_to_search=150,
                                                           pre_reorder_num_neighbors=250)
    
    # 
            self.onet_tasks['neighbors_'+str(year)] = neighbors.tolist()
            self.onet_tasks['distances_'+str(year)] = distances.tolist()
    
    # This funtion will count the number of patents that match a task above
    # a specific cosine similarity threshold.
    def count_above_threshold(self, entry, threshold):
        res = list(entry.replace('[','').replace(']','').split(', '))
        my_list = [float(i) for i in res]
        return sum(1 for x in my_list if x > threshold)
