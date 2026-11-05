import pandas as pd
import numpy as np
import torch 
from utils import construct_wan_from_text
import json 

WORD_COUNT = [50, 20, 10]
function_words_file = "data/function_words.txt"
window_width = 5

df_original = pd.read_csv("data/persuade_wan_cooc_qwen2515b.csv")

for wc in WORD_COUNT:
    print(f"Starting {wc}")
    df_new = df_original.copy(deep=True)
    df_new["output_text"] = df_new["output_text"].apply(lambda x: " ".join(x.split()[:wc]))
    df_new["WAN"] = [json.dumps(construct_wan_from_text(t, functionWordsFile=function_words_file, windowWidth=window_width)) for t in df_new["output_text"]]
    df_new.to_csv(f"data/persuade_wan_qwen2515b_{wc}.csv", index = False)