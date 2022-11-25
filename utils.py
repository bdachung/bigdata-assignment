import os
import torch
import re
import gensim
from vncorenlp import VnCoreNLP
import numpy as np

rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\\\n\(\)]'

def remove_special_characters(batch):
    batch = re.sub(chars_to_ignore_regex, '', batch)
    return batch

def remove_number_characters(batch):
    batch = re.sub("\d+",'', batch)
    return batch

def remove_any_special_left(batch):
    # Loại bỏ các kí tự đặc biệt
    batch = gensim.utils.simple_preprocess(batch)
    batch = ' '.join(batch)
    # Sau khi loại bỏ kí tự đặc biệt thì file sẽ bị mất nhận diện chuỗi -> thêm dấu nháy vào chuỗi
    return batch

def segment_text(text):
    batch = rdrsegmenter.tokenize(text)
    if len(batch) == 0:
        return ""
    else:
        return " ".join(batch[0]).strip()

def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


