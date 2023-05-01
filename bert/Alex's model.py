import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import numpy as np
from loader import TextDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from transformers import pipeline



if __name__ == "__main__":
    question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

    context = 'Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.'

    result = question_answerer(question="do you know what are you?",    context=context)
    print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
