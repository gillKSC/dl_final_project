from torch.utils.data import Dataset, DataLoader, random_split
import json
from collections import Counter

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
## Text Dataloader
class TextDataset(Dataset):
    
    def __init__(self,
                 input_dir,
                 filename,
                 label_type = 'binary',
                 transforms = None,
                 split = False):

        self.num_certain = 0
        self.num_uncertain = 0
        self.x_list = []
        self.y_list = []
        self.uncertainty_label = {"speculation_hypo_doxastic _": 1, 
                          "speculation_hypo_investigation _": 2,
                          "speculation_modal_probable_": 3,
                          "speculation_modal_possible_": 3,
                          "speculation_hypo_condition _": 4,
                          "multiple_uncertain": 5
                          }
                                  
        
        # with open(input_dir + filename, 'r') as f:
        #     self.data = json.load(f)
            
        
  
        if label_type == 'binary':
            for file in filename:
                with open(input_dir + file, 'r') as f:
                    ALLdata = json.load(f)
                XList, YList = self.parse_binary(ALLdata)
                self.x_list += XList
                self.y_list += YList

        elif label_type == 'multi':
            for file in filename:
                with open(input_dir + file, 'r') as f:
                    ALLdata = json.load(f)
                XList, YList = self.parse_multi(ALLdata)
                self.x_list += XList
                self.y_list += YList
      
    def max_occurrences(self, lst):
        """
        Returns the item with the maximum occurrences in a list. 
        If multiple elements have the same maximum occurrences, return -1.
        """
        count = Counter(lst)
        max_count = max(count.values())
        max_items = [k for k,v in count.items() if v == max_count]
        if len(max_items) > 1 and count[max_items[0]] == max_count:
            return -1
        else:
            return max_items[0]

    def find_label(self, sentence):
        if type(sentence['ccue']) == dict:
            label_type = sentence['ccue']['@type']
            return self.uncertainty_label[label_type]
        elif type(sentence['ccue']) == list:
            lst = []
            for s in sentence['ccue']:
                lst.append(self.uncertainty_label[s['@type']])
            label = self.max_occurrences(lst)
            if label == -1:
                label = self.uncertainty_label["multiple_uncertain"]
            return label

    def concat_uncertain_sentence(self, sentence, multi_label = False):
        """
          sentence: dict that contains 'ccue' keys
          return: complete str sentence, class label
        """
        y = 0
        if multi_label:
            y = self.find_label(sentence)

        # In some sentence, there's no head
        try:
            head = sentence['#text']
        except KeyError:
            head = ''

        if type(sentence['ccue']) == dict:
            keyword = sentence['ccue']['#text']
            x = head + keyword + sentence['ccue']['#tail']
        elif type(sentence['ccue']) == list:  ## multiple keywords
            x = head
            for s in sentence['ccue']:
                keyword = s['#text']
                x += keyword + s['#tail']
        return x, y

    def __len__(self):

        return len(self.x_list)

    def __getitem__(self, idx):
        text = self.x_list[idx]
        x_token = tokenizer(text,
                   padding='max_length',
                   max_length=512,
                   truncation=True,
                   return_tensors="pt")

        return {
            'input_ids': x_token['input_ids'][0],  # we only have one example in the batch
            'attention_mask': x_token['attention_mask'][0], # attention mask tells the model where tokens are padding
            'labels': self.y_list[idx]  # labels are the answers (yes/no)
            }

    def parse_binary(self, ALLdata):
        certain = 0
        uncertain = 1
        x_list = []
        y_list = []
        Document = ALLdata['Annotation']['DocumentSet']['Document']
        for doc in Document:
            DocumentPart = doc['DocumentPart']
            for paragraph in DocumentPart:
                try:
                    if type(paragraph) != dict:
                        raise KeyError()
                    paragraph['Sentence']
                except KeyError:
                    continue
                # print(paragraph['@type'])
                # if (not paragraph['@type'] == 'Text') and ('ccue' in paragraph['Sentence']):
                #     print(True)
                #     print(paragraph['Sentence']['@id'])
                # if paragraph['@type'] == 'Text' or paragraph['@type'] == 'unknown' or paragraph['@type'] == "FigureLegend":
                for sentence in paragraph['Sentence']:
                    if type(sentence) != dict:
                        continue
                    if 'ccue' in sentence:  # has uncertain keyword
                        self.num_uncertain += 1
                        x, _ = self.concat_uncertain_sentence(sentence)
                        y = uncertain
                        x_list.append(x)
                        y_list.append(y)
                    else:  # no uncertain keyword
                        self.num_certain += 1
                        y_list.append(certain)
                        x_list.append(sentence['#text'])
                # else:
                #     y_list.append(certain)
                #     x_list.append(paragraph['Sentence']['#text'])

        assert len(x_list) == len(y_list)
        return x_list, y_list

    def parse_multi(self, ALLdata):
        certain = 0
        x_list = []
        y_list = []
        Document = ALLdata['Annotation']['DocumentSet']['Document']
        for doc in Document:
            DocumentPart = doc['DocumentPart']
            for paragraph in DocumentPart:
                try:
                    if type(paragraph) != dict:
                        raise KeyError()
                    paragraph['Sentence']
                except KeyError:
                    continue
                for sentence in paragraph['Sentence']:
                    if type(sentence) != dict:
                        continue
                    if 'ccue' in sentence:  # has uncertain keyword
                        self.num_uncertain += 1
                        x, y = self.concat_uncertain_sentence(sentence, multi_label = True)
                        x_list.append(x)
                        y_list.append(y)
                    else:  # no uncertain keyword
                        self.num_certain += 1
                        y_list.append(certain)
                        x_list.append(sentence['#text'])

        assert len(x_list) == len(y_list)
        return x_list, y_list
      
          
     
def split_data(dataset, input_dir, filename , label_type,train_size = 0.8, val_size = 0.1):
    total_length = len(dataset)
    # train_idx = int(total_length * train_size)
    # val_idx = int(total_length * val_size) + train_idx
    # print(train_idx)
    # print(val_idx)
    train_length = int(total_length * train_size)
    val_length = int(total_length * val_size)
    test_length = int(total_length - (train_length + val_length))

    # train_dataset = TextDataset(input_dir,filename,label_type,split = True)
    # train_dataset.x_list = dataset.x_list[:train_idx]
    # train_dataset.y_list = dataset.y_list[:train_idx]
        
    # val_dataset = TextDataset(input_dir,filename,label_type,split = True)
    # val_dataset.x_list = dataset.x_list[train_idx:val_idx]
    # val_dataset.y_list = dataset.y_list[train_idx:val_idx]

    # test_dataset = TextDataset(input_dir,filename ,label_type,split = True)
    # test_dataset.x_list = dataset.x_list[val_idx:total_length]
    # test_dataset.y_list = dataset.y_list[val_idx:total_length]
    train_dataset, val_dataset, test_dataset= random_split(dataset, [train_length, val_length, test_length])


    
    print("training data point", len(train_dataset))
    print("validation data point", len(val_dataset))
    print("teting data point", len(test_dataset))

        
    return train_dataset, val_dataset, test_dataset

    

if __name__ == '__main__':
    input_dir = '/content/'
    #filename = 'wiki.json'
    filename = ['bmc.json', 'wiki.json', 'factbank.json', 'fly.json', 'hbc.json']
    label_type = 'binary'
    train_batch_size = 2
    val_batch_size = 5
    test_batch_size = 5


    #Get all data 
    dataset = TextDataset(input_dir, filename, label_type = label_type)
    print("total number of data point:", len(dataset))
    print("total number of certain point:", dataset.num_certain)
    print("total number of uncertain point:", dataset.num_uncertain)
    
    #split into train/val/test
    train_dataset,val_dataset,test_dataset = split_data(dataset,input_dir,filename,label_type)

    #put into different dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=True)
    

    print("Example:")
    for i, (X, Y) in enumerate(train_dataloader):
        print(X)
        print(Y)
        raise Exception

