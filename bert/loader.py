from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
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
        
        
        with open(input_dir + filename, 'r') as f:
            self.data = json.load(f)
            
        
        if split == False:
            if label_type == 'binary':
                self.x_list, self.y_list = self.parse_binary()
        else:
            self.x_list = list()
            self.y_list = list()
            
            

    @staticmethod
    def concat_uncertain_sentence(sentence):
        """
          sentence: dict that contains 'ccue' keys
          return: complete str sentence, class label
        """
        ## TODO: find class label y:
        y = 0

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

    def parse_binary(self):
        certain = 0
        uncertain = 1
        x_list = []
        y_list = []
        Document = self.data['Annotation']['DocumentSet']['Document']
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
    
      
          
     
def split_data(dataset, input_dir, filename , label_type,train_size = 0.8, val_size = 0.1):
    total_length = len(dataset.x_list)
    train_idx = int(total_length * train_size)
    val_idx = int(total_length * val_size) + train_idx


    train_dataset = TextDataset(input_dir,filename,label_type,split = True)
    train_dataset.x_list = dataset.x_list[:train_idx]
    train_dataset.y_list = dataset.y_list[:train_idx]
        
    val_dataset = TextDataset(input_dir,filename,label_type,split = True)
    val_dataset.x_list = dataset.x_list[train_idx:val_idx]
    val_dataset.y_list = dataset.y_list[train_idx:val_idx]

    test_dataset = TextDataset(input_dir,filename ,label_type,split = True)
    test_dataset.x_list = dataset.x_list[val_idx:total_length]
    test_dataset.y_list = dataset.y_list[val_idx:total_length]

       
    print("training data point", len(train_dataset))
    print("validation data point", len(val_dataset))
    print("teting data point", len(test_dataset))

        
    return train_dataset, val_dataset, test_dataset

    

if __name__ == '__main__':
    input_dir = '/Users/alexandra/Desktop/DL/dl_final_project/bert/data/'
    filename = 'bmc.json'
    label_type = 'binary'
    train_batch_size = 5
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
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    

    print("Example:")
    for i,sample in enumerate(train_dataloader):
        print(i)
        print(sample) 
             
        raise Exception
