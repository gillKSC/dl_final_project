from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
## Text Dataloader
class TextDataset(Dataset):

    def __init__(self,
                 input_dir,
                 filename,
                 label_type='binary',
                 transforms=None):

        self.num_certain = 0
        self.num_uncertain = 0
        with open(input_dir + filename, 'r') as f:
            self.data = json.load(f)

        if label_type == 'binary':
            self.x_list, self.y_list = self.parse_binary()

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
            'attention_mask': x_token['attention_mask'][0],
            # attention mask tells the model where tokens are padding
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

if __name__ == '__main__':
    input_dir = 'data/'
    filename = 'bmc.json'
    label_type = 'binary'
    train_batch_size = 3


    train_dataset=TextDataset(input_dir, filename, label_type = label_type)
    print("number of data point:", len(train_dataset))
    print("number of certain point:", train_dataset.num_certain)
    print("number of uncertain point:", train_dataset.num_uncertain)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    print("Example:")
    for i, (X, Y) in enumerate(train_dataloader):
        print(X)
        print(Y)
        raise Exception