import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import random
import numpy as np
from loader import TextDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import copy


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


def evaluate_model(model, dataloader, device, acc_only=True):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :param bool acc_only: return only accuracy if true, else also return ground truth and pred as tuple
    :return accuracy (also return ground truth and pred as tuple if acc_only=False)
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')
    
    # turn model into evaluation mode
    model.eval()

    #Y_true and Y_pred store for epoch
    Y_true = []
    Y_pred = []
    val_acc_batch = []
    
    
    val_accuracy_batch = evaluate.load('accuracy')
    
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
       
        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        Y_true += batch['labels'].tolist()
        Y_pred += predictions.tolist()
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])
        
        if acc_only == True:
            correct = (predictions.to(device) == batch['labels'].to(device)).sum().item()
            val_accuracy_batch = correct/len(predictions)
            val_acc_batch.append(val_accuracy_batch)
            
      

    # compute and return metrics
#     Y_true = np.squeeze(np.array(Y_true))
#     Y_pred = np.squeeze(np.array(Y_pred))
    
    load_new_list('val_acc_batch',val_acc_batch)
    
    return dev_accuracy.compute() if acc_only else (dev_accuracy.compute(),Y_true,Y_pred)

def train(mymodel, num_epochs, train_dataloader, validation_dataloader,device, lr):
    """ Train a PyTorch Module
    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """
    
    # store for plotting
    train_acc_epoch = []
    train_acc_batch = []
    val_acc_epoch = []
    val_acc_batch = []
    
    max_ = 0
    Best_model = None


    with open('val_acc_batch' + '.pickle', 'wb') as f:
        pickle.dump((val_acc_batch), f)
        f.close()


    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')
        

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):
            
            #load metrics
            #train_accuracy_batch = evaluate.load('accuracy')

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = mymodel(input_ids, attention_mask=attention_mask)
            predictions = output.logits

            model_loss = loss(predictions, labels)

            model_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = torch.argmax(predictions, dim=1)

            # update metrics for train epoch 
            train_accuracy.add_batch(predictions=predictions, references=labels)
            
            # update metrics for train batch
            correct = (predictions == labels).sum().item()
            train_accuracy_batch = correct/len(predictions)
            train_acc_batch.append(train_accuracy_batch)
                  
                   
        #computer for train epoch
        acc = train_accuracy.compute()
        train_acc_epoch.append(acc["accuracy"])
        
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy={acc}")
        
        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        val_acc_epoch.append(val_accuracy["accuracy"])
        
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        
        
        
        if (val_acc_epoch[epoch] > max_):
            max_ = val_acc_epoch[epoch]
            saved_model_path = 'my_saved_model.pth'
            torch.save(mymodel, saved_model_path)
            Best_model = copy.deepcopy(mymodel)
        
        
    with open('train_acc_epoch' + '.pickle', 'wb') as f:
        pickle.dump((train_acc_epoch), f)
        f.close()
       

    with open('train_acc_batch' + '.pickle', 'wb') as f:
        pickle.dump((train_acc_batch), f)
        f.close()
        
        
    with open('val_acc_epoch' + '.pickle', 'wb') as f:
        pickle.dump((val_acc_epoch), f)  
        f.close()                 
    
    return Best_model

def load_new_list(path, newdata):
    with open(path+ '.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
        loaded_data = loaded_data + newdata
        f.close()
        
        
        
    with open(path +'.pickle', 'wb') as f:
        pickle.dump((loaded_data), f)
        f.close()
       
    

def split_data(dataset, input_dir, filename, label_type,train_size = 0.8, val_size = 0.1):
    total_length = len(dataset)
    train_length = int(total_length * train_size)
    val_length = int(total_length * val_size)
    test_length = int(total_length - (train_length + val_length))
    train_dataset, val_dataset, test_dataset= random_split(dataset, [train_length, val_length, test_length])

    
    print("training data point", len(train_dataset))
    print("validation data point", len(val_dataset))
    print("teting data point", len(test_dataset))

        
    return train_dataset, val_dataset, test_dataset

def pre_process(model_name, batch_size, device, input_dir, filename, label_type='binary', num_labels=2, small_subset=True, train_size = 0.8, val_size = 0.1):

    dataset = TextDataset(input_dir, filename, label_type = label_type)
    train_dataset,val_dataset,test_dataset = split_data(dataset, input_dir, filename, label_type, train_size = 0.8, val_size = 0.1)
#     train_mask = range(int(len(dataset)*0.7))
#     val_mask = range(int(len(dataset)*0.7), int(len(dataset)*0.85))
#     test_mask = range(int(len(dataset)*0.85), int(len(dataset)*1))
    
#     train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_mask))
#     validation_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_mask))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    print(" >>>>>>>> Initializing the data loaders ... ")
    
    if small_subset:
        train_mask = range(50)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_mask))
        validation_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_mask))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_mask))
    
    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader

def plot_confusion_matrix(Y_true, Y_pred, saved_name= 'confusion_matrix.jpg'):
    confusion_matrix_array = confusion_matrix(Y_true, Y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_array)
    cm_display.plot()
    plt.show(block=True)
    plt.savefig(saved_name)
    

# the entry point of the program
if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="overfit")
    parser.add_argument("--small_subset", type=str, default=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--type_classification", type=str, default="binary")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")
    input_dir = './data/'
    filename = ['wiki.json', 'bmc.json', 'factbank.json','fly.json','hbc.json']
    # Handling argparse for small_subset param

    small_subset = str(args.small_subset).upper()
    if small_subset == 'TRUE' or small_subset == "1":
        small_subset = True
    else:
        small_subset = False
    
    if str(args.type_classification) == 'binary':
        num_class = 2
    elif str(args.type_classification) == 'multi':
        num_class = 6

    # load the data and models
    pretrained_model, train_dataloader,validation_dataloader, test_dataloader = pre_process(args.model,
                                                     args.batch_size,
                                                     args.device,
                                                     input_dir,
                                                     filename,
                                                     label_type = args.type_classification,
                                                     num_labels = num_class,
                                                     small_subset = small_subset,
                                                     train_size = 0.8,
                                                     val_size = 0.1
                                                    )


    print(" >>>>>>>>  Starting training ... ")
    trained_model = train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr)

    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()
    
    (val_accuracy,Y_true,Y_pred) = evaluate_model(trained_model, validation_dataloader, args.device, acc_only=False)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")
    plot_confusion_matrix(Y_true, Y_pred, saved_name='confusion_matrix_validation.jpg')
 
    (test_accuracy,Y_true_test,Y_pred_test) = evaluate_model(trained_model, test_dataloader, args.device, acc_only=False)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
    plot_confusion_matrix(Y_true_test, Y_pred_test, saved_name= 'confusion_matrix_test.jpg')
    
    
    certain = 0
    uncertain = 1
    
    #calculating precision, f-1 and recall for test
    if str(args.type_classification) == 'binary':        
        print("binary certain precision for score: {:.2f}".format(precision_score(Y_true_test, Y_pred_test, average='binary', pos_label = certain)))        
        print("binary uncertain precision for score: {:.2f}".format(precision_score(Y_true_test, Y_pred_test, average='binary', pos_label = uncertain)))
        
        print("binary certain f-1 score: {:.2f}".format(f1_score(Y_true_test, Y_pred_test, average='binary', pos_label = certain)))
        print("binary uncertain f-1 score: {:.2f}".format(f1_score(Y_true_test, Y_pred_test, average='binary', pos_label = uncertain)))
        
        print("binary certain recall score: {:.2f}".format(recall_score(Y_true_test, Y_pred_test, average='binary', pos_label=certain)))
        print("binary uncertain recall score: {:.2f}".format(recall_score(Y_true_test, Y_pred_test, average='binary', pos_label=uncertain)))



    else:
        precision = precision_score(Y_true_test, Y_pred_test, average=None)
        recall = recall_score(Y_true_test, Y_pred_test, average=None)
        f1 = f1_score(Y_true_test, Y_pred_test, average=None)

        # print the precision for each label
        for i, label in enumerate(precision):
            print(f"Precision for label {i}: {label:.2f}")

        for i, label in enumerate(recall):
            print(f"recall for label {i}: {label:.2f}")

        for i, label in enumerate(f1):
            print(f"f1 score for label {i}: {label:.2f}")
