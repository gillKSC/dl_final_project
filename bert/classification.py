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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
       
        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
#         print(predictions)
#         print(batch['labels'])
#         raise Exception
        
        Y_true.append(batch['labels'].tolist())
        Y_pred.append(predictions.tolist())
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])
      

    # compute and return metrics
    Y_true = np.squeeze(np.array(Y_true))
    Y_pred = np.squeeze(np.array(Y_pred))
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
            val_accuracy_batch = evaluate.load('accuracy')
            
            train_accuracy_batch = evaluate.load('accuracy')

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
            train_accuracy_batch.add_batch(predictions=predictions, references=labels)
            acc_train = train_accuracy_batch.compute()
            train_acc_batch.append(acc_train["accuracy"])
       
            
            #update metrics for validation batch
            #option11: batch validation using a batch 
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits
            predictions = torch.argmax(predictions, dim=1)
            val_accuracy_batch.add_batch(predictions=predictions, references=labels)
            acc_val = val_accuracy_batch.compute()
            val_acc_batch.append(acc_val["accuracy"])
            
            
            #obtion2: batch validation using all batches
            #val_accuracy_batch = evaluate_model(mymodel, validation_dataloader, device)
            #val_acc_batch.append(val_accuracy_batch["accuracy"])
                   
        #computer for train epoch
        acc = train_accuracy.compute()
        train_acc_epoch.append(acc["accuracy"])
        
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy={acc}")
        
        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        #add val epoch
        val_acc_epoch.append(val_accuracy["accuracy"])
        print(f" - Average validation metrics: accuracy={val_accuracy}")

    #plotting
    plotext.subplot(1,1)
    plotext.plot(train_acc_epoch, label='training')
    plotext.plot(val_acc_epoch, label='validation')
    plotext.xlabel('epoch')
    plotext.ylabel('epoch accuracy')
    
    plotext.plot(1,2)
    plotext.plot(val_acc_batch, label='validation')
    plotext.plot(train_acc_batch, label='training')
    plotext.xlabel('batch')
    plotext.ylabel('batch accuracy')
    plotext.show()
    
    return mymodel



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
    print("testing data point", len(test_dataset))

        
    return train_dataset, val_dataset, test_dataset


def pre_process(model_name, batch_size, device, input_dir, filename, label_type='binary', num_labels=2, small_subset=True):

    dataset = TextDataset(input_dir, filename, label_type = label_type)
    train_dataset,val_dataset,test_dataset = split_data(dataset,input_dir,filename,label_type)
#     train_mask = range(int(len(dataset)*0.7))
#     val_mask = range(int(len(dataset)*0.7), int(len(dataset)*0.85))
#     test_mask = range(int(len(dataset)*0.85), int(len(dataset)*1))
    
#     train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_mask))
#     validation_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_mask))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print(" >>>>>>>> Initializing the data loaders ... ")
    
    if small_subset:
        train_mask = range(5)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_mask))
        validation_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_mask))
        
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader #, test_dataloader



# the entry point of the program
if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", type=str, default=False)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--type_classification", type=str, default="binary")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")
    input_dir = './data/'
    filename = 'wiki.json'
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
    pretrained_model, train_dataloader,validation_dataloader = pre_process(args.model,
                                                     args.batch_size,
                                                     args.device,
                                                     input_dir,
                                                     filename,
                                                     label_type=args.type_classification,
                                                     num_labels = num_class,
                                                     small_subset=small_subset
                                                    )


    print(" >>>>>>>>  Starting training ... ")
    trained_model = train(pretrained_model, args.num_epochs, train_dataloader, args.device, args.lr)

    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()
    
    (val_accuracy,Y_true,Y_pred) = evaluate_model(trained_model, validation_dataloader, args.device, acc_only=False)
    print(val_accuracy)
    print(Y_true)
    print(Y_pred)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    confusion_matrix_array = confusion_matrix(Y_true, Y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_array)
    cm_display.plot()
    plt.show(block=True)

    '''
    test_accuracy = evaluate_model(trained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
    '''
