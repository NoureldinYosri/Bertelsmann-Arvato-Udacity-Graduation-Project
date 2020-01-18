import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
import argparse
import time
from six import BytesIO

class ArvatoClassifier (torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, drop_prob = 0.3):
        super(ArvatoClassifier, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()
        self.fc = []
        self.input_size = input_size
        prv_size = input_size
        for layer_size in hidden_sizes:
            fc = nn.Linear(prv_size, layer_size)
            self.fc.extend((fc, nn.ReLU(), self.drop))
            prv_size = layer_size
            
        self.fc.extend((nn.Linear(prv_size, output_size), self.sigmoid))
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):
#         x = x.view(-1, self.input_size)
        return self.fc(x)
    
NP_CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ArvatoClassifier(*model_info)

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model.
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The variable `out_label` should be a rounded value, either 1 or 0
    out = model(data)
    out_np = out.cpu().detach().numpy()
    out_label = out_np

    return out_label


def save_model(model, model_dir, model_params):
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        torch.save(model_params, f)


    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    if torch.cuda.is_available():
        model.cuda()
    
def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device, model_dir, model_params):
    best_val_loss = 1e18
    for epoch in range(1, epochs + 1):
        st = time.time()
        model.train()
        total_loss = 0
        for batch in train_loader:      
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).view(-1, 1)

            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_loss += loss.data.item()
        
        et = time.time() - st

        model.eval()
        val_loss = 0
        for batch in val_loader:      
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).view(-1, 1)
            
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            
            val_loss += loss.data.item()
        
        total_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            save_model(model, model_dir, model_params)
            best_val_loss = val_loss
            
        print("Epoch: {}, train loss: {}, val loss: {}, training done in {}m {}s".format(epoch, total_loss, val_loss,
                                                                                      int(et/60), int(et - int(et/60)*60)))
    print('best_val_loss: {}'.format(best_val_loss))

        



def _get_data_loader(batch_size, training_dir, task, is_test = False):
    print("Get {} data loader from {}.".format(task[:-4], training_dir))
    
    training_dir = os.path.join(training_dir, task)
        
    train_data = pd.read_csv(training_dir, header=None, names=None)
    if is_test:
        train_X = torch.from_numpy(train_data.values).float()
        train_ds = torch.utils.data.TensorDataset(train_X)
        return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    else:
        train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
        train_X = torch.from_numpy(train_data.drop([0], axis=1).values).float()

        train_ds = torch.utils.data.TensorDataset(train_X, train_y)

        return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--input_size', type=int, default=32, metavar='N',
                        help='size of the input (default: 32)')
    parser.add_argument('--hidden_dim1', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--hidden_dim2', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--hidden_dim3', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--hidden_dim4', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--hidden_dim5', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--hidden_dim6', type=int, default=0, metavar='N',
                        help='size of the hidden dimension (default: 0)')
    parser.add_argument('--output_size', type=int, default=1, metavar='N',
                        help='size of output (default: 1)')
    parser.add_argument('--drop_prob', type=float, default=0.3, metavar='N',
                        help='drop probability (default: 0.3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate (default: 1e-3)')
    

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val-dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_data_loader(args.batch_size, args.train_dir, 'train.csv')
    val_loader = _get_data_loader(args.batch_size, args.val_dir, 'val.csv')

    # Build the model.
    hidden_dim = (args.hidden_dim1, args.hidden_dim2, args.hidden_dim3, args.hidden_dim4, args.hidden_dim5, args.hidden_dim6)
    hidden_dim = tuple(filter(lambda x: x > 0, hidden_dim))
    model_params = (args.input_size, hidden_dim,
                    args.output_size, args.drop_prob)
    model = ArvatoClassifier(*model_params).to(device)


    # Train the model.
    optimizer = optim.Adam(model.parameters(), args.lr)
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, val_loader, args.epochs, optimizer, loss_fn, device, args.model_dir, model_params)

