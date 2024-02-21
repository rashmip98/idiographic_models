import yaml
import torch
from model import IdiographicClassifier
from dataset import BuildDataset, BuildDataloader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("/home1/r/rphadnis/idiographic_model/idiographic_models/parameters.yaml", "r") as stream:
        params_loaded = yaml.safe_load(stream)

    train_data_csv = pd.read_csv(params_loaded['data']['train_csv_path'])
    val_data_csv = pd.read_csv(params_loaded['data']['val_csv_path'])
    test_data_csv = pd.read_csv(params_loaded['data']['test_csv_path'])

    features = np.load(params_loaded['data']['features_path'],allow_pickle='TRUE').item()
    # unique_ids = data_csv['userID'].unique().tolist()    
    
    if params_loaded['test']==False:
        # pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', params_loaded['model']['pretrained'], pretrained=True)
        # feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1*params_loaded['model']['layer']])
        # feature_extractor.eval()
        model = IdiographicClassifier(params_loaded)
        optimizer = torch.optim.SGD(model.parameters(), lr=params_loaded['train']['lr'], momentum=params_loaded['train']['momentum'],weight_decay=params_loaded['train']['wt_decay'])
   
        if params_loaded['model']['individual']:
            pass
            # for user in unique_ids:
            #     df = data_csv.loc[data_csv['userID'] == user]

            #     avg_train_loss, avg_val_loss = train_loop(model, feature_extractor,optimizer, df, params_loaded)

            #     if avg_val_loss < best_vloss:
            #         best_vloss = avg_val_loss
            #         model_path = 'ind_{}'.format(user)
            #         torch.save(model.state_dict(), params_loaded['save']+'/'+model_path)
        
        else:
            train_loop(model, optimizer, train_data_csv, val_data_csv, test_data_csv, features, params_loaded)
            
            torch.save(model.state_dict(), params_loaded['save']+'/'+'global.pth')


def train_loop(model, optimizer, train_df, val_df, test_df, features, params_loaded):
    trainset = BuildDataset(train_df, features, params_loaded)
    valset = BuildDataset(val_df, features, params_loaded)
    testset = BuildDataset(test_df, features, params_loaded)
    # full_size = len(dataset)
    # train_size = int(full_size * 0.8)
    # test_size = int((full_size - train_size)/2)
    # val_size = full_size - train_size - test_size
    
    # trainset, valset, testset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_build_loader = BuildDataloader(trainset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()

    val_build_loader = BuildDataloader(valset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = val_build_loader.loader()
    train_loss=  []
    val_loss = []
    for epoch in range(params_loaded['train']['epochs']):
        model.train()
        train_epoch_loss = []
        for i,batch in enumerate(train_loader,0):
            
            feats = batch['features']
            ratings = batch['ratings']

            optimizer.zero_grad()

            # Make predictions for this batch
            # feats = feature_extractor(imgs)
            outputs = model(feats)

            # Compute the loss and its gradients
            loss = model.compute_loss(torch.squeeze(outputs), ratings)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_epoch_loss.append(loss.item())
        train_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))

        model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                feats = batch['features']
                ratings = batch['ratings']
                # feats = feature_extractor(imgs)
                out = model(feats)
                vloss = model.compute_loss(torch.squeeze(out), ratings)
                val_epoch_loss.append(vloss)
        val_loss.append(sum(val_epoch_loss)/len(val_epoch_loss))

    plt.plot(np.linspace(1,params_loaded['train']['epochs'],params_loaded['train']['epochs']).astype(int), train_loss, label='train')
    plt.plot(np.linspace(1,params_loaded['train']['epochs'],params_loaded['train']['epochs']).astype(int), val_loss, label='val')  
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")  
    plt.savefig(params_loaded['save']+"loss.jpg")

if __name__ == '__main__':
    main()
