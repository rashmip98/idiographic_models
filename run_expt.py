import yaml
import torch
from model import IdiographicClassifier
from dataset import BuildDataset, BuildDataloader
import pandas as pd


def main():
    with open("parameters.yaml", "r") as stream:
        params_loaded = yaml.safe_load(stream)

    data_csv = pd.read_csv(params_loaded['data']['csv_path'])
    unique_ids = data_csv['userID'].unique().tolist()
    
    # img_nms = data_csv['stimulus'].tolist()
    # img_paths = []
    # for img in img_nms:
    #     img_paths.append(params_loaded['data']['imgs_path']+'/'+img[36:44])
    
    if params_loaded['test']==False:
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', params_loaded['model']['pretrained'], pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1*params_loaded['model']['layer']])
        feature_extractor.eval()
        model = IdiographicClassifier(params_loaded, pretrained_model.fc.in_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=params_loaded['train']['lr'], momentum=params_loaded['train']['momentum'])
        best_vloss = 1000000.
   
        if params_loaded['individual']:
            for user in unique_ids:
                df = data_csv.loc[data_csv['userID'] == user]

                avg_train_loss, avg_val_loss = train_loop(model, feature_extractor,optimizer, df, params_loaded)

                if avg_val_loss < best_vloss:
                    best_vloss = avg_val_loss
                    model_path = 'ind_{}'.format(user)
                    torch.save(model.state_dict(), params_loaded['save']+'/'+model_path)
        
        else:
            avg_train_loss, avg_val_loss = train_loop(model, feature_extractor, optimizer, data_csv, params_loaded)
            
            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                model_path = 'global'
                torch.save(model.state_dict(), params_loaded['save']+'/'+model_path)


def train_loop(model, feature_extractor, optimizer, df, params_loaded):
    dataset = BuildDataset(df, params_loaded)
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = int((full_size - train_size)/2)
    val_size = full_size - train_size - test_size
    
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_build_loader = BuildDataloader(trainset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()

    val_build_loader = BuildDataloader(valset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = val_build_loader.loader()

    for epoch in params_loaded['train']['epochs']:
        model.train()
        for i,batch in enumerate(train_loader,0):
            imgs = batch['images']
            ratings = batch['ratings']

            optimizer.zero_grad()

            # Make predictions for this batch
            feats = feature_extractor(imgs)
            outputs = model(feats)

            # Compute the loss and its gradients
            loss = model.compute_loss(outputs, ratings)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_train_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        model.eval()
        running_vloss = 0.
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                imgs = batch['images']
                ratings = batch['ratings']
                feats = feature_extractor(imgs)
                out = model(feats)
                vloss = model.compute_loss(out, ratings)
                running_vloss +=vloss
        avg_vloss = running_vloss / (i+1)

        #return last_loss, avg_vloss

if __name__ == '__main__':
    main()
