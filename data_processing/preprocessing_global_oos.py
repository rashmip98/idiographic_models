import yaml
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    
    with open("/home1/r/rphadnis/idiographic_model/idiographic_models/parameters.yaml", "r") as stream:
        params_loaded = yaml.safe_load(stream)
    fields = ["userID", "stimulus", "response", "age", "race", "gender"]
    data_csv = pd.read_csv(params_loaded['data']['csv_path'], usecols=fields)
    data_csv['idCount'] = data_csv.groupby('userID', as_index=False)['userID'].transform('count')
    
    columns_list = data_csv.columns

    df_global_train = pd.DataFrame(columns = columns_list)
    df_global_val = pd.DataFrame(columns = columns_list)
    df_global_test = pd.DataFrame(columns = columns_list)
    df_global_train['response'].astype(float)
    df_global_val['response'].astype(float)
    df_global_test['response'].astype(float)
    
    unique_ids = data_csv['userID'].unique().tolist()  #438
    unique_imgs = data_csv['stimulus'].unique().tolist()  #500
    
    train_ids = random.sample(unique_ids, k=round(len(unique_ids) * 0.8))
    test_ids = random.sample([x for x in unique_ids if x not in train_ids], k=round(len(unique_ids) * 0.1))
    val_ids = [x for x in unique_ids if x not in train_ids and x not in test_ids]
    
    train_imgs = random.sample(unique_imgs, k=round(len(unique_imgs) * 0.8))
    test_imgs = random.sample([x for x in unique_imgs if x not in train_imgs], k=round(len(unique_imgs) * 0.1))
    val_imgs = [x for x in unique_imgs if x not in train_imgs and x not in test_imgs]
    
    train_ids_df = data_csv[data_csv['userID'].isin(train_ids)]
    test_ids_df = data_csv[data_csv['userID'].isin(test_ids)]
    val_ids_df = data_csv[data_csv['userID'].isin(val_ids)] 

    for img in train_imgs:
         df_id = train_ids_df.loc[train_ids_df['stimulus'] == img]
         idx = random.randint(0,df_id.shape[0]-1)
         df_global_train = pd.concat([df_global_train, df_id.iloc[[idx]]])
    
    for img in test_imgs:
         df_id = test_ids_df.loc[test_ids_df['stimulus'] == img]
         idx = random.randint(0,df_id.shape[0]-1)
         df_global_test = pd.concat([df_global_test, df_id.iloc[[idx]]])
    
    for img in val_imgs:
         df_id = val_ids_df.loc[val_ids_df['stimulus'] == img]
         idx = random.randint(0,df_id.shape[0]-1)
         df_global_val = pd.concat([df_global_val, df_id.iloc[[idx]]])

    # global normalized
    global_train_mean = df_global_train['response'].mean()
    global_train_std = df_global_train['response'].std()
    df_global_train['globalNormalizedResponse'] = df_global_train['response']
    df_global_train['globalNormalizedResponse'] -= global_train_mean
    df_global_train['globalNormalizedResponse'] /= global_train_std

    df_global_val['globalNormalizedResponse'] = df_global_val['response']
    df_global_val['globalNormalizedResponse'] -= global_train_mean
    df_global_val['globalNormalizedResponse'] /= global_train_std

    df_global_test['globalNormalizedResponse'] = df_global_test['response']
    df_global_test['globalNormalizedResponse'] -= global_train_mean
    df_global_test['globalNormalizedResponse'] /= global_train_std

    df_global_train.to_csv('/home1/r/rphadnis/idiographic_model/train_oos_ratings.csv', index=False, encoding='utf-8')
    df_global_val.to_csv('/home1/r/rphadnis/idiographic_model/val_oos_ratings.csv', index=False, encoding='utf-8')
    df_global_test.to_csv('/home1/r/rphadnis/idiographic_model/test_oos_ratings.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
