import yaml
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def main():
    
    with open("parameters.yaml", "r") as stream:
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
    
            
    # df_global_train['seenFace'] = True
    # df_global_val['seenFace'] = False
    # df_global_test['seenFace'] = False
    
    # df_global_train['normalizedResponse'] = 0
    # df_global_val['normalizedResponse'] = 0
    # df_global_test['normalizedResponse'] = 0

    # less_than_400 = data_csv[data_csv.idCount < 400].groupby('userID', as_index=False).ngroups
    # train_count_400 = int(0.8*less_than_400)
    # val_count_400 = int(0.1*less_than_400)
    # print(less_than_400,train_count_400,val_count_400)
    
    unique_ids = data_csv['userID'].unique().tolist()
    # running_count = 0
    print(unique_ids[-10:])
    for id in unique_ids[:-10]:
        if id=='A3CRUP15LA6ZB':
                # this one user rated everyone 100
                continue
        df_id = data_csv.loc[data_csv['userID'] == id]
        count = df_id.shape[0]
        # if count>=400:
            # write it into df_global_train and dr_global_test,val->have a column which has tag seen face, unseen face in test,val
            # calculate & save the normalization per train, test, val values per identity
        train = int(0.8*count)
        val = int(0.1*count)

        df_global_train = pd.concat([df_global_train, df_id[:train]])
        df_global_val = pd.concat([df_global_val, df_id[train+1:train+1+val]])
        df_global_test = pd.concat([df_global_test, df_id[train+val+1:]])
            
        # df_global_train.loc[df_global_train['userID'] == id, 'seenFace'] = True
        # df_global_val.loc[df_global_val['userID'] == id, 'seenFace'] = True
        # df_global_test.loc[df_global_test['userID'] == id, 'seenFace'] = True
            
        train_mean = df_global_train.loc[df_global_train['userID'] == id, 'response'].mean()
        train_std = df_global_train.loc[df_global_train['userID'] == id, 'response'].std()
        df_global_train.loc[df_global_train['userID'] == id, 'userNormalizedResponse'] = df_global_train.loc[df_global_train['userID'] == id, 'response']
        df_global_train.loc[df_global_train['userID'] == id, 'userNormalizedResponse'] -= train_mean
        df_global_train.loc[df_global_train['userID'] == id, 'userNormalizedResponse'] /= train_std

        # val_mean = df_global_val.loc[df_global_val['userID'] == id, 'response'].mean()
        # val_std = df_global_val.loc[df_global_val['userID'] == id, 'response'].std()
        df_global_val.loc[df_global_val['userID'] == id, 'userNormalizedResponse'] = df_global_val.loc[df_global_val['userID'] == id, 'response']
        df_global_val.loc[df_global_val['userID'] == id, 'userNormalizedResponse'] -= train_mean
        df_global_val.loc[df_global_val['userID'] == id, 'userNormalizedResponse'] /= train_std

        # test_mean = df_global_test.loc[df_global_test['userID'] == id, 'response'].mean()
        # test_std = df_global_test.loc[df_global_test['userID'] == id, 'response'].std()
        df_global_test.loc[df_global_test['userID'] == id, 'userNormalizedResponse'] = df_global_test.loc[df_global_test['userID'] == id, 'response']
        df_global_test.loc[df_global_test['userID'] == id, 'userNormalizedResponse'] -= train_mean
        df_global_test.loc[df_global_test['userID'] == id, 'userNormalizedResponse'] /= train_std
            
        # else:
            # rest is ids which don't have 400+ data
            # for number of identities, save the some ids entirely in train and rest in val,test
            # this divides in some ids which are entirely unseen
            # save normalization per ids in each train,test,val

            # running_count +=1

            # if running_count<train_count_400:
            #     df_global_train = pd.concat([df_global_train, df_id])
            #     df_global_train.loc[df_global_train['userID'] == id, 'seenFace'] = False
            #     train_mean = df_global_train.loc[df_global_train['userID'] == id, 'response'].mean()
            #     train_std = df_global_train.loc[df_global_train['userID'] == id, 'response'].std()
            #     df_global_train.loc[df_global_train['userID'] == id, 'normalizedResponse'] = df_global_train.loc[df_global_train['userID'] == id, 'response']
            #     df_global_train.loc[df_global_train['userID'] == id, 'normalizedResponse'] -= train_mean
            #     df_global_train.loc[df_global_train['userID'] == id, 'normalizedResponse'] /= train_std
            # if train_count_400<=running_count<train_count_400+val_count_400:
            #     df_global_val = pd.concat([df_global_val, df_id])
            #     df_global_val.loc[df_global_val['userID'] == id, 'seenFace'] = False
            #     val_mean = df_global_val.loc[df_global_val['userID'] == id, 'response'].mean()
            #     val_std = df_global_val.loc[df_global_val['userID'] == id, 'response'].std()
            #     df_global_val.loc[df_global_val['userID'] == id, 'normalizedResponse'] = df_global_val.loc[df_global_val['userID'] == id, 'response']
            #     df_global_val.loc[df_global_val['userID'] == id, 'normalizedResponse'] -= val_mean
            #     df_global_val.loc[df_global_val['userID'] == id, 'normalizedResponse'] /= val_std
            # if train_count_400+val_count_400<=running_count:
            #     df_global_test = pd.concat([df_global_test, df_id])
            #     df_global_test.loc[df_global_test['userID'] == id, 'seenFace'] = False
            #     test_mean = df_global_test.loc[df_global_test['userID'] == id, 'response'].mean()
            #     test_std = df_global_test.loc[df_global_test['userID'] == id, 'response'].std()
            #     df_global_test.loc[df_global_test['userID'] == id, 'normalizedResponse'] = df_global_test.loc[df_global_test['userID'] == id, 'response']
            #     df_global_test.loc[df_global_test['userID'] == id, 'normalizedResponse'] -= test_mean
            #     df_global_test.loc[df_global_test['userID'] == id, 'normalizedResponse'] /= test_std
    
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

    df_global_train.to_csv('/home1/r/rphadnis/idiographic_model/train_ratings.csv', index=False, encoding='utf-8')
    df_global_val.to_csv('/home1/r/rphadnis/idiographic_model/val_ratings.csv', index=False, encoding='utf-8')
    df_global_test.to_csv('/home1/r/rphadnis/idiographic_model/test_ratings.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()