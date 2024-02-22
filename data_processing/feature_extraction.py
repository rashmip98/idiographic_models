from PIL import Image
from torchvision import transforms, models
import numpy as np
import pandas as pd

def main():
    imgs_path= '/home1/r/rphadnis/idiographic_model/images'
    csv_path= '/home1/r/rphadnis/idiographic_model/ratings.csv'
    save_path = '/home1/r/rphadnis/idiographic_model/'

    data_csv = pd.read_csv(csv_path)

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize(224),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    result = {}

    feature_extractor = models.inception_v3(pretrained=True).eval()
    for param in feature_extractor.parameters():
            param.requires_grad = False

    unique_imgs = data_csv['stimulus'].unique().tolist()
    for i, img in enumerate(unique_imgs):
        unique_imgs[i] = imgs_path + "/" + img[36:44].lstrip("0") 
        image = Image.open(unique_imgs[i])
        image = preprocess(image)
        result[img[36:44].lstrip("0")] = image
    
    for key,val in result.items():
        val = val.unsqueeze(0)
        feat = feature_extractor(val)
        result[key] = feat
    
    # shape is 1X1000 for 1 feature
    np.save(save_path+'features.npy', result) 

    # # Load
    # read_dictionary = np.load(save_path+'features.npy',allow_pickle='TRUE').item()


        
    

if __name__ == '__main__':
    main()
