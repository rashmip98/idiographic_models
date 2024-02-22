import cv2 
import matplotlib.pyplot as plt
import yaml
import pandas as pd

def main():
    imgs_path= '/home1/r/rphadnis/idiographic_model/images'
    csv_path= '/home1/r/rphadnis/idiographic_model/train_ratings.csv'
    save_path = '/home1/r/rphadnis/idiographic_model/new_train_ratings.csv'
    
    # Importing Models and set mean values 
    face1 = "/home1/r/rphadnis/idiographic_model/age_gender_models/opencv_face_detector.pbtxt"
    face2 = "/home1/r/rphadnis/idiographic_model/age_gender_models/opencv_face_detector_uint8.pb"
    age1 = "/home1/r/rphadnis/idiographic_model/age_gender_models/age_deploy.prototxt"
    age2 = "/home1/r/rphadnis/idiographic_model/age_gender_models/age_net.caffemodel"
    gen1 = "/home1/r/rphadnis/idiographic_model/age_gender_models/gender_deploy.prototxt"
    gen2 = "/home1/r/rphadnis/idiographic_model/age_gender_models/gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 
    
    # Using models 
    # Face 
    face = cv2.dnn.readNet(face2, face1) 
    
    # age 
    age = cv2.dnn.readNet(age2, age1) 
    
    # gender 
    gen = cv2.dnn.readNet(gen2, gen1)

    # Categories of distribution 
    la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
    lg = ['Male', 'Female']

    data_csv = pd.read_csv(csv_path)
    n_imgs = data_csv.shape[0]
    # print(n_imgs)


    for i in range(n_imgs):
        img_path =imgs_path + '/' + data_csv.loc[i, 'stimulus'][36:44].lstrip("0")
        # print(img_path)
        image = cv2.imread(img_path) 
        # image = cv2.resize(image, (720, 640))
        
        if image is None:
            c
        # Face detection 
        fr_h = image.shape[0] 
        fr_w = image.shape[1] 
        # print(fr_h,fr_w)
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                    [104, 117, 123], True, False) 
        
        face.setInput(blob) 
        detections = face.forward()

        # Face bounding box creation 
        faceBoxes = [] 
        for i in range(detections.shape[2]): 
            
            #Bounding box creation if confidence > 0.7 
            confidence = detections[0, 0, i, 2] 
            if confidence > 0.7: 
                
                x1 = int(detections[0, 0, i, 3]*fr_w) 
                y1 = int(detections[0, 0, i, 4]*fr_h) 
                x2 = int(detections[0, 0, i, 5]*fr_w) 
                y2 = int(detections[0, 0, i, 6]*fr_h) 
                
                faceBoxes.append([x1, y1, x2, y2]) 
                
                cv2.rectangle(image, (x1, y1), (x2, y2), 
                            (0, 255, 0), int(round(fr_h/150)), 8)

        for faceBox in faceBoxes: 
      
            #Extracting face as per the faceBox 
            face_img = image[max(0, faceBox[1]-15): 
                        min(faceBox[3]+15, image.shape[0]-1), 
                        max(0, faceBox[0]-15):min(faceBox[2]+15, 
                                    image.shape[1]-1)] 
            
            #Extracting the main blob part 
            blob = cv2.dnn.blobFromImage( 
                face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False) 
            
            #Prediction of gender 
            gen.setInput(blob) 
            genderPreds = gen.forward() 
            gender = lg[genderPreds[0].argmax()] 
            
            #Prediction of age 
            age.setInput(blob) 
            agePreds = age.forward() 
            age_cat = la[agePreds[0].argmax()]

            data_csv.loc[i, "imgGender"] = gender
            data_csv.loc[i, "imgAge"] = age_cat

    data_csv.to_csv(save_path, index=False, encoding='utf-8')
            

if __name__ == '__main__':
    main()