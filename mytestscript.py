# load_model_sample.py
# from keras.models import load_model
# from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import csv
import tensorflow as tf
import shutil  
import random


ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,default='',help='path')
# ap.add_argument('-c','--csv',type=str,default='',help='path')
ap.add_argument('-m','--model',type=str,default='',help='path')
ap.add_argument('-d','--destination', type=str, default='true',
 help="path to destination folder")
args=vars(ap.parse_args())

dest = args["destination"]
image_path=args['image']

def preprocessing_img(img_path):
    #Converting the image to tensor
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(224, 224),interpolation = "lanczos")
    #this function Loads the image and resizes it to the specified size using PIL(Nearest Neighbour)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #this function converts the image to numpy Array
    img_array = np.expand_dims(img_array, axis=0)
    #(1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_array /= 255.
    #the numpy array are normalized between 0 and 1
    return img_array

#def load_image(img_path, show=False):

    #img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    #img_tensor = tf.keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
    #img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    # if show:
    #     plt.imshow(img_tensor[0])                           
    #     plt.axis('off')
    #     plt.show()
    #return img_tensor
ran_val=random.random()
str_ran_value=str(ran_val)

messi = args["model"]
model = tf.keras.models.load_model(messi)
images=os.listdir(image_path)
fi=open(str_ran_value,"x")
a=[]
count_of_total_images=0
count_of__FP =0
for each_image in images:
    count_of_total_images+=1
   # print("messi")
   # print(each_image)
    
    try:
        # print(str(each_image))
        img_path = os.path.join(image_path,each_image)
       # print("img_path is:",img_path)
        new_image = preprocessing_img(img_path)
        pred = model.predict(new_image)
        print(each_image , pred)
        if pred[0][0]<0.5:
            #val=pred[0][0]
            a.append(each_image)
            #fi.write( 'value = ' + val + '\n' )
                
            
            
        else:
            continue

        # with open(args['csv'],'a',newline='') as csvfile:
        #     fieldnames=['image_name','pred']
        #     writer=csv.DictWriter(csvfile,fieldnames)
        #     writer.writerow({'image_name':each_image,'pred':pred})
    except:
        print(each_image)
        continue
#print(a)
for i in a:
    s=str(i)
    #fi.write("\n")
    fi.write(s)
    fi.write(os.linesep)
fi.close()
#data moving script
src=image_path
fil=open(str_ran_value)
lines = fil.readlines()
for file_name in lines:
    count_of__FP+=1
    try:
       # print(file_name)
        #source = os.path.join(src,file_name)
        
        source=src+'/'+file_name
        #print(source)
        val=shutil.move(source[:-1], dest)
        #print(val)
        #os.system ("mv"+ " " + source + " " + dst)
    except:
        continue

fil.close()
print("text file created is:",str_ran_value)
print("total images processed are:",count_of_total_images)
print("total FP's are/total images moved to destination are",count_of__FP)
