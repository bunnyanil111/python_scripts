import os
import pandas as pd 
import numpy as np 
from shutil import copyfile
import cv2

	 
import os
outPath = "/home/shailu/Pictures/JIO_ID_Spoof/DKYC_BB_Full/"
outPath2 = "/home/shailu/Pictures/JIO_ID_Spoof/ii2/"

rootdir = os.getcwd()+"/dkyc_aps_29june/"

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        #print(filepath)
        img = cv2.imread(filepath)
        if filepath.endswith("_image.png"):
        	cv2.imwrite(outPath+file,img)
        else:
         	cv2.imwrite(outPath2+file,img)	

       