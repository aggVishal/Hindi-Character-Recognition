# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:08:49 2018

@author: Sam
"""

#if want to load the saved model
from keras.models import load_model
from tkinter import filedialog,Tk
import numpy as np
from keras.preprocessing import image


model=load_model("HCRmodel.h5")
model.summary()

pred_to_label={
             0:'character_10_yna',
             1:'character_11_taamatar',
             2:'character_12_thaa',
             3:'character_13_daa',
             4:'character_14_dhaa',
             5:'character_15_adna',
             6:'character_16_tabala',
             7:'character_17_tha',
             8:'character_18_da',
             9:'character_19_dha',
             10:'character_1_ka',
             11:'character_20_na',
             12:'character_21_pa',
             13:'character_22_pha',
             14:'character_23_ba',
             15:'character_24_bha',
             16:'character_25_ma',
             17:'character_26_yaw',
             18:'character_27_ra',
             19:'character_28_la',
             20:'character_29_waw',
             21:'character_2_kha',
             22:'character_30_motosaw',
             23:'character_31_petchiryakha',
             24:'character_32_patalosaw',
             25:'character_33_ha',
             26:'character_34_chhya',
             27:'character_35_tra',
             28:'character_36_gya',
             29:'character_3_ga',
             30:'character_4_gha',
             31:'character_5_kna',
             32:'character_6_cha',
             33:'character_7_chha',
             34:'character_8_ja',
             35:'character_9_jha',
             36:'0',
             37:'1',
             38:'2',
             39:'3',
             40:'4',
             41:'5',
             42:'6',
             43:'7',
             44:'8',
             45:'9'
             }

# Making new predictions
root=Tk()
root.filename=filedialog.askopenfilename()
print(root.filename)
root.withdraw()

test_image=image.load_img(root.filename,target_size=(32,32))
test_image=image.img_to_array(test_image)   
test_image=np.expand_dims(test_image,axis=0)
pred_image=model.predict_classes(test_image)
pred_image
pred_to_label[pred_image[0]]

