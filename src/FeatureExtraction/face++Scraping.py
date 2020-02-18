import requests
from json import JSONDecoder
import cv2
import numpy as np
import csv 
import uuid

#url_face
http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
#face++ key and secret
key = "Oz_jL-oCbMAQF3xP6gI9fVCO0g93WY19"
secret = "pVggj4TmD_3z7fAbT5p3gbMYBXDEeStK"

path = "../../data/compare1/"
good_indices = list(np.load('../../data/compare1_good_indices.npy'))

#face_defect
data = {"api_key":key,"api_secret":secret,"return_landmark":1,
        "return_attributes":"smiling,eyestatus,mouthstatus,eyegaze,blur,headpose"}

req_dict_faces = []
i = 1
for image_name in good_indices:
    print("第",i,"张图片：",image_name)
    # filepath = path + image_name[0] 
    filepath = path + image_name
    files = {"image_file":open(filepath,"rb")}
    requests.adapters.DEFAULT_RETRIES = 999
    s = requests.session()
    s.keep_alive = False
    response_face = s.post(http_url,data = data,files = files)
    req_con_face = response_face.content.decode('utf-8')
    req_dict_face = JSONDecoder().decode(req_con_face)
    req_dict_faces.append(req_dict_face)
    print(response_face)
    print(len(req_dict_face['faces']))
    print("共检测出" + str(len(req_dict_face['faces'])) + "张脸")
    i = i + 1
print(len(req_dict_faces))
np.save('../../data/compare1_req_dict_faces.npy', req_dict_faces)