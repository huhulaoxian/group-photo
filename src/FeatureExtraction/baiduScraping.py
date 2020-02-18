from PIL import Image
import numpy as np
from aip import AipFace
import base64
import time
import logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='compare_baiduScraping.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel('DEBUG')

APP_ID = '17753712'
API_KEY = 'QsPydlU5sEAp3Wq9G1pslB5j'
SECRET_KEY = 'CyvQ4XGLT2Eq9U8mRlpEB8z3GQrIe2Il'

def req(filepath):
    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    with open(filepath,'rb') as f:
        base64_data=base64.b64encode(f.read())
    image = str(base64_data,'utf-8')
    imageType = "BASE64"
    """ 调用人脸检测 """
    client.detect(image, imageType)
    """ 如果有可选参数 """
    options = {}
    options["face_field"] = "expression,glasses,quality,emotion"
    options["max_face_num"] = 10
    options["face_type"] = "LIVE"
    time.sleep(1)
    """ 带参数调用人脸检测 """
    req_dict_face = client.detect(image, imageType, options)
    return(req_dict_face)


if __name__ == '__main__':
    path = "../../data/compare1/"
    good_indices = list(np.load('../../data/compare1_good_indices.npy'))
    req_dict_faces = []
    i = 1
    for image_name in good_indices:
        print("第",i,"张图片：",image_name)
        # filepath = path + image_name[0] 
        filepath = path + image_name
        req_dict_face = req(filepath)
        req_dict_faces.append(req_dict_face)
        i = i + 1
    print(len(req_dict_faces))
    np.save('../../data/compare1_req_dict_faces_baidu.npy', req_dict_faces)

    j = 0
    ind = []
    for index,req_face in enumerate(req_dict_faces):
        if(req_face["error_code"]!=0):
            req_dict_faces[index] = req(path + good_indices[index])
            ind.append(index)
            j = j + 1
    np.save('../../data/compare1_req_dict_faces_baidu_repair.npy', req_dict_faces)    
    logger.debug(ind)    
    print(j,"次请求失败")