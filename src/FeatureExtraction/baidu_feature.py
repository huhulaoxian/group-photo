import numpy as np 


req_dict_faces = list(np.load('../../data/compare1_req_dict_faces_baidu_repair.npy'))

#每张图中脸的数量
faces = []
occlusions = []

for index,req_face in enumerate(req_dict_faces):
    print(index+1)
    if(req_face['result']==None):
        face_number = 0
        faces.append(face_number)
        occlusions.append(0)
        continue
    face_number = req_face['result']['face_num']
    faces.append(face_number)
    occlusion_number = 0

    for j in range(face_number):
        if j < 10:
          
            # occlusion
            left_eye = req_face['result']['face_list'][j]['quality']['occlusion']['left_eye']
            right_eye = req_face['result']['face_list'][j]['quality']['occlusion']['right_eye']
            nose = req_face['result']['face_list'][j]['quality']['occlusion']['nose']
            mouth = req_face['result']['face_list'][j]['quality']['occlusion']['mouth']
            left_cheek = req_face['result']['face_list'][j]['quality']['occlusion']['left_cheek']
            right_cheek = req_face['result']['face_list'][j]['quality']['occlusion']['right_cheek']
            chin_contour = req_face['result']['face_list'][j]['quality']['occlusion']['chin_contour']
            glasses = req_face['result']['face_list'][j]['glasses']['type']
            if (glasses == 'sun'):
                if (nose > 0.3 or mouth> 0.3 or left_cheek >0.3 or right_cheek > 0.3 or chin_contour > 0.3):
                    occlusion_number = occlusion_number+1
            else:
                if (left_eye > 0.5 or right_eye > 0.5 or nose > 0.3 or mouth> 0.3 or left_cheek >0.3 or right_cheek > 0.3 or chin_contour > 0.3):
                    occlusion_number = occlusion_number+1
        else:
            break 
    print("图中共" + str(face_number) + "人，" + str(occlusion_number) + "被遮挡")
    occlusions.append(occlusion_number)
    print("有这么多张图片",len(faces))
np.save('../../data/compare1_faces_baidu.npy',faces)
np.save('../../data/compare1_occlusion.npy',occlusions)
