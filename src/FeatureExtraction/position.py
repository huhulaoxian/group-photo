import numpy as np 

# req_dict_faces = list(np.load('../../data/req_dict_faces.npy'))
req_dict_faces = list(np.load('../../data/test_req_dict_faces.npy'))
# print(req_dict_faces[193]['faces'])

#每张图中脸的数量
faces = []
meanpostions = []
for index,req_face in enumerate(req_dict_faces):
    print(index+1)
    face_number = len(req_face['faces'])
    positions = []
    for j in range(face_number):
        if j < 5:
            left = req_face['faces'][j]['face_rectangle']['left']
            width = req_face['faces'][j]['face_rectangle']['width']
            position = ((left + width)+left)/2
            print(position)
            positions.append(position)
        else:
            break
    # print(positions)
    meanpostion = np.array(positions)
    if len(meanpostion)>0:
        meanpostion = meanpostion.mean()
        print('人物中心点：',meanpostion)
        meanpostions.append(meanpostion)
    else:
        meanpostion = 0
        print('人物中心点：',meanpostion)
        meanpostions.append(meanpostion)
np.save('../../data/test_meanpostions.npy',meanpostions)
