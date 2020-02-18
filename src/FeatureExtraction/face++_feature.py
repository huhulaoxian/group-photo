import numpy as np 

req_dict_faces = list(np.load('../../data/compare1_req_dict_faces.npy'))

#The face number of image
faces = []
smiles = []
eyes = []
blurs = []
covers = []
headposes = []
eyegazes = []
meanpostions = []

for index,req_face in enumerate(req_dict_faces):
    print(index+1)
    face_number = len(req_face['faces'])
    faces.append(face_number)
    smile_number = 0
    openeye_number = 0
    blur_number = 0
    cover_number = 0 
    headpose_number = 0
    eyegaze_number = 0
    positions = []

    for j in range(face_number):
        if j < 5:
            # smile
            smile = req_face['faces'][j]['attributes']['smile']
            if smile['value']>=50:
                smile_number = smile_number + 1
            # eye_open
            left_eye = req_face['faces'][j]['attributes']['eyestatus']['left_eye_status']
            right_eye = req_face['faces'][j]['attributes']['eyestatus']['right_eye_status']
            left_normal_open = left_eye['normal_glass_eye_open']
            left_no_open = left_eye['no_glass_eye_open']
            left_dark_open = left_eye['dark_glasses']
            right_normal_open = right_eye['normal_glass_eye_open']
            right_no_open = right_eye['no_glass_eye_open']
            right_dark_open = right_eye['dark_glasses']
            
            if (left_normal_open > 50 or left_no_open > 50 or left_dark_open > 50) or (right_normal_open >50 or right_no_open > 50 or right_dark_open >50):
                openeye_number = openeye_number+1
            
            #eyegaze
            l_no_glass_eye_close = req_face['faces'][j]['attributes']['eyestatus']['left_eye_status']['no_glass_eye_close']
            l_occlusion = req_face['faces'][j]['attributes']['eyestatus']['left_eye_status']['occlusion']
            l_normal_glass_eye_close = req_face['faces'][j]['attributes']['eyestatus']['left_eye_status']['normal_glass_eye_close']
            r_no_glass_eye_close = req_face['faces'][j]['attributes']['eyestatus']['right_eye_status']['no_glass_eye_close']
            r_occlusion = req_face['faces'][j]['attributes']['eyestatus']['right_eye_status']['occlusion']
            r_normal_glass_eye_close = req_face['faces'][j]['attributes']['eyestatus']['right_eye_status']['normal_glass_eye_close']
            headpose_yaw = req_face['faces'][j]['attributes']['headpose']['yaw_angle']
            bottom_y = req_face['faces'][j]['landmark']['mouth_lower_lip_top']['y']
            l_x = req_face['faces'][j]['landmark']['left_eye_center']['x']
            l_y = req_face['faces'][j]['landmark']['left_eye_center']['y']
            r_x = req_face['faces'][j]['landmark']['right_eye_center']['x']
            r_y = req_face['faces'][j]['landmark']['right_eye_center']['y']
            O_x = (l_x + r_x)/2
            O_y = (l_y + r_y)/2
            R = req_face['faces'][j]['face_rectangle']['width']
            v_l_x = req_face['faces'][j]['attributes']['eyegaze']['left_eye_gaze']['vector_x_component']
            v_l_y = req_face['faces'][j]['attributes']['eyegaze']['left_eye_gaze']['vector_y_component']
            v_r_x = req_face['faces'][j]['attributes']['eyegaze']['right_eye_gaze']['vector_x_component']
            v_r_y = req_face['faces'][j]['attributes']['eyegaze']['right_eye_gaze']['vector_y_component']
            D_x = (v_l_x + v_r_x)/2
            D_y = (v_l_y + v_r_y)/2
            P_x = O_x + D_x * R
            P_y = O_y + D_y * R
            if l_occlusion > 50 or r_occlusion > 50:
                continue
            if (l_no_glass_eye_close > 50 or l_normal_glass_eye_close >50) or (r_no_glass_eye_close > 50 or r_normal_glass_eye_close >50):
                continue
            if headpose_yaw > 30 or headpose_yaw < -30:
                continue
            if P_y < bottom_y and (P_x < r_x and P_x > l_x):
                eyegaze_number = eyegaze_number + 1

            #cover
            if l_occlusion > 50 or r_occlusion > 50:
                cover_number = cover_number + 1

            #position
            left = req_face['faces'][j]['face_rectangle']['left']
            width = req_face['faces'][j]['face_rectangle']['width']
            position = ((left + width)+left)/2
            positions.append(position)

            #Blur
            blur = req_face['faces'][j]['attributes']['blur']['blurness']
            if blur['value'] >= 50:
                blur_number = blur_number + 1
            
            #headpose
            headpose_yaw = req_face['faces'][j]['attributes']['headpose']['yaw_angle']
            if headpose_yaw < 10 or headpose_yaw > -10:
                headpose_number =  + 1
        else:
            break
    meanpostion = np.array(positions)
    if len(meanpostion)>0:
        meanpostion = meanpostion.mean()
        print('人物中心点：',meanpostion)
        meanpostions.append(meanpostion)
    else:
        meanpostion = 0
        print('人物中心点：',meanpostion)
        meanpostions.append(meanpostion)    
    print("图中共" + str(face_number) + "人，" + str(openeye_number) + "人睁眼"+ str(smile_number) + "人微笑"+ str(blur_number) + "人模糊"+ str(eyegaze_number) + "人看镜头")
    smiles.append(smile_number)
    eyes.append(openeye_number)
    blurs.append(blur_number)
    headposes.append(headpose_number)
    covers.append(cover_number)
    eyegazes.append(eyegaze_number)    
np.save('../../data/compare1_faces.npy',faces) 
np.save('../../data/compare1_smiles.npy',smiles)
np.save('../../data/compare1_eyes_new.npy',eyes)
np.save('../../data/compare1_blurs.npy',blurs)
np.save('../../data/compare1_covers.npy',covers)
np.save('../../data/compare1_headposes.npy',headposes)
np.save('../../data/compare1_eyegazes.npy',eyegazes)
np.save('../../data/compare1_meanpostions.npy',meanpostions)