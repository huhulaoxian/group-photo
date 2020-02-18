import WuFeatures
import cv2
import argparse
import csv
import MachajdikFeatures
import os
import numpy as np

def main():

    feature_attribute = [
     'pleasure','arousal','dominance','colofulness','black','silver',
     'gray','white','maroon','red','purple','fuchsia','green','lime','olive','yellow','navy',
     'blue','teal','aqua',
     'h_contrast','h_correlation','h_energy','h_homogeneity',
     's_contrast','s_correlation','s_energy','s_homogeneity',
     'v_contrast','v_correlation','v_energy','v_homogeneity',
     'len_statics',
     'degree_statics','abs_degree_statics','len_dynamics','degree_dynamics','abs_degree_dynamics','Level_of_detail'
    ]
    
    good_indices = list(np.load('../../../data/good_indices.npy'))
    path = "../../../data/GPD/"
    feature_vec = []
    for i, img_path in enumerate(good_indices):
        print ("图：",i+1)
        img = cv2.imread(path+img_path)

        resized_img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        feature_vector = []

        color = WuFeatures.Color()
        texture = WuFeatures.Texture()

        # The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance)
        wu_f1_3 = color.f1_3(img)
        for i in wu_f1_3:
            feature_vector.append(i)

        # Colorfulness
        feature_vector.append(color.f4(img))

        # W3C colors
        wu_f10_25 = color.f10_25(img)
        for i in wu_f10_25:
            feature_vector.append(i)

        # Gray-Level Co-occurance Matrix
        wu_f41_44 = texture.f41_44(img)
        for i in wu_f41_44:
            feature_vector.append(i)

        # Dynamic features(e.g 'len_statics','degree_statics','abs_degree_statics','len_dynamics','degree_dynamics','abs_degree_dynamics')
        dynamics_f45_50 = MachajdikFeatures.dynamics(img)
        for i in dynamics_f45_50:
            feature_vector.append(i)
            
        # Level of Details
        LOD = MachajdikFeatures.LevelOfDetail(img)
        feature_vector.append(LOD)
        feature_vec.append(feature_vector)
    print("共",len(feature_vec),"张图")
    print("共",len(feature_vec[0]),"维特征")
    np.save('../../../data/WuFeature.npy',feature_vec)
if __name__=="__main__":
    main()