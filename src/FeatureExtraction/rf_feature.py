import numpy as np

all_feature = np.load('../../data/feature_vec/compare1_all_feature.npy')
# all_feature = all_feature[:,:7]
select_feature = np.load("rf_ranking.npy")
print(select_feature)
print(all_feature.shape)

new_feature = []
for i in select_feature[:20]:
    new_feature.append(all_feature[:,i])
new_feature = np.array(new_feature)
new_feature = np.transpose(new_feature)
print(type(new_feature))
print(new_feature.shape)
np.save('../../data/feature_vec/compare1_rf_all.npy',new_feature)