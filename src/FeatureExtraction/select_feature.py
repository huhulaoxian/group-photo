import numpy as np

all_feature = np.load('../../data/feature_vec/all_feature_new.npy')
all_feature = all_feature[:,7:]
select_feature = np.load("svr_20.npy")
print(all_feature.shape)
new_feature = []
# for i in select_feature:
#     new_feature.append(all_feature[:,i])
###################
for index,i in enumerate(select_feature):
    if i == 1:
        new_feature.append(all_feature[:,index])
new_feature = np.array(new_feature)
new_feature = np.transpose(new_feature)
print(type(new_feature))
print(new_feature.shape)

np.save('../../data/feature_vec/svr_20.npy',new_feature)