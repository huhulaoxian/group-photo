import csv
import numpy as np

good_indices = list(np.load('../../data/good_indices.npy'))
reader = csv.reader(open("../../data/score_decimal.csv"))
# reader = csv.reader(open("../../data/image.csv"))
y = []
for image_name in good_indices:
    for row in reader:
        print(image_name)
        if(image_name == row[1]):
            print('-----------------------------------------------')
            y.append(float(row[2]))
            break
        else:
            continue
print(len(y))

np.save('../../data/score_decimal_change.npy', y)