#!/usr/bin/python3

from sklearn.model_selection import train_test_split
import glob

rgb = glob.glob("/home/previato/Dropbox/IC/dataset/correct_20m_4/rgb/*.png")
depth = glob.glob("/home/previato/Dropbox/IC/dataset/correct_20m_4/depth/*.png")

rgb_depth = list(zip(rgb, depth))
# print(rgb_depth)

x_train, x_test = train_test_split(rgb_depth, test_size=0.2)
print("x_train:", len(x_train))
print("x_test:", len(x_test))

with open("/home/previato/StructuredAttentionDepthEstimation/StructuredAttentionDepthEstimation/utils/filenames/ssnda_train_pairs.txt", "w") as f:
    f.writelines("%s %s\n" % (l[0], l[1]) for l in x_train)

with open("/home/previato/StructuredAttentionDepthEstimation/StructuredAttentionDepthEstimation/utils/filenames/ssnda_test_pairs.txt", "w") as f:
    f.writelines("%s %s\n" % (l[0], l[1]) for l in x_test)
