# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    ret = {}
    # loop over all 42 classes
    for c in range(0,43):
        print("reading class{}".format(c))
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.DictReader(gtFile, delimiter=';') # csv parser for annotations file
        # loop over all images in current annotations file
        for row in gtReader:
            x1 = int(row["Roi.X1"])
            y1 = int(row["Roi.Y1"])
            x2 = int(row["Roi.X2"])
            y2 = int(row["Roi.Y2"])
            im = plt.imread(prefix + row["Filename"])
            im = im[y1:y2, x1:x2,:]
            y, x, a= im.shape
            if 1 == 1:
                images.append(im) # the 1th column is the filename
                labels.append(row["ClassId"]) # the 8th column is the label
        gtFile.close()
    ret = {"features": images, "labels":labels}
    return ret

if __name__ == "__main__":
    ret = readTrafficSigns("/Users/neotrinity/Downloads/交通标志分类/GTSRB_final_training/Final_Training/Images")
    print(len(ret["features"]))
    with open("train_all.p", 'wb') as output:
        pickle.dump(ret,output)

