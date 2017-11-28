import numpy
from sklearn.cluster import KMeans as kmeans
import math

N = 18576
alpha = .1

def log_e(d, e):
    if d == 0:
        return 0
    else:
        return - d * math.log2(d / e)

if __name__ == "__main__":
    p = numpy.load("parr")
    points = kmeans(2).fit(p)

    # compute purity
    labels = {}
    with open("Labels.txt") as f:
        line = " "
        while len(line) > 0:
            line = f.readline()
            if len(line) == 0:
                break
            point = int(line.split(" ")[0])
            label = int(line.split(" ")[1])
            labels[point] = label

    class_0 = {"Clinton": 0, "Trump": 0}
    class_1 = {"Clinton": 0, "Trump": 0}
    total = 0
    for idx, p in enumerate(points.labels_):
        if idx not in labels:
            continue
        total += 1
        if p == 0:
            if labels[idx] == 0:
                class_0["Clinton"] += 1
            else:
                class_0["Trump"] += 1
        else:
            if labels[idx] == 0:
                class_1["Clinton"] += 1
            else:
                class_1["Trump"] += 1

    if class_0["Clinton"] > class_0["Trump"]:
        class_0_label = "Clinton"
    else:
        class_0_label = "Trump"

    if class_1["Clinton"] > class_1["Trump"]:
        class_1_label = "Clinton"
    else:
        class_1_label = "Trump"

    purity = (class_0[class_0_label] + class_1[class_1_label]) / total



    # compute entropy

    class_0_total = 0
    for k, v in class_0.items():
        class_0_total += v

    class_1_total = 0
    for k, v in class_1.items():
        class_1_total += v

    entropy = log_e(class_0_total, total) + log_e(class_1_total, total)

    # compute NMI
    clinton = class_0["Clinton"] + class_1["Clinton"]
    trump = class_0["Trump"] + class_1["Trump"]
    HC = log_e(clinton, total) + log_e(trump, total)

    ioc = log_e(class_0["Clinton"], total) + log_e(class_0["Trump"], total) \
            + log_e(class_1["Trump"], total) + log_e(class_1["Clinton"], total)

    NMI = ioc / ((entropy + HC) / 2)

    print("purity " + str(purity))
    print("entropy " + str(entropy))
    print("nmi " + str(NMI))
