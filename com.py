import numpy
from sklearn.cluster import KMeans as kmeans
import math

N = 18576
alpha = .1

def log_e(d, e):
    if d == 0:
        return 0
    else:
        return - d / e * math.log(d / e)

def log_ce(wc, w, c, n):
    if N == 0:
        return 0
    else:
        return wc / n * math.log2(n * wc / (w * c))


if __name__ == "__main__":
    arr = numpy.zeros((N, N))
    with open("Graph.txt") as f:
        line = " "
        while len(line) > 0:
            line = f.readline()
            if len(line) <= 0:
                break
            linarr = line.split(" ")
            fr, to = int(linarr[0]), int(linarr[1])
            arr[fr - 1][to - 1] = 1
            arr[to - 1][fr - 1] = 1

    p = numpy.zeros((N, N))

    for i in range(0, N):
        for j in range(i + 1, N):
            same = 0
            sor = 0
            for z in range(N):
                if arr[i][z] == 1 and arr[j][z] == 1:
                    same += 1
                    sor += 1
                elif arr[i][z] == 1 or arr[j][z] == 1:
                    sor += 1
            if sor == 0:
                print("problem")
            p[i][j] = same / sor
            p[j][i] = same / sor

    p.dump("com")
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
        total += 1
        if idx not in labels:
            continue
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

    ioc = log_ce(class_0["Trump"], class_0_total, trump, total) \
            + log_ce(class_0["Clinton"], class_0_total, clinton, total) \
            + log_ce(class_1["Trump"], class_1_total, trump, total) \
            + log_ce(class_1["Clinton"], class_1_total, clinton, total)

    NMI = ioc / ((entropy + HC) / 2)

    print("purity " + str(purity))
    print("entropy " + str(entropy))
    print("nmi " + str(NMI))
