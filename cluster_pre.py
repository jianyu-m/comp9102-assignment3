import numpy
from sklearn.cluster import KMeans as kmeans
import math

N = 18576
k = 16
alpha = .1

def log_e(d, e):
    if d == 0:
        return 0
    else:
        return - d / e * math.log2(d / e)

def log_ce(wc, w, c, n):
    if wc == 0:
        return 0
    else:
        return wc / n * math.log2(n * wc / (w * c))


if __name__ == "__main__":
    p = numpy.load("parr")
    points = kmeans(k).fit(p)

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

    class_set = [{"Clinton": 0, "Trump": 0} for i in range(k)]

    total = 0
    for idx, p in enumerate(points.labels_):
        if idx not in labels:
            continue
        total += 1
        if labels[idx] == 0:
            class_set[p]["Clinton"] += 1
        else:
            class_set[p]["Trump"] += 1

    label_set = []
    for r in class_set:
        if r["Clinton"] > r["Trump"]:
            label_set.append("Clinton")
        else:
            label_set.append("Trump")

    purity = 0
    for idx, label in enumerate(label_set):
        purity += class_set[idx][label]
    purity = purity / total

    # compute entropy

    class_total = [c["Clinton"] + c["Trump"] for c in class_set]

    entropy = 0

    for c in class_total:
        entropy += log_e(c, total)

    # compute NMI
    clinton = 0
    trump = 0
    for c in class_set:
        clinton += c["Clinton"]
        trump += c["Trump"]

    HC = log_e(clinton, total) + log_e(trump, total)

    ioc = 0
    for idx, c in enumerate(class_set):
        ioc += log_ce(c["Trump"], class_total[idx], trump, total) + log_ce(c["Clinton"], class_total[idx], clinton, total)

    NMI = ioc / ((entropy + HC) / 2)

    print("purity " + str(purity))
    print("entropy " + str(entropy))
    print("nmi " + str(NMI))
