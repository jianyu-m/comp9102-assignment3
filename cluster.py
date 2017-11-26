import numpy
from sklearn.cluster import KMeans as kmeans
import math

N = 18576
alpha = .1

def log_e(d, e):
    if d == 0:
        return 0
    else:
        return - d * math.log(d / e)

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

#    N = 100

    # arr = numpy.array([[0, 1, 0, 1, 1],
    #                    [1, 0, 0, 0, 1],
    #                    [0, 0, 0, 1, 1],
    #                    [1, 0, 1, 0, 0],
    #                    [1, 1, 1, 0, 0]])

    A = numpy.zeros((N, N))

    sum_all = [sum(arr[i]) for i in range(N)]

    for i in range(N):
        for j in range(N):
            if arr[i][j] == 1:
                A[i][j] = 1 / sum_all[j]
            else:
                A[i][j] = 0

    e = numpy.zeros((N, N))
    p = numpy.ones((N, N))
    for i in range(N):
        for j in range(N):
            if arr[i][j] == 1:
                e[i][j] = 1 / sum_all[i]
            else:
                e[i][j] = 0

    itr = 10

    convergence = 10e-5
    tmp = 100

    while tmp > convergence:
        p_now = (1 - alpha) * numpy.dot(A, p) + alpha * e
        tmp = sum(sum((p_now - p)**2))
        p = p_now

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

    ioc = log_e(class_0["Clinton"], total) + log_e(class_0["Trump"], total) \
            + log_e(class_1["Trump"], total) + log_e(class_1["Clinton"], total)

    NMI = ioc / ((entropy + HC) / 2)

    print("purity " + str(purity))
    print("entropy " + str(entropy))
    print("nmi " + str(NMI))
