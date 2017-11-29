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

    all_set = []
    for i in range(N):
        se = set()
        for j in range(N):
            if arr[i][j] != 0:
                se.add(j)
        all_set.append(se)
    print("init")
    for i in range(0, N):
        print(str(i))
        for j in range(i + 1, N):
            same = len(all_set[i] & all_set[j])
            sor = len(all_set[i] | all_set[j])
            if sor == 0:
                tmp = 1
            else:
                tmp = same / sor
            p[i][j] = tmp
            p[j][i] = tmp

    p.dump("com")
