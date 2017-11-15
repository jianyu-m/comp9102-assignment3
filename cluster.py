import numpy
from sklearn.cluster import KMeans as kmeans

N = 18576
alpha = .1
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

    N = 100

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

    points = kmeans(10).fit(p)

    print(points.cluster_centers_)