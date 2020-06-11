from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iterations):

    j_hist = []

    m = len(y)

    for i in range(iterations):
        a = X@theta - y
        theta = theta - (alpha/m)*(a.T @ X).T
        j_hist.append(computeCost(X,y,theta,alpha,iterations))

    return theta,j_hist