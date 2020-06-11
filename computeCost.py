def computeCost(X,y,theta,alpha,iterations):

    cost = sum(((X@theta - y)**2))/(2*len(y))
    return cost
    #print(X,y,theta)