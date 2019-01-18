'''

Isadora Salvetti, January 2019
GPR final - Implementation of question 1.

Packages needed:

- numpy
- scipy
- matplotlib

'''


from scipy.spatial import distance
from Ex1.Plot import *
PI = 3.14159265359


# Data points
def getPoints():
    p = []
    for i in range(11):
        for j in range(21):
            x = (10+i)/10
            y = j/10
            z = (np.sin(PI*x)*np.cos(PI*y)) / (1 + x*x*y*y)
            p.append((x, y, z))
    return p


def getAvrageDist(p):
    dist = 0
    for Point1 in p:
        for Point2 in p:
            dist += distance.euclidean(Point1, Point2)
    dist = dist / len(p)
    return dist


def getw(s, adist):
        w = 0.00000000000001
        if s < radius:
            w = np.exp(- (s * s / (adist * adist)))
        # print(w)
        return w


def reductionPlace(adist, refP, p):
    tmp_A , tmp_B = [], []
    for point in p:
        w = getw(distance.euclidean(refP, point), adist)
        tmp_A.append([point[0]*w, point[1]*w, 1])
        tmp_B.append(point[2]*w)
    b = np.matrix(tmp_B).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b

    # print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    return fit.item(0), fit.item(1), fit.item(2)


def localApproximation(xy, height, adist):
    tmp_A, tmp_B = [], []
    for i in range(len(xy)):
        w = getw(np.linalg.norm(np.array(xy[i])), adist)
        tmp_A.append(np.array([xy[i][0], xy[i][1], 1])*w)
        tmp_B.append(height[i]*w)
    b = np.matrix(tmp_B).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    # print(fit)
    return fit.item(0), fit.item(1), fit.item(2)


def getOrthogonalProjection(points, plane):
    # Point in plane
    plpx, plpy = 1, 1
    plpz = plane[0]*plpx + plane[1]*plpy + plane[2]
    plpoint = np.array([plpx, plpy, plpz])

    pn = np.array([plane[0], plane[1], -1])
    pn = pn / np.linalg.norm(pn)

    def projectPoint(point):
        pt = np.array(point)
        pq = pt - plpoint
        lenProjPq = (np.dot(pq, pn)).item(0)
        projPoint = (pt - lenProjPq * pn)

        return projPoint.item(0), projPoint.item(1), projPoint.item(2)

    projectedPoints = []
    for point in points:
        p = projectPoint(point)
        projectedPoints.append(p)

    return projectedPoints


def getOrthogonalProjectionSingle(point, plane):
    # Point in plane
    plpx, plpy = 1, 1
    plpz = plane[0]*plpx + plane[1]*plpy + plane[2]
    plpoint = np.array([plpx, plpy, plpz])

    pn = np.array([plane[0], plane[1], -1])
    pn = pn / np.linalg.norm(pn)

    def projectPoint(p):
        pt = np.array(p)
        pq = pt - plpoint
        lenProjPq = (np.dot(pq, pn)).item(0)
        projPoint = (pt - lenProjPq * pn)

        return projPoint.item(0), projPoint.item(1), projPoint.item(2)

    return projectPoint(point)


def ChangeCoordinates(origin, plane, points):
    pn = np.array([plane[0], plane[1], -1])
    pn = pn / np.linalg.norm(pn)
    e1 = np.cross(pn, np.array([0, 0, 1]))
    e2 = np.cross(pn, e1)
    changeMat = np.array([e1, e2, pn]).T

    # print("Matrix= \n", changeMat)
    changeMat = np.linalg.inv(changeMat)

    def change(p):
        tPoint = np.array(p)-np.array(origin)
        myPoint = np.dot(changeMat, tPoint)
        # print("Point= from: ", np.array(p), "to: ", myPoint)
        return myPoint.item(0), myPoint.item(1)

    cPoints = []
    for p in points:
        cPoints.append(change(p))

    return cPoints


# Data
refPoint = (1.5, 1.0, 0.4)
points = getPoints()
# max distance to consider a point:
radius = 1
# average distance between points:
avrgDistance = getAvrageDist(points)

# 1) Compute weighted reduction plane Hr
plane = reductionPlace(avrgDistance, refPoint, points)
pn = np.array([plane[0], plane[1], -1])
pn = pn / np.linalg.norm(np.array(pn))
# Display points and plane
plotPointsPlane(points, plane, refPoint)

# 2) Compute projections of original points on Hr
planeProjections = getOrthogonalProjection(points, plane)
projRefPoint = getOrthogonalProjectionSingle(refPoint, plane)
# Display points and projection
plotPointsPlane(planeProjections, plane, projRefPoint)

# 3) Compute points represented in Hr coordinate system
# And height field
heights = []
pd = plane[2]
planeCoordinates = ChangeCoordinates(projRefPoint, plane, planeProjections)
for p in points:
    heights.append(np.dot(np.array(pn), np.array(p))-pd)

# 3) Compute weighted local approximation on height
heightLineAprox = localApproximation(planeCoordinates, heights, avrgDistance)
# r = q + p(0)
finalPoint = projRefPoint + heightLineAprox[2]*pn
# Display final point
plotSolution(points, refPoint, finalPoint)

'''

Sources:
Regression plane and matplotlib usage examples: https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
More on linear regression: https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/
Orthogonal projection: https://www.youtube.com/watch?v=HW3LYLLc60I
MLS: Mesh Independent Surface Interpolation, David Levin

Discussed with other students on the master and Louis Clergue.

'''
