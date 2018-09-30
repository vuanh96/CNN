from math import sqrt
from itertools import combinations
from random import seed, randint


def distance(key):
    a, b = key
    x = a[0] - b[0]
    y = a[1] - b[1]
    return sqrt(x * x + y * y)


def brute_force(array):
    temp = min(combinations(array, 2), key=distance)
    point1, point2 = temp
    mi = distance(temp)

    return point1, point2, mi


def generate_points(num_points, ceil):
    testcase = [(randint(0, ceil), randint(0, ceil)) for _ in range(num_points)]
    return testcase


def sort_array(arr):
    ax = sorted(arr, key=lambda x: x[0])
    ay = sorted(arr, key=lambda x: x[1])

    return ax, ay


def min_dist(x_sorted_arr, y_sorted_arr):
    leng_x_sorted_arr = len(x_sorted_arr)
    if leng_x_sorted_arr <= 3:
        return brute_force(x_sorted_arr)

    mid_x_sorted_array_index = leng_x_sorted_arr // 2
    Lx = x_sorted_arr[:mid_x_sorted_array_index]
    Rx = x_sorted_arr[mid_x_sorted_array_index:]

    x_midpoint = x_sorted_arr[mid_x_sorted_array_index][0]
    Ly = list()
    Ry = list()
    for p in y_sorted_arr:
        if p[0] <= x_midpoint:
            Ly.append(p)
        else:
            Ry.append(p)

    #    After splitting phase, call recursively both arrays

    (p1, q1, min1) = min_dist(Lx, Ly)
    (p2, q2, min2) = min_dist(Rx, Ry)

    if min1 <= min2:
        delta = min1
        pq = (p1, q1)
    else:
        delta = min2
        pq = (p2, q2)

    # investigate in the distance of points in 2 splitted parts (on the boundary)
    (p3, q3, min3) = closest_pair_in_denta_region(x_sorted_arr, y_sorted_arr, delta, pq)

    if delta <= min3:
        return pq[0], pq[1], delta
    else:
        return p3, q3, min3


def closest_pair_in_denta_region(x_sorted_arr, y_sorted_arr, delta, best_pair):
    leng_x = len(x_sorted_arr)
    x_midpoint = x_sorted_arr[leng_x // 2][0]
    #     Create sub region of points from midpoint - delta  to midpoint + delta
    sub_y = [p for p in y_sorted_arr if x_midpoint - delta <= p[0] <= x_midpoint + delta]
    best_dist = delta
    leng_sub_y = len(sub_y)
    for i in range(leng_sub_y - 1):
        for j in range(i + 1, min(i + 7, leng_sub_y)):
            p, q = sub_y[i], sub_y[j]
            key = p, q
            dist = distance(key)
            if dist < best_dist:
                best_pair = p, q
                best_dist = dist

    return best_pair[0], best_pair[1], best_dist


def testcase_results(algorithm, testcase):
    if algorithm == 'brute_force':
        p1, p2, min = brute_force(testcase)
        print(
            "In this testcase, When using the brute force algorithm, the closest points are {} {} with their distance is {}".format(
                p1, p2, min))
    if algorithm == 'optimal':
        ax, ay = sort_array(testcase)
        p1, p2, min = min_dist(ax, ay)
        print(
            "In this testcase, When using the optimal algorithm, the closest points are {} {} with their distance is {}".format(
                p1, p2, min))


# Test test cases
# Test case 1
testcase1 = generate_points(100, 100)

# # Test case 2
testcase2 = generate_points(200, 200)

# # Test case 1
testcase3 = generate_points(300, 300)

# # Test case 1
testcase4 = generate_points(400, 400)

# # Test case 1
testcase5 = generate_points(500, 500)

# 2.1 Test for brute force algorithm
testcase_results('brute_force', testcase1)

# 2.2 Test for optimal closest pair algorithm
testcase_results('optimal', testcase1)
