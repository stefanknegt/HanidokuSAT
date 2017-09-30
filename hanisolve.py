import pycosat as ps
import subprocess
from io import StringIO
import sys
import io
import contextlib
import ast
import numpy as np
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

#This list shows all the relationships between squares in a hanidoku
lists = [[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
 [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
 [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)],
 [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)],
 [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9)],
 [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8)],
 [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
 [(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6)],
 [(9, 1), (9, 2), (9, 3), (9, 4), (9, 5)],
 [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
 [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1)],
 [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 2), (7, 1)],
 [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 3), (7, 2), (8, 1)],
 [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)],
 [(2, 6), (3, 6), (4, 6), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)],
 [(3, 7), (4, 7), (5, 7), (6, 6), (7, 5), (8, 4), (9, 3)],
 [(4, 8), (5, 8), (6, 7), (7, 6), (8, 5), (9, 4)],
 [(5, 9), (6, 8), (7, 7), (8, 6), (9, 5)],
 [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1)],
 [(4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)],
 [(3, 1), (4, 2), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3)],
 [(2, 1), (3, 2), (4, 3), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4)],
 [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)],
 [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 6), (7, 6), (8, 6)],
 [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7), (7, 7)],
 [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8)],
 [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9)]]

#This function determines the average distance from the givens to the extremes (1 and 9)
def distribution(hints):
    n = len(hints)

    mean = sum(hints)/n
    dist_from_1_9 = [(9 - h) for h in hints if h >= 5] + [(h - 1) for h in hints if h < 5]

    variance = sum(dist_from_1_9) / n

    return n,variance

#This function gets the unit clauses for a given hanidoku
def getUnitClauses(input):
    hanidoku = input[5:]

    size = [5,6,7,8,9,8,7,6,5]

    n = 0
    units = []
    hints = []

    # n indicates the index in the original string
    for i,s in enumerate(size):
        for j in range(1,s+1):
            if hanidoku[n] != '0':
                units.append([(i+1)*100+j*10+int(hanidoku[n])])
                hints.append(int(hanidoku[n]))
            n += 1

    dist = distribution(hints)

    return units, dist

#This function determines all constraints (which are equal for every hanidoku)
def generateStandardClauses():
    clauses1 = []
    clauses2 = []
    clauses3 = []
    shape_small = [2,3,2]
    shape = [5,6,7,8,9,8,7,6,5]

    # Add all rules that make sure that if one cell contains a value it cannot contain any other value (uniqueness)
    for i in range(1,10):
    #for i in range(1,4):
        for j in range(1,shape[i-1]+1):
            clauses1.append( [int(str(i)+str(j)+str(q)) for q in range(1,10)] ) #Add clauses to make sure each field contains at least one value
            for x in range(1,10):
                for y in range(1,10):
                    if x != y:
                        clauses1.append([-int(str(i)+str(j)+str(x)),-int(str(i)+str(j)+str(y))])

    # Permutation rule
    for l in lists:
        length = len(l)
        for tup1 in l:
            for tup2 in l:
                if tup1 != tup2:
                    for x in range(1,10):
                        impossibleValues = set([q for q in range(1,10)]) - set([z for z in range(x-length+1, (x+length))])
                        impossibleValues.add(x)
                        for y in impossibleValues:
                            clauses3.append([-int(str(tup1[0])+str(tup1[1])+str(x)),-int(str(tup2[0])+str(tup2[1])+str(y))])

    clauses = clauses1+clauses3

    return clauses

#Solve all hanidokus in the dataset
def solve_all():
    with open('5200_easy_hanidoku.txt') as f:
        lines = f.readlines()

    standard_clauses = generateStandardClauses()

    for line in lines[0:5]:
        result = solve(line,standard_clauses)
        if result == 'UNSAT':
            break

#Solve a single hanidoku
def solve(input,standard_clauses):
    unit_clauses = getUnitClauses(input)

    all_clauses = standard_clauses + unit_clauses

    sol = timeit.timeit(ps.solve(all_clauses,verbose=1))

    if sol == 'UNSAT':
        final_sol = sol
    else:
        final_sol = [s for s in sol if s > 0 ]

#This function uses the output of Picosat (in a .txt file) to get the relevant data
def statistics():
    with open('stat.txt') as f:
        stats = ast.literal_eval(f.read())

    with open('dist.txt') as f2:
        raw_dists = list(map(lambda s: s.strip('\n'), f2.readlines()))
        dists = list(map(lambda s: ast.literal_eval(s),raw_dists))

    dists_with_stats = list(zip(dists,stats))

    sorted_stats = list(sorted(dists_with_stats, key=lambda dist: dist[0][1]))

    distance_to_conflicts = [(i[0][1],i[1][-1][6]) for i in sorted_stats]

    interval_mean = get_interval_mean(distance_to_conflicts,1.3,2.9,0.1)

    return interval_mean

#Helper function to get the mean of a distance measure interval
def get_interval_mean(data,i,n,di):
    interval_means = []

    while i < n:
        j = i+di
        interval = [d[1] for d in data if d[0]>i and d[0]<j]

        interval_mean = sum(interval)/len(interval)

        interval_means.append((j,len(interval),interval_mean))

        i = j

    return interval_means

#This function plots the given distance to time and conflicts and prints the correlation coefficients
def plot_and_correlation(data):
    sns.set_style("whitegrid")
    blue, = sns.color_palette("muted", 1)
    data2 = [(1.4000000000000001, 5, 0.005421113967895508),
 (1.5000000000000002, 16, 0.004512116312980652),
 (1.6000000000000003, 42, 0.00474136783963158),
 (1.7000000000000004, 132, 0.00472391735423695),
 (1.8000000000000005, 193, 0.0046310202445390925),
 (1.9000000000000006, 326, 0.00474408579750295),
 (2.0000000000000004, 916, 0.0047122206229830415),
 (2.1000000000000005, 628, 0.004806636625034794),
 (2.2000000000000006, 673, 0.004912112478338415),
 (2.3000000000000007, 680, 0.004994861518635469),
 (2.400000000000001, 631, 0.005118076094734688),
 (2.500000000000001, 544, 0.0052526983268120706),
 (2.600000000000001, 237, 0.005722900986168455),
 (2.700000000000001, 123, 0.0058767233437638944),
 (2.800000000000001, 33, 0.005575382348262902),
 (2.9000000000000012, 10, 0.006513237953186035)]
    interval = []
    conflicts = []
    occurences = []
    time = []
    for triple in data:
        if(not (triple[0] < 1.6 or triple[0] > 2.8)):
            interval.append(triple[0])
            occurences.append(triple[1])
            conflicts.append(triple[2])
    for triple in data2:
        if(not (triple[0] < 1.6 or triple[0] > 2.8)):
            time.append(triple[2])

    print(np.corrcoef(interval,time)) #Gives 88.40%
    plt.plot(interval,time)
    plt.ylabel('Time to solve')
    plt.xlabel('Distance measure')
    plt.show()

    print(np.corrcoef(interval,conflicts)) #Gives 87.51%
    plt.plot(interval,conflicts)
    plt.ylabel('Number of conflicts')
    plt.xlabel('Distance measure')
    plt.show()

    fig = plt.figure()
    n, bins, patches = plt.hist([interval, occurences])
    plt.show()
