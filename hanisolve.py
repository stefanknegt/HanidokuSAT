# -*- coding: UTF-8 -*-

import pycosat as ps
import ast
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as gui
import time 


# Helper functions

def distribution(hints):
    n = len(hints)

    mean = sum(hints)/n
    dist_from_1_9 = [(9 - h) for h in hints if h >= 5] + [(h - 1) for h in hints if h < 5]

    variance = sum(dist_from_1_9) / n

    return n,variance

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

    #print(dist)
    
    return units

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

def generateStandardClauses():
    base_clauses = []
    perm_clauses = []
    mandatory_clauses = []

    shape = [5,6,7,8,9,8,7,6,5]

    # Add all rules that make sure that if one cell contains a value it cannot contain any other value (uniqueness)
    for i in range(1,10):
        for j in range(1,shape[i-1]+1):
            base_clauses.append( [int(str(i)+str(j)+str(q)) for q in range(1,10)] ) #Add clauses to make sure each field contains at least one value
            for x in range(1,10):
                for y in range(1,10):
                    if x != y:
                        base_clauses.append([-int(str(i)+str(j)+str(x)),-int(str(i)+str(j)+str(y))])

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
                            perm_clauses.append([-int(str(tup1[0])+str(tup1[1])+str(x)),-int(str(tup2[0])+str(tup2[1])+str(y))])
    
    #Mandatory values rule
    for l in lists:
        length = len(l)
        mandatory_values = [5] #5 is always mandatory
        for i in range(length-5+1):
            if i > 0:
                mandatory_values.extend([5+i,5-i])
                
        for m in mandatory_values:
            clause = [x*100+y*10+m for (x,y) in l]
            mandatory_clauses.append(clause)
    
    clauses = base_clauses+perm_clauses#+mandatory_clauses

    return clauses

def solve_all():
    with open('hanidata\\5200_easy_hanidoku.txt') as f:
        lines = f.readlines()

    standard_clauses = generateStandardClauses()

    for line in lines:
        result = solve(line,standard_clauses)


def solve(input,standard_clauses):
    unit_clauses = getUnitClauses(input)

    all_clauses = standard_clauses + unit_clauses

    t1 = time.time()
    
    sol = ps.solve(all_clauses,verbose=0)

    t2 = time.time() - t1   
    
    print(t2)
    
    if sol == 'UNSAT':
        final_sol = sol
    else:
        final_sol = [s for s in sol if s > 0 ]

#standard_clauses = generateStandardClauses()
#solve('HQV1G0000000000000000045000907004000000000000000000000600000000000',standard_clauses)
#solve_all()

def get_interval_mean(data,i,n,di):
    interval_means = []

    while i < n:
        j = i+di
        interval = [d[1] for d in data if d[0]>i and d[0]<j]
        
        if interval:
            interval_mean = sum(interval)/len(interval)
            interval_means.append((j,len(interval),interval_mean))

        i = j

    return interval_means

def statistics():
    with open('statistics\easy_time_minimal.txt') as f:
        raw_stats = list(map(lambda s: s.strip('\n'), f.readlines()))
        stats = list(map(lambda s: ast.literal_eval(s),raw_stats))

    with open('statistics\easy_dist.txt') as f2:
        raw_dists = list(map(lambda s: s.strip('\n'), f2.readlines()))
        dists = list(map(lambda s: ast.literal_eval(s),raw_dists))
        
    dists_with_stats = list(zip(dists,stats))
    
    sorted_stats = list(sorted(dists_with_stats, key=lambda dist: dist[0][1]))
    
    
    distance_to_conflicts = [(i[0][1],i[1]) for i in sorted_stats]

    interval_mean = get_interval_mean(distance_to_conflicts,1.3,3.2,0.1)

    print(interval_mean)

#statistics()    
    

def plot_and_correlation(data_means):
    interval = []
    hani_in_interval = []
    conflicts = []
    for triple in data_means:
        if(not (triple[0] < 1.5 or triple[0] > 2.8)):
            interval.append(triple[0])
            hani_in_interval.append(triple[1])
            conflicts.append(triple[2])

    print(np.corrcoef(interval,hani_in_interval))
    plt.plot(interval,conflicts)
    plt.ylabel('Number of conflicts')
    plt.xlabel('Upper bound interval distance')
    plt.show()
    

#data_means = statistics()
#plot_and_correlation(data_means)   
    
def distribution_hist(data):
    number_of_clues = [x[0] for x in data]
    distance = [x[1] for x in data]
    
    min_clue = min(number_of_clues)
    max_clue = max(number_of_clues)
    
    min_dist = round(min(distance),1)
    max_dist = round(max(distance),1)
    
    plt.hist(distance,int((max_dist-min_dist)*10), align='left', range = (min_dist,max_dist), facecolor='blue', linewidth=1.2, edgecolor='black', alpha=0.75)
    plt.ylabel('Frequency')
    plt.xlabel('Average distance of clues to 1 or 9')
    plt.xticks(np.arange(min_dist,max_dist,0.2))
    plt.show()
    
    plt.hist(number_of_clues,max_clue-min_clue, align='left', range = (min_clue,max_clue), facecolor='green', linewidth=1.2, edgecolor='black', alpha=0.75)
    plt.ylabel('Frequency')
    plt.xlabel('Number of clues')
    plt.xticks(range(min_clue,max_clue))
    plt.show()
    
with open('statistics\easy_dist.txt') as f:
    raw_dist = list(map(lambda s: s.strip('\n'), f.readlines()))
    dists = list(map(lambda s: ast.literal_eval(s),raw_dist))

#distribution_hist(dists)
    
# Generation of hanidoku
# Hanicue should be the next window (by alt+tab)
# Your text editor the second window
# Example call: gen_n(20,'h') for hard, gen_n(20,'e') for easy generation 
def init():
    alt_tab()
    
    gui.keyDown('alt')
    gui.press('tab')
    gui.press('tab')
    #gui.hotkey('enter')
    gui.keyUp('alt')

def alt_tab():
    gui.hotkey('alt', 'tab')
    #gui.hotkey('enter')

    
def gen_n(n,x):
    init()
    for i in range(n):
        alt_tab()
        time.sleep(0.1)
        gui.hotkey('ctrl',x)
        time.sleep(0.8)
        gui.hotkey('ctrl','c')
        #time.sleep(0.1)

        alt_tab()
        time.sleep(0.15)

        gui.hotkey('ctrl','v')
        gui.hotkey('enter')
        #time.sleep(0.1)

