
# Degree greedy-based algorithm
def dgba(case, start_node, length_H):
    V_H = [start_node]
    neighbors = []
    E_H = []
    while len(V_H) < length_H:
        node = V_H[-1]
        for edge in case['branch']:
            if edge[0] == node and edge[1] not in V_H:
                neighbors.append(edge[1])
            if edge[1] == node and edge[0] not in V_H:
                neighbors.append(edge[0])
        neighbors = list(set(neighbors))
        degrees = []
        for neighbor in neighbors:
            degree = 0
            for edge in case['branch']:
                if (edge[0] == neighbor and edge[1] not in V_H) or (edge[1] == neighbor and edge[0] not in V_H):
                    degree += 1
            degrees.append(degree)

        if degrees:
            max_degree = max(degrees)
            max_index = degrees.index(max_degree)
            if max_degree > 0:
                V_H.append(neighbors[max_index])
                neighbors.pop(max_index)
                degrees.pop(max_index)
            else:
                break
        else:
            break
    for i in range(case['branch'].shape[0]):
        if (case['branch'][i, 0] in V_H) and (case['branch'][i, 1] in V_H):
            E_H.append(i)

    return V_H, E_H