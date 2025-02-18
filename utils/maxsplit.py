import numpy as np
import networkx as nx

def adjacency_list(adj_matrix):
    """
    Input:
        adj_matrix (list of list of int): A square matrix representing the graph, where each element is either 0 or 1.
            An element of 1 at position (i, j) indicates the presence of an edge from vertex i to vertex j.
    Output:
        dict: A dictionary representing the graph in adjacency list format. Each key is a vertex index, and its value
            is a list of indices corresponding to its neighboring vertices.
    Function:
        This function converts an adjacency matrix representation of a graph into an adjacency list representation.
        It iterates over each vertex (row) of the matrix and collects the indices of vertices (columns) for which 
        the matrix entry is 1, thereby indicating an edge between the vertices.
    """
    adj_list = {}
    n = len(adj_matrix)
    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                neighbors.append(j)
        adj_list[i] = neighbors
    return adj_list

def maxsplit(vertex: list, edge: dict):
    """
    Input:
        vertex (list): A list of vertices in the graph. Each vertex is assumed to be a hashable identifier.
        edge (dict): A dictionary mapping each vertex to a list of its adjacent vertices, representing the graph's
            edge connections.
    Output:
        list of list: A list where each element is a sublist (a maximal independent set) of vertices.
            A maximal independent set is defined as a set of vertices where no two vertices are adjacent,
            and no additional vertex can be included without violating this property.
    Function:
        This function employs a greedy algorithm to partition the vertices of a graph into maximal independent sets.
        Initially, it marks all vertices as unvisited and computes the degree (number of adjacent vertices) for 
        each vertex. It then sorts the vertices (based on their keys as obtained from the edge dictionary) and 
        iteratively selects the first unvisited vertex from the sorted order. Once a vertex is selected, it forms 
        an independent set by adding all unvisited vertices that are not adjacent to any vertex already in the set. 
        Each time a vertex is added to a set, it is marked as visited and its neighbors are recorded to prevent 
        conflicts. This process repeats until all vertices are assigned to a set. Such a partitioning is useful 
        in problems like graph coloring and network decomposition.
    """
    split = []
    visit = [0 for _ in range(len(vertex))]
    visitmp = {}
    for i in range(len(vertex)):
        visitmp[vertex[i]] = i
    degree = dict()
    for v in edge:
        degree[v] = len(edge[v])
    degree = sorted(degree.items())
    while sum(visit) < len(vertex):
        num = 0
        nowsplit = []
        # Select the first unvisited vertex from the sorted degree list.
        while num < len(degree):
            if not visit[visitmp[degree[num][0]]]:
                nowsplit.append(degree[num][0])
                visit[visitmp[degree[num][0]]] = 1
                break
            else:
                num += 1
        # Initialize the set of neighbors for the chosen vertex.
        neibor = set()
        for u in edge[degree[num][0]]:
            neibor.add(u)
        # Add other unvisited vertices that are not neighbors to the current independent set.
        for v in vertex:
            if not visit[visitmp[v]] and v not in neibor:
                nowsplit.append(v)
                visit[visitmp[v]] = 1
                for u in edge[v]:
                    neibor.add(u)
        split.append(nowsplit)
    return split
