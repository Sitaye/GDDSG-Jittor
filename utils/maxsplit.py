import numpy as np
import networkx as nx

def adjacency_list(adj_matrix):
    adj_list = {}
    n = len(adj_matrix)
    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                neighbors.append(j)
        adj_list[i] = neighbors
    return adj_list

def maxsplit(vertex:list,edge:dict):
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
        while num < len(degree):
            if not visit[visitmp[degree[num][0]]]:
                nowsplit.append(degree[num][0])
                visit[visitmp[degree[num][0]]] = 1
                break
            else:
                num += 1
        neibor = set()
        for u in edge[degree[num][0]]:
            neibor.add(u)
        for v in vertex:
            if not visit[visitmp[v]] and v not in neibor:
                nowsplit.append(v)
                visit[visitmp[v]] = 1
                for u in edge[v]:
                    neibor.add(u)
        split.append(nowsplit)
    return split


if __name__ == '__main__':
    adj_matrix = np.load(r"C:\Users\laigu\Desktop\ITCIL\array_dogs.npy")
    print(adj_matrix)
    for i in range(120):
        if adj_matrix[47][i] == 1:
            print(i,end = " ")
    adj_list = adjacency_list(adj_matrix)
    print(len(maxsplit([i for i in range(len(adj_matrix))],adj_list)),maxsplit([i for i in range(len(adj_matrix))],adj_list))
    def dfs(node, visited):
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, visited)

    def count_connected_components(adj_matrix):
        num_nodes = len(adj_matrix)
        visited = [False] * num_nodes
        num_components = 0

        for i in range(num_nodes):
            if not visited[i]:
                dfs(i, visited)
                num_components += 1

        return num_components

    num_connected_components = count_connected_components(adj_matrix)
    print(f"该图有 {num_connected_components} 个联通分量。")