import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class KNP:
    def __init__(self, n: int):
        self.n = n
        self.graph_matrix = self.fill_graph_matrix()

    def fill_graph_matrix(self):
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.choice([True, False]):
                    matrix[i][j] = np.random.randint(0, 100)
                    matrix[j][i] = matrix[i][j]
        return matrix

    def draw_graph(self, matrix=None):
        matrix = self.graph_matrix if matrix is None else matrix
        graph = nx.from_numpy_matrix(matrix)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.show()

    def get_nearest_node(self, matrix, visited_nodes):
        result = [-1, -1]
        min_d = 9999999999999
        for node in visited_nodes:
            for i in range(len(matrix[node])):
                if matrix[node][i] == 0:
                    continue
                if matrix[node][i] < min_d:
                    min_d = matrix[node][i]
                    result = [node, i]

        for node in visited_nodes:
            matrix[node][result[1]] = matrix[result[1]][node] = 0
        return result, min_d

    def find_minimal_tree(self):
        min_tree = nx.Graph()
        visited_nodes = [0]
        unvisited_nodes = np.arange(start=0, stop=10, step=1).tolist()

        while len(unvisited_nodes) != 0:
            nearest_node, min_d = self.get_nearest_node(self.graph_matrix, visited_nodes)
            if nearest_node[0] == -1:
                break
            new_node = nearest_node[1]
            visited_nodes.append(new_node)
            unvisited_nodes.remove(new_node)

            min_tree.add_edge(*nearest_node, weight=min_d)
        return nx.to_numpy_array(min_tree)

    def clustering(self, matrix):
        tree = nx.from_numpy_array(matrix)
        max_node = [-1, -1]
        max_d = -1

        nodes_list = list(tree.edges.data())
        for i in nodes_list:
            if i[2]["weight"] > max_d:
                max_d = i[2]["weight"]
                max_node = [i[0], i[1]]
        tree.remove_edge(*max_node)
        return nx.to_numpy_array(tree)


if __name__ == '__main__':
    knp = KNP(10)
    knp.draw_graph()

    minimal_tree = knp.find_minimal_tree()
    knp.draw_graph(minimal_tree)

    knp.draw_graph(knp.clustering(minimal_tree))
