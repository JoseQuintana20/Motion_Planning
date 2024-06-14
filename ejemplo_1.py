"""
This module is part of a labyrinth project. It uses the Grafo class to create a graph representation of a labyrinth.

The graph is created by adding vertices (rooms in the labyrinth) and edges (paths between the rooms). The weight of an
edge determines whether there is a path between two rooms (weight 1) or a wall (weight 0).

After creating the graph, it is printed to the terminal and saved as a JSON file. The JSON file can be used by other
parts of the project, such as the GUI.

This module is intended to be run as a standalone script.
"""

from grafo import Grafo

if __name__ == '__main__':
    grafo = Grafo()
    # Add vertices to the graph
    grafo.add_edge(0, 1, 1)
    grafo.add_edge(0, 3, 0)
    grafo.add_edge(1, 2, 0)
    grafo.add_edge(1, 4, 1)
    grafo.add_edge(2, 5, 0)
    grafo.add_edge(3, 4, 1)
    grafo.add_edge(4, 5, 1)

    # show graph on terminal
    print(grafo)

    # save graph to the comm file. This file is read by tue GUI.
    grafo.save_graph('/dev/shm/graph.json')  # For operation in Linux
    # grafo.save_graph(r'C:\Users\German Andres\Desktop\grafo.json') # For operation in Windows
