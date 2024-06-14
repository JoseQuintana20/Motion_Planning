"""
This module defines the Grafo class which represents a graph, specifically a labyrinth.

The Grafo class includes methods for initializing the graph, representing the graph as a string, getting the graph,
sending the graph to a Queue, saving the graph as a JSON file, and adding an edge to the graph.

The graph is represented as a dictionary with three keys: 'V', 'E', and 'turtle'. 'V' maps to a dictionary where each
key is a vertex and the value is a list of vertices adjacent to the key. 'E' maps to a dictionary where each key is a
tuple of two vertices and the value is the weight of the edge between the vertices. 'turtle' maps to a dictionary where
each key is a vertex and the value is the vertex that the turtle is facing towards. If the value is 'f', it means the
turtle is in the last node.

The main function of this module creates a graph with a specific adjacency list of vertices and weighted edges, and a
specific list of vertices to show a turtle and the turtle's goal. It then saves the graph as a JSON file.

This module uses the 'json' module for saving the graph as a JSON file and the 'cola' module from the 'globales' package
for sending the graph to a Queue.

Daniel Zapata Y.
German A Holguin L.
UTP - Pereira, Colombia 2024.
"""

import json
from globales import cola, candado
from copy import deepcopy


class Grafo:
    """
    A class to represent a graph, specifically a labyrinth.

    The Grafo class includes methods for initializing the graph, representing the graph as a string, getting the graph,
    sending the graph to a Queue, saving the graph as a JSON file, and adding an edge to the graph.

    Attributes:
    ----------
    V : dict
        The vertices of the graph. Each key is a vertex and the value is a list of vertices adjacent to the key.
    E : dict
        The edges of the graph. Each key is a tuple of two vertices and the value is the weight of the edge between the vertices.
    turtle : dict
        The turtle's position and direction. Each key is a vertex and the value is the vertex that the turtle is facing towards.
        If the value is 'f', it means the turtle is in the last node and facing up.
    colors : dict
        The colors of the vertices. Each key is a vertex and the value is the color of the vertex.
    _show_graph : bool
        A flag to control the visibility of the graph in the GUI. If True, the graph will be displayed in the GUI; if False,
        the graph will not be displayed.
    _cluster_graphs : bool
        A flag to control whether only the colored vertices should be shown in the GUI. If True, only the colored vertices
        will be displayed; if False, all vertices will be displayed.

    Methods:
    -------
    __init__(self, V: dict = None, E: dict = None, turtle: dict = None):
        Initializes the graph with the given vertices, edges, and the turtle's position.
    __repr__(self):
        Returns the graph as a string.
    get_graph(self):
        Returns the graph as a dictionary.
    send_graph(self):
        Gets the graph and puts it into the global queue 'cola'.
    save_graph(self, path: str):
        Gets the graph and saves it as a JSON file at the specified path.
    add_edge(self, vertex_o: int, vertex_i: int, weight: int):
        Adds an edge between two vertices in the graph.
    show(self):
        Sets the _show_graph attribute to True. This attribute is used to show the graph in the GUI.
    """

    def __init__(self, V: dict = None, E: dict = None, turtle: dict = None, colors: dict = None):
        """
        Initialize the graph with the vertices, edges, and the turtle's position.

        This method initializes the graph with the given vertices, edges, and the turtle's position. If any of these
        parameters are not provided, it initializes them as empty dictionaries.

        :param V: (dict) The vertices of the graph. Each key is a vertex and the value is a list of vertices adjacent to
                    the key. Default is an empty dictionary.
        :param E: (dict) The edges of the graph. Each key is a tuple of two vertices and the value is the weight of the
                    edge between the vertices. Default is an empty dictionary.
        :param turtle: (dict) The turtle's position and direction. Each key is a vertex and the value is the vertex that
                    the turtle is facing towards. If the value is 'f', it means the turtle is in the last node and
                    facing up. Default is an empty dictionary.
        :param colors: (dict) The colors of the vertices. Each key is a vertex and the value is the color of the vertex.
                    Default is an empty dictionary.
        :return: None
        """
        if V is None:
            V = dict()
        self.V = V
        if E is None:
            E = dict()
        self.E = E
        if turtle is None:
            turtle = dict()
        self.turtle = turtle
        if colors is None:
            colors = dict()
        self.colors = colors
        self._show_graph = False

    def __repr__(self):
        """
        Return the graph as a string
        """
        return f'Vertices: {self.V}\nEdges: {self.E}\nTurtle: {self.turtle}\nColors: {self.colors}'

    def get_graph(self):
        """
        Get the graph.

        This method returns the graph as a dictionary. The dictionary includes the vertices, edges, and the turtle's
        position and direction in the graph.

        :return: (dict) The graph. It includes 'V' followed by the vertices of the graph, 'E' followed by the edges of the
                 graph, and 'turtle' followed by the turtle's position and direction.
        """
        grafo_g = {'V': self.V, 'E': self.E, 'turtle': self.turtle, 'colors': self.colors,
                   'show': self._show_graph}
        return grafo_g

    def send_graph(self):
        """
        Send the graph to the Queue.

        This method gets the graph and puts it into the global queue 'cola'. This can be used to share the graph between
        different parts of the program or with different threads.

        :return: None
        """
        grafo_g = self.get_graph()
        with candado:
            # Put the graph into the global queue 'cola'
            # The deepcopy function is used to avoid
            # the Queue storing a reference to the graph to the original graph
            cola.put(deepcopy(grafo_g))

    def save_graph(self, path: str):
        """
        Save the graph as a JSON file.

        This method gets the graph and saves it as a JSON file at the specified path. The JSON file is indented by 4 spaces
        for readability. After writing the JSON file, the file is closed.

        :param path: (str) The path where the JSON file will be saved.
        :return: None
        """
        grafo_g = self.get_graph()
        with open(path, 'w') as file_graph:
            json.dump(grafo_g, file_graph, indent=4)
        # Close the file_graph
        file_graph.close()

    def delete_edge(self, vertex_o: int, vertex_i: int):
        """
        This method deletes an edge between two vertices in the graph. If the edge does not exist, it prints a message and
        does not delete the edge. If the vertices do not exist in the graph, it prints a message and does not delete the edge.

        :param vertex_o: (int) The origin vertex of the edge.
        :param vertex_i: (int) The destination vertex of the edge.
        :return: None
        """
        # Verify if the edge exists
        logical_a = f"({vertex_o}, {vertex_i})" in self.E
        logical_b = f"({vertex_i}, {vertex_o})" in self.E
        if logical_a or logical_b:
            # Delete the edge from the graph
            if logical_a:
                del self.E[f"({vertex_o}, {vertex_i})"]
            else:
                del self.E[f"({vertex_i}, {vertex_o})"]

            # Delete the edge from the vertices
            self.V[vertex_o].remove(vertex_i)
            self.V[vertex_i].remove(vertex_o)
        else:
            if __name__ == '__main__':
                print(f"The edge ({vertex_o}, {vertex_i}) does not exist.")

    def add_edge(self, vertex_o: int, vertex_i: int, weight: int):
        """
        This method adds an edge between two vertices in the graph. If the edge already exists, it prints a message and
        does not add the edge. If the vertices do not exist in the graph, it adds them. The edge is represented as a
        string of the form '(vertex_o, vertex_i)' and the weight of the edge is an integer.

        :param vertex_o: (int) The origin vertex of the edge.
        :param vertex_i: (int) The destination vertex of the edge.
        :param weight: (int) The weight of the edge. If the weight is 0, there is no path between the nodes
                       (a wall exists), if the weight is 1, there is a path between the nodes (a wall does not exist).
        :return: None
        """
        # Verify if the edge already exists
        if f"({vertex_o}, {vertex_i})" in self.E or f"({vertex_i}, {vertex_o})" in self.E:
            # If exists but with different weight, update the weight
            if f"({vertex_o}, {vertex_i})" in self.E:
                if self.E[f"({vertex_o}, {vertex_i})"] != weight:
                    self.E[f"({vertex_o}, {vertex_i})"] = weight

            elif f"({vertex_i}, {vertex_o})" in self.E:
                if self.E[f"({vertex_i}, {vertex_o})"] != weight:
                    self.E[f"({vertex_i}, {vertex_o})"] = weight

            if __name__ == '__main__':
                print(f"The edge ({vertex_o}, {vertex_i}) already exists.")
        else:
            # Verify if vertices exist or add them if they are not in the graph
            if vertex_o not in self.V:
                self.V[vertex_o] = [vertex_i]
            else:
                self.V[vertex_o].append(vertex_i)
            if vertex_i not in self.V:
                self.V[vertex_i] = [vertex_o]
            else:
                self.V[vertex_i].append(vertex_o)
            # Add the edge to the graph
            self.E[f"({vertex_o}, {vertex_i})"] = weight

    def show(self, show: bool = True):
        """
        This method is used to control the visibility of the graph in the GUI. It sets the _show_graph attribute to the
        value passed in the 'show' parameter. If 'show' is True, the graph will be displayed in the GUI; if 'show' is False,
        the graph will not be displayed.

        :param show: (bool) A flag to control the visibility of the graph in the GUI. Default is True.
        :return: None
        """
        self._show_graph = show


if __name__ == '__main__':
    # Create a dictionary with the adjacency list of vertices of the graph
    vertex_list = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
    # Create a dictionary with the adjacency list of weighted edges of the graph
    # If edge weight is 0, there is no path between the nodes (a wall exists),
    # if edge weight is 1, there is a path between the nodes (a wall does not exist)
    edges_list = {'(0, 1)': 1, '(0, 3)': 0, '(1, 2)': 0, '(1, 4)': 1, '(2, 5)': 0, '(3, 4)': 1, '(4, 5)': 1}
    # List of all vertices to show a turtle (the key) and the turtle's goal (the value)
    turtle_list = {}  # {0: 1, 1: 4, 4: 3, 5: 'f'}  # 'f' is a centinel value to represent turtle's exit
    # Create a dictionary with the colors of the vertices
    colors_list = {"0": 'red', "1": 'blue', "2": 'green', "4": 'black', "5": 'white'}
    # Create a graph
    grafo = Grafo(vertex_list, edges_list, turtle_list, colors_list)
    grafo.show(False)

    # Save the graph as a json file
    grafo.save_graph('/dev/shm/graph.json')
    print(grafo)
