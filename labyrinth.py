"""
This module defines the Labyrinth class which is used to create, draw, and update a labyrinth. The labyrinth is
represented as a grid of tiles, each of which can have borders on each of its sides. The labyrinth is drawn on a
Tkinter Canvas, and it can be updated in real time based on the information in a JSON file or a Queue.

The Labyrinth class includes methods for creating the labyrinth, drawing it on the canvas, updating the labyrinth
based on the JSON file or the Queue, and handling the turtle's visualization and orientation.

The Queue is used to store the graph structure of the labyrinth. It is checked before the JSON file for updates.
If there's an update in the Queue, it is used to update the labyrinth. If the Queue is empty, the JSON file is checked
for updates.

This module is part of a labyrinth project.

Daniel Zapata Y.
German A Holguin L.
UTP - Pereira, Colombia 2024.
"""

from tiles import Tile
import tkinter as tk
import os
import json
from globales import candado, cola


class Labyrinth:
    """
    The Labyrinth class includes methods for initializing the labyrinth, creating the canvas for the labyrinth,
    generating the board for the labyrinth, calculating the size of the canvas, updating the labyrinth based on
    the graph structure, updating the border of a tile in the labyrinth, getting a specific tile from the list_tiles
    list, and marking the turtle's position and direction on the labyrinth.

    Attributes:
    ----------
    path : str
        The path to the JSON file that contains the labyrinth data.
    list_tiles : list
        List to store the tiles.
    _list_edges : list
        List to store the edges IDs.
    _list_nodes : list
        List to store the nodes IDs.
    rows : int
        The number of rows in the labyrinth.
    columns : int
        The number of columns in the labyrinth.
    tile_array : list
        A 2D array to store the tiles.
    tile_length : int
        The length of each tile in pixels.
    canvas_sz : tuple
        The size of the canvas.
    window : tk.Tk
        The Tkinter window.
    canvas : tk.Canvas
        The Tkinter Canvas object.

    Methods:
    -------
    __init__(self, rows: int, columns: int, path=''):
        Initializes the Labyrinth object with the specified number of rows and columns.
    start(self):
        Start the Tkinter event loop.
    _create_canvas(self):
        Create a canvas for the labyrinth.
    get_board(self):
        Generate the board for the labyrinth.
    _get_canvas_sz(self):
        Calculate the size of the canvas.
    update_maze(self, imprimir=True):
        Update the labyrinth based on the graph structure.
    _check_walls(self, graph: dict):
        Check and update the walls of the labyrinth based on the graph structure.
    _is_diagonal(self, vertex_o: int, vertex_i: int):
        Check if the edge between two vertices is diagonal.
    _update_border(self, vertex_o: int, vertex_i: int, state=False):
        Update the border of a tile in the labyrinth.
    get_tile(self, row, column):
        Get a specific tile from the list_tiles list.
    _mark_turtle(self, turtle_positions: dict):
        Mark the turtle's position and direction on the labyrinth.
    draw_graph(self, graph: dict):
        Draw the graph on the canvas.
    delete_graph(self):
        Delete the graph from the canvas.
    _draw_node(self, center: tuple, radius: int, color: str):
        Draw a node (circle) on the canvas.
    _draw_edge(self, start: tuple, end: tuple):
        Draw an edge (line) on the canvas.
    """

    def __init__(self, rows: int, columns: int, path=''):
        """
        This method initializes the Labyrinth object with the specified number of rows and columns. It also sets up
        the Tkinter window and canvas for drawing the labyrinth, and schedules the update_maze method to be called
        after 10 milliseconds.

        :param rows: (int) The number of rows in the labyrinth.
        :param columns: (int) The number of columns in the labyrinth.
        :param path: (str) The path to the JSON file that contains the labyrinth data. Default is an empty string.
        """
        self.path = path  # Path to the JSON file

        self.list_tiles = list()  # List to store the tiles
        self._list_edges = list()  # List to store the edges IDs
        self._list_nodes = list()  # List to store the nodes IDs

        self.rows, self.columns = rows, columns  # Number of rows and columns in the labyrinth
        # Create a 2D array to store the tiles (not a definitive feature. Could be deleted)
        self.tile_array = [[0 for _ in range(columns)] for _ in range(rows)]
        self.tile_length = 50  # Length of each tile in pixels
        self.tiles_centers = list()  # List to store the center point of each tile
        self._tiles_marks = list()  # List to store the marks IDs of the tiles

        self.canvas_sz = self._get_canvas_sz()  # Size of the canvas
        self.window = tk.Tk()  # Create a new Tkinter window
        self.window.title("Maze")  # Set the title of the window
        self.window.configure(bg='dark gray')

        self._create_canvas()  # Create the canvas for the labyrinth
        self.get_board()  # Generate the board for the labyrinth
        self.window.after(10, self.update_maze)  # Schedule the update_maze method to be called after 10 milliseconds

    def start(self):
        """
        Start the Tkinter event loop.

        This method starts the Tkinter event loop which waits for events and updates the GUI.
        It should be called after all widgets have been created and configured.
        """
        self.window.mainloop()

    def _create_canvas(self):
        """
        Create a canvas for the labyrinth.

        This method creates a new Tkinter Canvas object, sets its width and height to the size of the canvas,
        and then packs it into the window. The canvas is where the labyrinth will be drawn.

        :return: None
        """
        self.canvas = tk.Canvas(self.window, width=self.canvas_sz[0], height=self.canvas_sz[1])
        self.canvas.pack()  # Pack the canvas into the window

    def get_board(self):
        """
        Generate the board for the labyrinth.
        This method creates a list of Tile objects, one for each cell in the labyrinth.
        Each Tile object is drawn on the canvas.

        :return: (list) A list of Tile objects representing the labyrinth.
        """
        for i in range(self.rows):
            for j in range(self.columns):
                # Calculate the position of the tile on the canvas
                tile_pos_x = j * self.tile_length + 20  # Add 20 pixels to the x position for the window border
                tile_pos_y = i * self.tile_length + 20  # Add 20 pixels to the y position for the window border
                tile_mn = Tile(self.canvas, tile_pos_x, tile_pos_y, length=self.tile_length)
                self.tiles_centers.append((tile_pos_x + self.tile_length // 2,
                                           tile_pos_y + self.tile_length // 2))  # Store the top left point of the tile
                self.tile_array[i][j] = tile_mn  # This still is a possible feature (could be deleted)
                self.list_tiles.append(tile_mn)  # Add the tile to the list of tiles
                tile_mn.draw()  # Draw the tile on the canvas

        return self.list_tiles

    def _get_canvas_sz(self):
        """
        Calculate the size of the canvas.

        This method calculates the size of the canvas based on the number of rows
        and columns in the labyrinth and the length of each tile.

        :return: (tuple) A tuple containing the width and height of the canvas.
        """
        height = self.tile_length * self.rows + 40  # Add 40 pixels to the height for the window border
        width = self.tile_length * self.columns + 40  # Add 40 pixels to the width for the window border

        return width, height

    def update_maze(self, imprimir=True):
        """
        Update the labyrinth based on the graph structure.

        This method first checks the Queue for updates. If the Queue is not empty, it retrieves the graph structure
        from the Queue, updates the walls of the labyrinth based on the graph, and marks the turtle's position.

        If the Queue is empty, it checks the JSON file for updates. If the JSON file exists, it reads the graph structure
        from the file, updates the walls of the labyrinth based on the graph, and marks the turtle's position.

        If there are no updates in the Queue or the JSON file, it prints "Nothing to update.".

        This method is scheduled to be called every 20 milliseconds.

        :param imprimir: (bool) A flag used to control the printing of the "Nothing to update." message. Default is True.
        :return: None
        """
        # First check the pipe, if there's nothing there, check the file.
        if not cola.empty():
            with candado:
                graph = cola.get()
            imprimir = True
            if __name__ == '__main__':
                print('The graph structure has been updated from Queue.')
            self._check_walls(graph)

            if graph['show']:
                self.draw_graph(graph)
            else:
                self.delete_graph()

            self._mark_tiles(graph['colors'])
            self._mark_turtle(graph['turtle'])

        else:
            # read json file, if it does not exist, do nothing
            if os.path.exists(self.path):
                with candado:
                    with open(self.path, 'r') as f:
                        graph = json.load(f)
                    f.close()
                    os.remove(self.path)
                imprimir = True
                if __name__ == '__main__':
                    print('The graph structure has been updated from file.')
                self._check_walls(graph)

                if graph['show']:
                    self.draw_graph(graph)
                else:
                    self.delete_graph()

                self._mark_tiles(graph['colors'])
                self._mark_turtle(graph['turtle'])

        if imprimir:
            if __name__ == '__main__':
                print("Nothing to update.")
            imprimir = False

        self.canvas.after(10, self.update_maze, imprimir)

    def _mark_tiles(self, colors: dict):
        """
        Draw a node (circle) on the canvas for each tile in the labyrinth.
        :return: None
        """
        self._delete_marks()
        for node in colors:
            color = colors[node]
            center = self.tiles_centers[int(node)]
            self._tiles_marks.append(self._draw_node(center, self.tile_length // 4, color, outline=''))

    def _delete_marks(self):
        """
        Delete the nodes from the canvas.
        :return: None
        """
        if self._tiles_marks:
            for mark in self._tiles_marks:
                self.canvas.delete(mark)
            self._tiles_marks = list()

    def _check_walls(self, graph: dict):
        """
         Check and update the walls of the labyrinth based on the graph structure.

         This method iterates over the vertices in the graph. For each pair of vertices, it checks if there is an edge
         between them in the graph. If there is an edge and its value is 0, it means there is a wall between the
         vertices in the labyrinth, so it calls the _update_border method to update the border of the tile at the
         position of the first vertex to exist. If the value of the edge is not 0, it means there is no wall between the
         vertices in the labyrinth, so it calls the _update_border method to update the border of the tile at the
         position of the first vertex to not exist.

         :param graph: (dict) The graph structure of the labyrinth. It is a dictionary with two keys: 'V' and 'E'.
                       'V' maps to a dictionary where each key is a vertex and the value is a list of vertices adjacent to the key.
                       'E' maps to a dictionary where each key is a tuple of two vertices and the value is the weight of the edge
                       between the vertices.
         :return: None
         """
        vertex_list = graph['V']
        edges_list = graph['E']
        # vertex_o is the origin vertex, vertex_i is the destination vertex
        for vertex_o in vertex_list:
            for vertex_i in vertex_list[vertex_o]:
                if edges_list.get(f"({vertex_o}, {vertex_i})") == 0 or edges_list.get(f"({vertex_i}, {vertex_o})") == 0:
                    # print(f"There is a wall in edge:     ({vertex_o}, {vertex_i})")
                    self._update_border(int(vertex_o), int(vertex_i), state=True)
                else:
                    # print(f"There is not a wall in edge: ({vertex_o}, {vertex_i})")
                    if not self._is_diagonal(int(vertex_o), int(vertex_i)) and not self._is_far(int(vertex_o),
                                                                                                int(vertex_i)):
                        self._update_border(int(vertex_o), int(vertex_i))

    def _is_far(self, vertex_o: int, vertex_i: int):
        """
        Check if the edge between two vertices is far.

        This method calculates the row and column positions of the vertices, and then checks if the vertices are
        far from each other. If they are, it returns True. Otherwise, it returns False.

        :param vertex_o: (int) The origin vertex.
        :param vertex_i: (int) The destination vertex.
        :return: (bool) True if the edge is far, False otherwise.
        """
        row_o, col_o = divmod(vertex_o, self.columns)
        row_i, col_i = divmod(vertex_i, self.columns)

        return abs(row_o - row_i) > 1 or abs(col_o - col_i) > 1

    def _is_diagonal(self, vertex_o: int, vertex_i: int):
        """
        Check if the edge between two vertices is diagonal.

        This method calculates the row and column positions of the vertices, and then checks if the vertices are
        diagonally adjacent to each other. If they are, it returns True. Otherwise, it returns False.

        :param vertex_o: (int) The origin vertex.
        :param vertex_i: (int) The destination vertex.
        :return: (bool) True if the edge is diagonal, False otherwise.
        """
        row_o, col_o = divmod(vertex_o, self.columns)
        row_i, col_i = divmod(vertex_i, self.columns)

        return abs(row_o - row_i) >= 1 and abs(col_o - col_i) >= 1

    def _update_border(self, vertex_o: int, vertex_i: int, state=False):
        """
        Update the border of a tile in the labyrinth.

        This method calculates the row and column positions of two vertices, determines which border of the tile
        at the position of the first vertex needs to be updated based on the relative positions of the vertices,
        and then updates the border's state.

        :param vertex_o: (int) The origin vertex.
        :param vertex_i: (int) The destination vertex.
        :param state: (bool) The state to set the border to. If True, the border is set to exist.
                      If False, the border is set to not exist.
        :return: None
        """
        # Calculate the row and column positions of the vertices
        row_o, col_o = divmod(vertex_o, self.columns)
        row_i, col_i = divmod(vertex_i, self.columns)

        # Determine the border to be deleted
        if row_o == row_i:  # The vertices are in the same row
            border_id = 3 if col_o < col_i else 2  # Delete left border if vertex_o < vertex_i, else delete right border
            # print(f"row_o: {row_o}, col_o: {col_o}, row_i: {row_i}, col_i: {col_i}, border_id: {border_id}")
        else:  # The vertices are in the same column
            border_id = 1 if row_o < row_i else 0  # Delete upper border if vertex_o < vertex_i, else delete bottom
            # border
        # Get the tile and delete the border
        tile = self.get_tile(row_o, col_o)
        tile.update_border_visualization(border_id, state=state)

    def get_tile(self, row, column):
        """
        Get a specific tile from the list_tiles list.
        :param row: (int) The row position of the tile.
        :param column: (int) The column position of the tile.
        :return: (Tile) The Tile object at the specified position.
        """
        index = row * self.columns + column
        return self.list_tiles[index]

    def _mark_turtle(self, turtle_positions: dict):
        """
         Mark the turtle's position and direction on the labyrinth.

         This method first erases the turtle from all tiles in the labyrinth. Then it iterates over the turtle_positions
         dictionary. For each pair of vertices, it calculates the row and column positions of the vertices, gets the tile
         at the position of the first vertex, and draws the turtle on the tile.

         If the second vertex is 'f', it means the turtle is facing up. Otherwise, it determines the direction of the turtle
         based on the relative positions of the vertices and rotates the turtle to the determined direction.

        :param turtle_positions: (dict) A dictionary where each key is a vertex and the value is the vertex that the turtle
                                  is facing towards. If the value is 'f', it means the turtle is in the last node
                                  and facing up.

         :return: None
         """
        for tile in self.list_tiles:
            tile.change_turtle_state(erase=True)

        for vertex_o, vertex_i in turtle_positions.items():
            if __name__ == '__main__':
                print(f"Path: {vertex_o} -> {vertex_i}")
            # Calculate the row and column positions of the vertices
            if vertex_i == 'f':
                row_o, col_o = divmod(int(vertex_o), self.columns)
                tile = self.get_tile(row_o, col_o)
                tile.rotate_turtle(direction='u')  # Rotate the turtle to the up direction
                # Draw the turtle on the tile
                tile.change_turtle_state(erase=False)

            else:
                row_o, col_o = divmod(int(vertex_o), self.columns)
                row_i, col_i = divmod(int(vertex_i), self.columns)
                # Get the tile
                tile = self.get_tile(row_o, col_o)
                # Determine the direction of the turtle
                if row_o == row_i:  # The vertices are in the same row
                    direction = 'r' if col_o < col_i else 'l'  # Move right if vertex_o < vertex_i, else move left
                else:  # The vertices are in the same column
                    direction = 'd' if row_o < row_i else 'u'  # Move down if vertex_o < vertex_i, else move up
                # Rotate the turtle to the determined direction
                tile.rotate_turtle(direction)
                # Draw the turtle on the tile
                tile.change_turtle_state(erase=False)

    def draw_graph(self, graph: dict):
        """
        Draw the graph on the canvas.

        This method first checks if the graph exists. If it does, it deletes the previous graph drawn on the canvas.
        Then it calculates the radius of the nodes to be drawn.

        It then iterates over the edges in the graph. For each edge that exists (value is not 0), it splits the edge into
        its origin and destination vertices, calculates the center points of these vertices, and draws the edge on the canvas.

        It then checks if the origin and destination vertices have a specified color in the graph. If they do, it uses that
        color to draw the nodes. If they don't, it uses the default color 'coral' to draw the nodes.

        :return: None
        """

        # Delete the previous graph drawn on the canvas
        self.delete_graph()
        bg_color = 'lightblue'
        radius = self.tile_length // 2 - self.tile_length // 7  # Calculate the radius of the nodes
        # Iterate over the edges in the graph
        for edge in graph['E']:
            # Check if the edge exists
            vertex_o, vertex_i = edge[1:-1].split(', ')  # Split the edge into origin and destination vertices
            vertex_o, vertex_i = int(vertex_o), int(vertex_i)
            center_o = self.tiles_centers[vertex_o]
            center_i = self.tiles_centers[vertex_i]

            # If there is a path between the nodes, draw the edge
            # if graph['E'][edge] != 0:
            # Draw the edge on the canvas
            self._list_edges.append(self._draw_edge(center_o, center_i))

        for node in graph['V']:
            center = self.tiles_centers[int(node)]
            self._list_nodes.append(self._draw_node(center, radius, color=bg_color))

    def delete_graph(self):
        """
        Delete the graph from the canvas.

        This method iterates over the list of edges and nodes in the graph. For each edge and node,
        it deletes it from the canvas.
        :return: None
        """
        if self._list_edges:
            for edge in self._list_edges:
                self.canvas.delete(edge)
            self._list_edges = list()  # Reset the list of edges

        if self._list_nodes:
            for node in self._list_nodes:
                self.canvas.delete(node)
            self._list_nodes = list()  # Reset the list of nodes

    def _draw_node(self, center: tuple, radius: int, color: str = '', outline: str = 'black'):
        """
        Draw a node (circle) on the canvas.

        This method creates a circle on the canvas at the specified center point with the specified radius and color.
        The circle is filled with the specified color.

        :param center: (tuple) A tuple containing the x and y coordinates of the center of the circle.
        :param radius: (int) The radius of the circle.
        :param color: (str) The color to fill the circle with. Default is an empty string. If color is an empty string,
                        the node is not filled.
        :param outline: (str) The color of the outline of the circle. Default is 'black'. If outline is an empty string,
                        the circle has no outline.
        :return: (int) The ID of the created circle.
        """
        circle = self.canvas.create_oval(center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius,
                                         fill=color, outline=outline)
        return circle

    def _draw_edge(self, start: tuple, end: tuple):
        """
        Draw an edge (line) on the canvas.

        This method creates a line on the canvas from the start point to the end point with the specified color.

        :param start: (tuple) A tuple containing the x and y coordinates of the start point of the line.
        :param end: (tuple) A tuple containing the x and y coordinates of the end point of the line.
        :return: (int) The ID of the created line.
        """
        line = self.canvas.create_line(start[0], start[1], end[0], end[1])
        return line


if __name__ == '__main__':
    maze = Labyrinth(4, 4, path='/dev/shm/graph.json')  # linux
    # maze = Labyrinth(2, 3, path=r'C:\Users\German Andres\Desktop\grafo.json')  # windows
    maze.start()
