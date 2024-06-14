"""
This module defines the Tile class which is used as the basic unit in a labyrinth. Each Tile represents a square in the
labyrinth and has properties such as its position, size, and whether it has borders on each of its sides.

The Tile class also includes methods for drawing the tile on a tkinter Canvas, including its borders and a turtle image
that can be used to represent a player's position. The turtle's orientation can be changed, and it can be drawn or
erased from the tile.

The module includes a main section that creates a tkinter window and canvas, and draws a single tile on the canvas.

This module is part of a labyrinth project. The project is implemented in Python using the tkinter library for the GUI.

Daniel Zapata Y.
German A Holguin L.
UTP - Pereira, Colombia 2024
"""

import tkinter as tk


class Tile:
    """
    The Tile class includes methods for initializing the tile, drawing the tile and its borders, updating the visualization
    of a border, calculating the coordinates for borders, rotating the turtle, drawing the turtle, and changing the state
    of the turtle on the tile.

    Attributes:
    ----------
    border_width : int
        The width of the borders of the tile.
    canvas : tk.Canvas
        The tkinter Canvas on which the tile will be drawn.
    position : tuple
        The position of the top-left corner of the tile.
    _length : int
        The length of the sides of the tile.
    borders : list
        A list of booleans representing whether each border exists.
    borders_ID : list
        A list to store the ID of the borders on the canvas.
    turtle_ID : int
        The ID of the turtle on the canvas.
    turtle : bool
        If True, draw a turtle in the current tile to simulate the player move.
    turtle_orientation : str
        The orientation of the turtle: 'r' for right, 'l' for left, 'u' for up, 'd' for down.
    turtle_image : tk.PhotoImage
        A PhotoImage object created from the turtle image file.
    bg_ID : int
        The ID of the background on the canvas.

    Methods:
    -------
    __init__(self, sketch: tk.Canvas, pos_x=0, pos_y=0, length=100, width=2):
        Initializes a Tile object.
    _get_turtle_image(self):
        Selects an appropriate image size for the turtle based on the size of the canvas.
    draw(self, bg='lightblue', turtle=False):
        Draws the tile on the canvas.
    _draw_border(self, border_id: int):
        Draws a border on the tile based on the given border_id.
    update_border_visualization(self, border_id: int, state: bool):
        Updates the visualization of a border in the canvas.
    _get_line_coords(self, border_id: int):
        Calculates the coordinates for borders of a given tile.
    rotate_turtle(self, direction: str):
        Rotates the turtle based on the given direction.
    _draw_turtle(self):
        Draws a turtle on the tile.
    change_turtle_state(self, erase=True):
        Changes the state of the turtle on the tile.
    """
    def __init__(self, sketch: tk.Canvas, pos_x=0, pos_y=0, length=100, width=2):
        """
        Initializes a Tile object.

        :param sketch: (tk.Canvas) The tkinter Canvas on which the tile will be drawn.
        :param pos_x: (int) The x-coordinate of the top-left corner of the tile. Default is 0.
        :param pos_y: (int) The y-coordinate of the top-left corner of the tile. Default is 0.
        :param length: (int) The length of the sides of the tile. Default is 100.
        :param width: (int) The width of the borders of the tile. Default is 2.
        """
        self.border_width = width  # Width of the borders
        self.canvas = sketch  # Canvas where the tile will be drawn
        self.position = (pos_x, pos_y)  # Position of the top-left corner of the tile
        self._length = length  # Length of the sides of the tile
        # Represent borders as: [Top, bottom, left, right]
        self.borders = [True, True, True, True]
        self.borders_ID = [None, None, None, None]  # Store the ID of the borders on the canvas
        self.turtle_ID = None  # Store the ID of the turtle on the canvas
        self.turtle = False  # If True, draw a turtle in the current tile to simulate de player move
        self.turtle_orientation = 'u'  # Define turtle orientation: r: right, l: left, u: up, d: down
        self.turtle_image = self._get_turtle_image()  # Image as instance attribute to avoid python garbage collection
        self.bg_ID = None  # Store the ID of the background on the canvas

    def _get_turtle_image(self):
        """
        This method selects an appropriate image size for the turtle based on the size of the canvas.
        It constructs the path to the turtle image file and returns a PhotoImage object created from that file.

        The method checks the length of the tile and based on its size, it selects the appropriate turtle image size.
        The turtle image size can be "100", "75", or "50", corresponding to a tile length greater than 200,
        greater than 100, or less than or equal to 100, respectively.

        The path to the turtle image file is constructed using the selected turtle image size and the current
        orientation of the turtle. The path is then used to create a PhotoImage object, which is returned by the method.

        :return tk.PhotoImage: A PhotoImage object created from the turtle image file.
        """
        # Select an appropriate image size for the turtle based on the size of the tile
        if self._length > 200:
            turtle_img_size = "100"
        elif self._length > 100:
            turtle_img_size = "75"
        elif self._length > 75:
            turtle_img_size = "50"
        else:
            turtle_img_size = "25"
        # Construct the path to the turtle image file
        turtle_img_path = f"resources/{turtle_img_size}/turtle_{turtle_img_size}px_{self.turtle_orientation}.png"
        return tk.PhotoImage(file=turtle_img_path)

    def draw(self, bg='lightblue', turtle=False):
        """
        This method draws the tile on the canvas. It first draws the background color for the tile,
        then draws the turtle if it exists, and finally draws the borders of the tile.

        :param bg: (str) The background color of the tile. Default is 'lightblue'.
        :param turtle: (bool) If True, a turtle is drawn on the tile. Default is False.
        """

        # Draw the background color for the tile
        # The create_rectangle method of the canvas is used to draw the background of the tile.
        # The position of the tile and its length are used to determine the coordinates of the rectangle.
        # The fill parameter is used to set the color of the rectangle, and the outline parameter is set to '' to
        # remove the outline.
        # Draw the background color for the tile
        self.bg_ID = self.canvas.create_rectangle(
            self.position[0], self.position[1], self.position[0] + self._length, self.position[1] + self._length,
            fill=bg,
            outline='')
        # Draw the turtle over the tile background
        self.turtle = turtle
        if self.turtle:
            self._draw_turtle()

        # Draw the borders of the tile
        # The borders are drawn by iterating over the borders attribute, which is a list of booleans representing
        # whether each border exists.
        # For each border, the _get_line_coords method is called to get the coordinates of the border line.
        # If the border exists (the corresponding element in the borders list is True), the create_line method of the
        # canvas is used to draw the border.
        # If the border does not exist (the corresponding element in the borders list is False), the create_line
        # method is used to draw a white line, effectively erasing the border.
        for k in range(len(self.borders)):
            # Line coords get a 4 element tuple: (x_i, y_i, x_f, y_f)
            self._draw_border(k)

    def _draw_border(self, border_id: int):
        """
        This method draws a border on the tile based on the given border_id.

        The method first gets the coordinates for the border line using the _get_line_coords method.
        If the border exists (the corresponding element in the borders list is True), the create_line method of the
        canvas is used to draw the border.
        If the border does not exist (the corresponding element in the borders list is False), the create_line
        method is used to draw a line with the same color as the background, effectively erasing the border.

        :param border_id: (int) The id of the border to be drawn. The id corresponds to the following borders:
                          0 - Top border
                          1 - Bottom border
                          2 - Left border
                          3 - Right border
        """
        # Get coordinates for the current border
        line_coords = self._get_line_coords(border_id)

        # If the border exists, draw the border
        if self.borders[border_id]:
            self.borders_ID[border_id] = self.canvas.create_line(line_coords[0], line_coords[1], line_coords[2],
                                                                 line_coords[3],
                                                                 width=self.border_width)
        # If the border does not exist, draw a line with similar background color to erase the border
        else:
            self.borders_ID[border_id] = self.canvas.create_line(line_coords[0], line_coords[1], line_coords[2],
                                                                 line_coords[3], fill='lightblue1',
                                                                 width=self.border_width)

    def update_border_visualization(self, border_id: int, state: bool):
        """
        This method updates the visualization of a border in the canvas.
        :param border_id: (int): The id of the border to be updated. The id corresponds to the following borders:
                              0 - Top border
                              1 - Bottom border
                              2 - Left border
                              3 - Right border
        :param state: (bool): The state of the border (True for existing, False for non-existing).
        """
        try:
            self.borders[border_id] = state
        except ValueError:
            raise ValueError('Border ID have to be an int between [0, 3].')

        self.canvas.delete(self.borders_ID[border_id])
        self._draw_border(border_id)

    def _get_line_coords(self, border_id: int):
        """
        This method calculates the coordinates for borders of a given tile.
        :param border_id: (int): The id of the border for which the coordinates are to be calculated.
                              The id corresponds to the following borders:
                              0 - Top border
                              1 - Bottom border
                              2 - Left border
                              3 - Right border
        :return: tuple: A tuple containing the initial and final coordinates (x_init, y_init, x_final, y_final) of the
        border line.
        """
        if border_id == 0:  # Top border
            x_init = self.position[0]
            y_init = self.position[1]
            x_final = self.position[0] + self._length
            y_final = self.position[1]
        elif border_id == 1:  # Draw bottom border
            x_init = self.position[0]
            y_init = self.position[1] + self._length
            x_final = x_init + self._length
            y_final = y_init
        elif border_id == 2:  # Draw left border
            x_init = self.position[0]
            y_init = self.position[1]
            x_final = self.position[0]
            y_final = self.position[1] + self._length
        else:  # Draw right border
            x_init = self.position[0] + self._length
            y_init = self.position[1]
            x_final = x_init
            y_final = y_init + self._length

        return x_init, y_init, x_final, y_final

    def rotate_turtle(self, direction: str):
        """
        This method rotates the turtle based on the given direction.
        :param direction: (str) The direction to rotate the turtle.
                          It can be 'r' for right, 'l' for left, 'u' for up, and 'd' for down.
        """
        if direction in ['r', 'l', 'u', 'd']:
            self.turtle_orientation = direction
            self.turtle_image = self._get_turtle_image()
        else:
            raise ValueError("Invalid direction. It must be 'r' for right, 'l' for left, 'u' for up, or 'd' for down.")

    def _draw_turtle(self):
        """
        This method draws a turtle on the tile. The turtle is positioned at the center of the tile.

        The turtle's position is calculated as the tile's position plus a quarter of the tile's length
        (both horizontally and vertically), which places the turtle at the center of the tile.

        The turtle's image is created on the canvas at the calculated position, with its top-left corner
        anchored at the calculated position. The ID of the turtle image on the canvas is stored in the
        `turtle_ID` attribute for future reference (e.g., to erase the turtle when needed).
        """
        pos_x = self.position[0] + self._length // 4
        pos_y = self.position[1] + self._length // 4
        self.turtle_ID = self.canvas.create_image(pos_x, pos_y, image=self.turtle_image, anchor=tk.NW)

    def change_turtle_state(self, erase=True):
        """
        This method changes the state of the turtle on the tile.

        If 'erase' is True, it removes the turtle from the tile by deleting the turtle's image from the canvas.
        If 'erase' is False, it draws the turtle on the tile.

        :param erase: (bool) If True, the turtle is removed from the tile. If False, the turtle is drawn on the tile.
                      Default is True.
        """
        if erase:
            if self.turtle:  # Check if the turtle exists before trying to delete it
                self.canvas.delete(self.turtle_ID)
                self.turtle = False
        else:
            if self.turtle_ID:
                self.canvas.delete(self.turtle_ID)  # Delete the turtle if it exists
            self._draw_turtle()
            self.turtle = True


if __name__ == '__main__':
    # Create window
    window = tk.Tk()
    window.title("Tiles")
    window.configure(bg='dark gray')
    window.resizable(False, False)
    # Create Canvas
    canvas_size = 600
    canvas = tk.Canvas(window, bg="light gray", height=canvas_size, width=canvas_size)
    canvas.pack()
    tile = Tile(canvas, 0, 0, 100)
    tile.draw()
    window.mainloop()
