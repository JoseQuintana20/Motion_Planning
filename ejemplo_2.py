"""
This is an example of multi-threading execution
Front-end and Back-end modules are running in different threads
Inter-thread comm is implemented with a Queue and a mutex lock

Daniel Zapata Y.
German A Holguin L.
UTP - Pereira, Colombia 2024.
"""

from labyrinth import Labyrinth
import threading
from worker import trabajador

# Macro expansions -sort of
ROWS = 10
COLUMNS = 20


def create_labyrinth():
    """
    This function creates a Labyrinth object with the specified number of rows and columns, and starts the Tkinter event loop.

    The function uses the global constants ROWS and COLUMNS to determine the size of the labyrinth. The labyrinth is then
    displayed on the screen by calling the start method of the Labyrinth class.

    :return: None
    """
    maze = Labyrinth(ROWS, COLUMNS)
    maze.start()


def create_graph():
    """
    This function creates a graph representation of the labyrinth using the trabajador function.

    The function uses the global constants ROWS and COLUMNS to determine the size of the graph. The graph is then
    created by calling the trabajador function with the number of rows and columns as arguments.
    :return: None
    """
    trabajador(ROWS, COLUMNS)


if __name__ == '__main__':
    # Create two threads to run the create_labyrinth and create_graph functions concurrently
    # create_labyrinth function is executed in a separate thread
    hilo1 = threading.Thread(target=create_labyrinth)
    hilo1.start()  # start the thread

    # create_graph function is executed in a separate thread
    hilo2 = threading.Thread(target=create_graph)
    hilo2.start()

    # Wait for the threads to finish
    hilo2.join()
    hilo1.join()
