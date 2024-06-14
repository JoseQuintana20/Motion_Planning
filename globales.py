"""
This module contains global variables that are used for thread synchronization and inter-thread communication in a
labyrinth project. The module uses Python's built-in threading and queue modules to create a lock object and a queue
object. The lock object, `candado`, is used to ensure that only one thread can execute a particular section of code at a
time, which is useful for preventing race conditions. The queue object, `cola`, is used for inter-thread communication,
where one thread can put messages (or any Python data type) into the queue, and another thread can retrieve them.

This module is intended to be imported by other modules in the project that require thread synchronization and
inter-thread communication.

Attributes:
----------
candado : threading.Lock
    A lock object to handle synchronization between threads.
cola : queue.Queue
    A queue object to handle inter-thread communication.
"""

import threading
import queue

# A lock object to handle synchronization between threads
candado = threading.Lock()

# A queue object to handle inter-thread communication
cola = queue.Queue()
