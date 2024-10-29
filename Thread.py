"""
Function to handle closing of thread

"""

import threading
import time
import inspect
import ctypes

def raise_async_exception(thread_id, exception_type):
    """
    Raises an asynchronous exception in the specified thread.
    Performs cleanup if necessary.

    Parameters:
    - thread_id: ID of the thread where the exception will be raised.
    - exception_type: Type of the exception to be raised.
    """
    # Convert thread ID to a ctypes long object
    thread_id = ctypes.c_long(thread_id)

    # Ensure the exception type is a class
    if not inspect.isclass(exception_type):
        exception_type = type(exception_type)

    # Attempt to raise the exception in the target thread
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(exception_type))
    
    if result == 0:
        # No thread found with the given ID
        raise ValueError("Invalid thread ID")
    elif result != 1:
        # If more than one thread was affected, cancel the exception
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    """
    Stops the given thread by raising a SystemExit exception.
    
    Parameters:
    - thread: The thread object to be stopped.
    """
    raise_async_exception(thread.ident, SystemExit)


if __name__ == "__main__":
    pass
   