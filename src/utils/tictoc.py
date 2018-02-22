"""Tic toc for measuring execution time.

Credits: https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
"""

import time
import datetime


class TicToc(object):
    """TicToc with context management.

    E.g.:
    with TicToc("Test"):
        do_something()

    """

    def __init__(self, name=None, *, format_method=None):
        """Initialize instance.

        Args:
            name (:obj:`str`): Name of this context. Defaults to None.
            format_method (:obj:`function`): format_method for outputting. Defaults to None.
        """
        self.name = name
        self.format_method = format_method

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.format_method:
            self.format_method(self.name, time.time() - self.t_start)
        else:
            if self.name:
                print("[{}]".format(self.name), end="")
            print(" Elapsed: {:.2f}s".format(time.time() - self.t_start))


# Generator method.
def TicTocGenerator():
    """Generator that returns time difference.
    """
    init_time = 0  # Initial time.
    final_time = time.time()  # Final time.
    while True:
        init_time = final_time
        final_time = time.time()
        yield final_time - init_time


tictoc = TicTocGenerator()


def tic():
    """Start recording time."""
    next(tictoc)


def toc():
    """Record and return time difference in seconds.

    Args:
        None.

    Returns:
        float: Time difference in seconds.
    """
    return next(tictoc)


def seconds_to_readable(seconds: float):
    """Convert seconds to human readable format. E.g., 61.0 --> 0 days 0 hours 1 minutes 1 seconds

    Args:
        seconds (float): Seconds to be converted.

    Returns:
        str: Human readable time format.
    """
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)

    return "{:d} days {:02d}:{:02d}:{:02d}".format(int(days), int(hours), int(mins), int(secs))


if __name__ == "__main__":

    print("Testing context manager within 5s...")
    with TicToc("Test"):
        time.sleep(5)

    print("\nTesting tic toc within 5s...")
    tic()
    time.sleep(5)
    time_diff = toc()
    print("toc() returns {}.".format(seconds_to_readable(time_diff)))




