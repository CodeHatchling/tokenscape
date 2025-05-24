import time
import functools
nesting_depth: int = 0

class Timer:
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        global nesting_depth
        print(f"{'. ' * nesting_depth}Entering {self.name}", flush=True)
        nesting_depth += 1
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        global nesting_depth
        nesting_depth -= 1
        self.interval = self.end - self.start
        print(f"{'. ' * nesting_depth}{self.name} took {self.interval * 1000:.6f} milliseconds", flush=True)


# Wrap it as a decorator
def timed(label="task"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(label):
                return func(*args, **kwargs)
        return wrapper
    return decorator
