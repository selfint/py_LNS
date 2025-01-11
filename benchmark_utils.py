from collections import deque
from functools import wraps
from contextlib import contextmanager
import time

def benchmark(n: int = 10):
    def decorator(func):
        times = deque(maxlen=n)  # Store the last n execution times

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1_000_000  # Convert to μs
            times.append(execution_time)
            rolling_mean = sum(times) / len(times)
            print(f"{func.__name__} rolling mean execution time: {rolling_mean:.6f} μs (over {len(times)} runs)")
            return result

        return wrapper

    return decorator


class Benchmark:
    def __init__(self, n: int = 10):
        self.times = deque(maxlen=n)  # Store the last n execution times
        self.n = n

    @contextmanager
    def benchmark(self, label: str = "Benchmark"):
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1_000_000  # Convert to μs
        self.times.append(execution_time)
        rolling_mean = sum(self.times) / len(self.times)
        print(
            f"{label} rolling mean execution time: {rolling_mean:.6f} μs (over {len(self.times)} runs)"
        )
