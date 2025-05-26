import threading
from concurrent.futures import Future

class LLMThread:
    class _Job:
        def __init__(self, func, priority_func, future, args, kwargs):
            self.func = func
            self.priority_func = priority_func
            self.future = future
            self.args = args
            self.kwargs = kwargs

    def __init__(self):
        self.job_hash: set[LLMThread._Job] = set()
        self.lock = threading.Lock()
        self._not_empty = threading.Condition(self.lock)
        self.thread = threading.Thread(target=self.llm_worker_loop, daemon=True)
        self.thread.start()

    def llm_worker_loop(self):
        while True:
            with self._not_empty:
                while not self.job_hash:
                    self._not_empty.wait()

                # Get the highest priority job
                highest_job: None | LLMThread._Job = None

                # Priority is a tuple of (frame last drawn, area in pixels on screen)
                highest_priority: tuple[int, float] = (0, 0)
                for job in self.job_hash:
                    priority = job.priority_func()
                    if highest_job is None:
                        highest_job = job
                        highest_priority = priority
                    else:
                        # If one was requested more recently...
                        if highest_priority[0] < priority[0]:
                            # ...do it first.
                            highest_job = job
                            highest_priority = priority
                        elif highest_priority[0] == priority[0]:
                            # Otherwise, if they are just as recent...
                            if highest_priority[1] < priority[1]:
                                # Do the one that has a larger visible area first.
                                highest_job = job
                                highest_priority = priority

                # Remove future from the hashset
                if highest_job is not None:
                    self.job_hash.remove(highest_job)

            if highest_job is None:
                continue

            if highest_job.future.cancelled():
                continue

            try:
                result = highest_job.func(*highest_job.args, **highest_job.kwargs)
                highest_job.future.set_result(result)
            except Exception as e:
                highest_job.future.set_exception(e)

    # @timed("submit")
    def submit(self, func, priority_func, *args, **kwargs) -> Future:
        future = Future()
        job = LLMThread._Job(func, priority_func, future, args, kwargs)
        with self._not_empty:
            self.job_hash.add(job)
            self._not_empty.notify()
        return future
