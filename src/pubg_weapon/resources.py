"""Measure wall time, process-tree memory and volume use by pipeline stage."""
from contextlib import contextmanager
import shutil
from threading import Event, Thread
import time
import sys

import psutil

MIB = 1024 ** 2


@contextmanager
def measure(stages, name, directory):
    started = time.perf_counter()
    process = psutil.Process()
    done = Event()
    start_free_disk = shutil.disk_usage(directory).free
    peak_rss = 0
    min_free_disk = start_free_disk
    initial_cpu = process.cpu_times()
    torch = sys.modules.get('torch')
    use_gpu = torch is not None and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def sample():
        nonlocal peak_rss, min_free_disk
        rss = 0
        for child in [process, *process.children(recursive=True)]:
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_rss = max(peak_rss, rss)
        min_free_disk = min(min_free_disk, shutil.disk_usage(directory).free)

    def monitor():
        while not done.wait(0.25):
            sample()

    sample()
    watcher = Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()
        sample()
        duration = time.perf_counter() - started
        if use_gpu:
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
        final_cpu = process.cpu_times()
        stages[name] = {
            'wall_seconds': round(duration, 3),
            'process_tree_peak_rss_mib': round(peak_rss / MIB, 1),
            'volume_peak_extra_used_mib': round((start_free_disk - min_free_disk) / MIB, 1),
            'minimum_volume_free_mib': round(min_free_disk / MIB, 1),
            'main_process_cpu_seconds': round(final_cpu.user + final_cpu.system
                                              - initial_cpu.user - initial_cpu.system, 3),
            'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / MIB, 1) if use_gpu else None,
        }
        print(f'{name}: {duration:.1f}s, RAM đỉnh {peak_rss / MIB:,.0f} MiB', flush=True)
