"""Forward-module timers; CUDA events avoid synchronizing every decoding token."""
from time import perf_counter

import torch


class ModuleTimer:
    def __init__(self, module, device):
        self.device = torch.device(device)
        self.samples = []
        self.pending = []
        self.handles = [module.register_forward_pre_hook(self.start),
                        module.register_forward_hook(self.stop)]

    def stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return perf_counter()

    def start(self, module, args):
        self.pending.append(self.stamp())

    def stop(self, module, args, output):
        self.samples.append((self.pending.pop(), self.stamp()))

    def seconds(self):
        # Caller synchronizes after generation, before reading CUDA events.
        if self.device.type == "cuda":
            return [start.elapsed_time(end) / 1000 for start, end in self.samples]
        return [end - start for start, end in self.samples]

    def close(self):
        for handle in self.handles:
            handle.remove()
