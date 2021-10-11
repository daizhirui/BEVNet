import os
import sys
import tempfile

import torch
from torch.profiler import ProfilerActivity


class PatchedAutogradProfiler(torch.autograd.profiler.profile):

    def export_chrome_trace(self, path):
        self._check_finish()
        if self.kineto_results is not None:
            if sys.platform == 'win32':
                fp = tempfile.NamedTemporaryFile(
                    'w+t', suffix='.json', delete=False
                )
                fp.close()
                self.kineto_results.save(fp.name)
                with open(fp.name) as fin:
                    with open(path, 'wt') as fout:
                        for line in fin.readlines():
                            if "Call stack" in line:
                                line = line.replace("\\", "\\\\")
                            fout.write(line)
                os.remove(fp.name)
            else:
                self.kineto_results.save(path)
        else:
            assert self.function_events is not None
            return self.function_events.export_chrome_trace(path)

    def prepare_trace(self):
        self._prepare_trace()


class PatchedProfiler(torch.profiler.profile):
    def _start_warmup(self):
        self.profiler = PatchedAutogradProfiler(
            use_cuda=(ProfilerActivity.CUDA in self.activities),
            use_cpu=(ProfilerActivity.CPU in self.activities),
            record_shapes=self.record_shapes,
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            use_kineto=True,
        )
        self.profiler.prepare_trace()
