#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The original version of this file is from https://github.com/wookayin/gpustat
Thanks to the author of gpustat, Jongwook Choi
"""
import json
import locale
import os.path
import platform
import sys
import time
from datetime import datetime

import psutil
from blessed import Terminal
from six.moves import cStringIO as StringIO

from .pynvml import NVMLError
from .pynvml import NVML_TEMPERATURE_GPU
from .pynvml import nvmlDeviceGetComputeRunningProcesses
from .pynvml import nvmlDeviceGetCount
from .pynvml import nvmlDeviceGetDecoderUtilization
from .pynvml import nvmlDeviceGetEncoderUtilization
from .pynvml import nvmlDeviceGetEnforcedPowerLimit
from .pynvml import nvmlDeviceGetFanSpeed
from .pynvml import nvmlDeviceGetGraphicsRunningProcesses
from .pynvml import nvmlDeviceGetHandleByIndex
from .pynvml import nvmlDeviceGetIndex
from .pynvml import nvmlDeviceGetMemoryInfo
from .pynvml import nvmlDeviceGetName
from .pynvml import nvmlDeviceGetPowerUsage
from .pynvml import nvmlDeviceGetTemperature
from .pynvml import nvmlDeviceGetUUID
from .pynvml import nvmlDeviceGetUtilizationRates
from .pynvml import nvmlInit
from .pynvml import nvmlShutdown
from .pynvml import nvmlSystemGetCudaDriverVersion
from .pynvml import nvmlSystemGetDriverVersion
from .util import bytes2human
from .util import prettify_commandline
from ..log import get_logger

NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024

IS_WINDOWS = 'windows' in platform.platform().lower()

__all__ = ['GPUStat', 'GPUStatCollection']

logger = get_logger(__name__)


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError(
                'entry should be a dict, {} given'.format(type(entry))
            )
        self.entry = entry

    def __repr__(self):
        return self.print_to(StringIO()).getvalue()

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def index(self):
        """
        Returns the index of GPU (as in nvidia-smi).
        """
        return self.entry['index']

    @property
    def uuid(self):
        """
        Returns the uuid returned by nvidia-smi,
        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def name(self):
        """
        Returns the name of GPU card (e.g. GeForce Titan X)
        """
        return self.entry['name']

    @property
    def memory_total(self):
        """
        Returns the total memory (in MB) as an integer.
        """
        return int(self.entry['memory.total'])

    @property
    def memory_used(self):
        """
        Returns the occupied memory (in MB) as an integer.
        """
        return int(self.entry['memory.used'])

    @property
    def memory_free(self):
        """
        Returns the free (available) memory (in MB) as an integer.
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def memory_available(self):
        """
        Returns the available memory (in MB) as an integer.
        Alias of memory_free.
        """
        return self.memory_free

    @property
    def temperature(self):
        """
        Returns the temperature (in Celsius) of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def fan_speed(self):
        """
        Returns the fan speed percentage (0-100) of maximum intended speed
        as an integer, or None if the information is not available.
        """
        v = self.entry['fan.speed']
        return int(v) if v is not None else None

    @property
    def utilization(self):
        """
        Returns the GPU utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.gpu']
        return int(v) if v is not None else None

    @property
    def utilization_enc(self):
        """
        Returns the GPU encoder utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.enc']
        return int(v) if v is not None else None

    @property
    def utilization_dec(self):
        """
        Returns the GPU decoder utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.dec']
        return int(v) if v is not None else None

    @property
    def power_draw(self):
        """
        Returns the GPU power usage in Watts,
        or None if the information is not available.
        """
        v = self.entry['power.draw']
        return int(v) if v is not None else None

    @property
    def power_limit(self):
        """
        Returns the (enforced) GPU power limit in Watts,
        or None if the information is not available.
        """
        v = self.entry['enforced.power.limit']
        return int(v) if v is not None else None

    @property
    def processes(self):
        """
        Get the list of running processes on the GPU.
        """
        return self.entry['processes']

    def print_to(
        self, fp,
        with_colors=True,  # deprecated arg
        as_list=False,
        show_cmd=False,
        show_full_cmd=False,
        show_user=False,
        show_pid=False,
        show_fan_speed=None,
        show_codec="",
        show_power=None,
        gpuname_width=16,
        term=None,
    ):
        if term is None:
            term = Terminal(stream=sys.stdout)

        # color settings
        colors = {}

        def _conditional(
            cond_fn, true_value, false_value,
            error_value=term.bold_black
        ):
            try:
                return cond_fn() and true_value or false_value
            except Exception:
                return error_value

        _ENC_THRESHOLD = 50

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CBold'] = term.bold
        colors['CName'] = term.blue
        colors['CTemp'] = _conditional(lambda: self.temperature < 50,
                                       term.red, term.bold_red)
        colors['FSpeed'] = _conditional(lambda: self.fan_speed < 30,
                                        term.cyan, term.bold_cyan)
        colors['CMemU'] = term.bold_yellow
        colors['CMemT'] = term.yellow
        colors['CMemP'] = term.yellow
        colors['CCPUMemU'] = term.yellow
        colors['CUser'] = term.bold_black  # gray
        colors['CUtil'] = _conditional(lambda: self.utilization < 30,
                                       term.green, term.bold_green)
        colors['CUtilEnc'] = _conditional(
            lambda: self.utilization_enc < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CUtilDec'] = _conditional(
            lambda: self.utilization_dec < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CCPUUtil'] = term.green
        colors['CPowU'] = _conditional(
            lambda: (self.power_limit is not None and
                     float(self.power_draw) / self.power_limit < 0.4),
            term.magenta, term.bold_magenta
        )
        colors['CPowL'] = term.magenta
        colors['CCmd'] = term.color(24)  # a bit dark

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value):
            return none_value if v is None else v

        # build one-line display information
        # we want power use optional, but if deserves being grouped with
        # temperature and utilization
        reps = u"%(C1)s[{entry[index]}]%(C0)s " \
               "%(CName)s{entry[name]:{gpuname_width}}%(C0)s |" \
               "%(CTemp)s{entry[temperature.gpu]:>3}°C%(C0)s, "

        if show_fan_speed:
            reps += "%(FSpeed)s{entry[fan.speed]:>3} %%%(C0)s, "

        reps += "%(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s"
        if show_codec:
            codec_info = []
            if "enc" in show_codec:
                codec_info.append(
                    "%(CBold)sE: %(C0)s"
                    "%(CUtilEnc)s{entry[utilization.enc]:>3} %%%(C0)s")
            if "dec" in show_codec:
                codec_info.append(
                    "%(CBold)sD: %(C0)s"
                    "%(CUtilDec)s{entry[utilization.dec]:>3} %%%(C0)s")
            reps += " ({})".format("  ".join(codec_info))

        if show_power:
            reps += ",  %(CPowU)s{entry[power.draw]:>3}%(C0)s "
            if show_power is True or 'limit' in show_power:
                reps += "/ %(CPowL)s{entry[enforced.power.limit]:>3}%(C0)s "
                reps += "%(CPowL)sW%(C0)s"
            else:
                reps += "%(CPowU)sW%(C0)s"

        reps += " | %(C1)s%(CMemU)s{entry[memory.used]:>5}%(C0)s " \
                "/ %(CMemT)s{entry[memory.total]:>5}%(C0)s MB"
        reps = reps % colors
        reps = reps.format(
            entry={k: _repr(v, '??') for k, v in self.entry.items()},
            gpuname_width=gpuname_width
        )
        reps += " |"

        def process_repr(_p):
            r = ''
            if not show_cmd or show_user:
                r += "{CUser}{}{C0}".format(
                    _repr(_p['username'], '--'), **colors
                )
            if show_cmd:
                if r:
                    r += ':'
                r += "{C1}{}{C0}".format(
                    _repr(_p.get('command', _p['pid']), '--'), **colors
                )

            if show_pid:
                r += ("/%s" % _repr(_p['pid'], '--'))
            r += '({CMemP}{}M{C0})'.format(
                _repr(_p['gpu_memory_usage'], '?'), **colors
            )
            return r

        def list_process_info(_p):
            r = "{C0} ├─ {:>6} ".format(
                _repr(_p['pid'], '--'), **colors
            )
            r += "{C0}({CCPUUtil}{:4.0f}%{C0}, {CCPUMemU}{:>6}{C0}".format(
                _repr(_p['cpu_percent'], '--'),
                bytes2human(_repr(_p['cpu_memory_usage'], 0)), **colors
            )
            r += ', {CMemP}{}MB{C0}, {})'.format(
                _repr(_p['gpu_memory_usage'], '?'), '+'.join(_p['mode']),
                **colors
            )
            if show_full_cmd:
                command_pretty = prettify_commandline(
                    _p['full_command'], colors['C1'], colors['CCmd'])
            else:
                command_pretty = prettify_commandline(
                    _p['command'], colors['C1'], colors['CCmd']
                )
            r += "{C0}: {CCmd}{}{C0}".format(
                _repr(command_pretty, '?'),
                **colors
            )
            return r

        processes = self.entry['processes']
        list_processes = []
        if processes is None:
            # None (not available)
            reps += ' ({})'.format(NOT_SUPPORTED)
        else:
            for p in processes:
                if show_full_cmd or as_list:
                    list_processes.append(os.linesep + list_process_info(p))
                else:
                    reps += ' ' + process_repr(p)
        if (show_full_cmd or as_list) and list_processes:
            list_processes[-1] = list_processes[-1].replace('├', '└', 1)
            reps += ''.join(list_processes)
        fp.write(reps)
        return fp

    def jsonify(self):
        o = self.entry.copy()
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        return o


class GPUStatCollection(object):
    global_processes = {}

    def __init__(self, gpu_list, driver_version=None, cuda_version=None):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        self.driver_version = driver_version
        self.cuda_version = cuda_version

    @staticmethod
    def clean_processes():
        for pid in list(GPUStatCollection.global_processes.keys()):
            if not psutil.pid_exists(pid):
                del GPUStatCollection.global_processes[pid]

    @staticmethod
    def new_query():
        """Query the information of all the GPUs on local machine"""

        nvmlInit()

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode('utf-8')  # for python3, to unicode
            return b

        def get_gpu_info(_handle):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(_nv_process):
                """Get the process information of specific pid"""
                _process = {}
                if _nv_process.pid not in GPUStatCollection.global_processes:
                    GPUStatCollection.global_processes[_nv_process.pid] = \
                        psutil.Process(pid=_nv_process.pid)
                ps_process = GPUStatCollection.global_processes[_nv_process.pid]

                # TODO: ps_process is being cached, but the dict below is not.
                _process['username'] = ps_process.username()
                # cmdline returns full path;
                # as in `ps -o comm`, get short cmdnames.
                _cmdline = ps_process.cmdline()
                if not _cmdline:
                    # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                    _process['command'] = '?'
                    _process['full_command'] = ['?']
                else:
                    _process['command'] = os.path.basename(_cmdline[0])
                    _process['full_command'] = _cmdline
                # Bytes to MBytes
                # if drivers are not TTC this will be None.
                used_mem = _nv_process.usedGpuMemory // MB if \
                    _nv_process.usedGpuMemory else None
                _process['gpu_memory_usage'] = used_mem
                _process['cpu_percent'] = ps_process.cpu_percent()
                _process['cpu_memory_usage'] = \
                    round((ps_process.memory_percent() / 100.0) *
                          psutil.virtual_memory().total)
                _process['pid'] = _nv_process.pid
                return _process

            name = _decode(nvmlDeviceGetName(_handle))
            uuid = _decode(nvmlDeviceGetUUID(_handle))

            try:
                temperature = nvmlDeviceGetTemperature(
                    _handle, NVML_TEMPERATURE_GPU
                )
            except NVMLError as err:
                logger.warn(str(err))
                temperature = None  # Not supported

            try:
                fan_speed = nvmlDeviceGetFanSpeed(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                fan_speed = None  # Not supported

            try:
                memory = nvmlDeviceGetMemoryInfo(_handle)  # in Bytes
            except NVMLError as err:
                logger.warn(str(err))
                memory = None  # Not supported

            try:
                utilization = nvmlDeviceGetUtilizationRates(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                utilization = None  # Not supported

            try:
                utilization_enc = nvmlDeviceGetEncoderUtilization(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                utilization_enc = None  # Not supported

            try:
                utilization_dec = nvmlDeviceGetDecoderUtilization(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                utilization_dec = None  # Not supported

            try:
                power = nvmlDeviceGetPowerUsage(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                power = None

            try:
                power_limit = nvmlDeviceGetEnforcedPowerLimit(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                power_limit = None

            try:
                nv_comp_processes = \
                    nvmlDeviceGetComputeRunningProcesses(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = \
                    nvmlDeviceGetGraphicsRunningProcesses(_handle)
            except NVMLError as err:
                logger.warn(str(err))
                nv_graphics_processes = None  # Not supported

            if nv_comp_processes is None and nv_graphics_processes is None:
                processes = None
            else:
                processes = dict()

                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                # A single process might run in both of graphics and compute
                # mode, However we will display the process only once
                # seen_pids = set()
                for mode, nv_processes in zip(
                    ['C', 'G'], [nv_comp_processes, nv_graphics_processes]
                ):
                    for nv_process in nv_processes:
                        if nv_process.pid in processes:
                            processes[nv_process.pid]['mode'].append(mode)
                            continue
                        # seen_pids.add(nv_process.pid)
                        try:
                            process = get_process_info(nv_process)
                            process['mode'] = [mode]
                            processes[nv_process.pid] = process
                        except psutil.NoSuchProcess:
                            # TODO: add some reminder for NVML broken context
                            # e.g. nvidia-smi reset  or  reboot the system
                            pass
                        except psutil.AccessDenied:
                            pass
                        except FileNotFoundError:
                            # Ignore the exception which probably has occurred
                            # from psutil, due to a non-existent PID (see #95).
                            # The exception should have been translated, but
                            # there appears to be a bug of psutil. It is
                            # unlikely FileNotFoundError is thrown in
                            # different situations.
                            pass

                # TODO: Do not block if full process info is not requested
                time.sleep(0.1)
                for pid, process in processes.items():
                    cache_process = GPUStatCollection.global_processes[pid]
                    process['cpu_percent'] = cache_process.cpu_percent()

            _gpu_info = {
                'index': nvmlDeviceGetIndex(_handle),
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'fan.speed': fan_speed,
                'utilization.gpu':
                    utilization.gpu if utilization else None,
                'utilization.enc':
                    utilization_enc[0] if utilization_enc else None,
                'utilization.dec':
                    utilization_dec[0] if utilization_dec else None,
                'power.draw':
                    power // 1000 if power is not None else None,
                'enforced.power.limit': power_limit // 1000
                if power_limit is not None else None,
                # Convert bytes into MBytes
                'memory.used': memory.used // MB if memory else None,
                'memory.total': memory.total // MB if memory else None,
                'processes': list(processes.values()),
            }
            GPUStatCollection.clean_processes()
            return _gpu_info

        # 1. get the list of gpu and status
        gpu_list = []
        device_count = nvmlDeviceGetCount()

        for index in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(index)
            gpu_info = get_gpu_info(handle)
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

        # 2. additional info (driver version, etc).
        try:
            driver_version = _decode(nvmlSystemGetDriverVersion())
        except NVMLError as e:
            logger.warn(str(e))
            driver_version = None  # N/A

        try:
            cuda_version = nvmlSystemGetCudaDriverVersion() // 10
            main_version = cuda_version // 100
            sub_version = cuda_version % 100
            cuda_version = f'{main_version}.{sub_version}'
        except NVMLError as e:
            logger.warn(str(e))
            cuda_version = None  # N/A

        nvmlShutdown()
        return GPUStatCollection(gpu_list, driver_version, cuda_version)

    def __len__(self):
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self):
        s = 'GPUStatCollection(host=%s, [\n' % self.hostname
        s += '\n'.join('  ' + str(g) for g in self.gpus)
        s += '\n])'
        return s

    # --- Printing Functions ---

    def print_formatted(
        self, fp=sys.stdout, force_color=False, no_color=False, as_list=True,
        show_cmd=False, show_full_cmd=False, show_user=False,
        show_pid=False, show_fan_speed=None,
        show_codec="", show_power=None,
        gpuname_width=16, show_header=True,
        eol_char=os.linesep,
    ):
        # ANSI color configuration
        if force_color and no_color:
            raise ValueError("--color and --no_color can't"
                             " be used at the same time")

        if force_color:
            TERM = os.getenv('TERM') or 'xterm-256color'
            t_color = Terminal(kind=TERM, force_styling=True)

            # workaround of issue #32 (watch doesn't recognize sgr0 characters)
            t_color._normal = u'\x1b[0;10m'
        elif no_color:
            t_color = Terminal(force_styling=False)
        else:
            t_color = Terminal()  # auto, depending on isatty

        # appearance settings
        entry_name_width = [len(g.entry['name']) for g in self]
        gpuname_width = max([gpuname_width or 0] + entry_name_width)

        # header
        if show_header:
            if IS_WINDOWS:
                # no localization is available; just use a reasonable default
                # same as str(time_str) but without ms
                time_str = self.query_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_format = locale.nl_langinfo(locale.D_T_FMT)
                time_str = self.query_time.strftime(time_format)
            header_template = '{t.bold_white}{hostname:{width}}{t.normal}  '
            header_template += '{time_str}  '
            header_template += '{t.bold_black}Driver:{driver_version}{t.normal}'
            header_template += '  {t.bold_black}CUDA:{cuda_version}{t.normal}'

            header_msg = header_template.format(
                hostname=self.hostname,
                width=gpuname_width + 3,  # len("[?]")
                time_str=time_str,
                driver_version=self.driver_version,
                cuda_version=self.cuda_version,
                t=t_color,
            )

            fp.write(header_msg.strip())
            fp.write(eol_char)

        # body
        for g in self:
            g.print_to(fp,
                       as_list=as_list,
                       show_cmd=show_cmd,
                       show_full_cmd=show_full_cmd,
                       show_user=show_user,
                       show_pid=show_pid,
                       show_fan_speed=show_fan_speed,
                       show_codec=show_codec,
                       show_power=show_power,
                       gpuname_width=gpuname_width,
                       term=t_color)
            fp.write(eol_char)

        fp.flush()

    def jsonify(self):
        return {
            'hostname': self.hostname,
            'query_time': self.query_time,
            "gpus": [g.jsonify() for g in self]
        }

    def print_json(self, fp=sys.stdout):
        def date_handler(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                raise TypeError(type(obj))

        o = self.jsonify()
        json.dump(o, fp, indent=4, separators=(',', ': '),
                  default=date_handler)
        fp.write(os.linesep)
        fp.flush()
