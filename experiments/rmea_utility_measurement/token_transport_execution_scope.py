"""Lifetime guard for one short, checkpoint-free scheduled Python process."""
import ctypes
import json
import os
from pathlib import Path
import sys
import time

class Timespec(ctypes.Structure):
    _fields_=[('seconds',ctypes.c_long),('nanoseconds',ctypes.c_long)]
class Itimerspec(ctypes.Structure):
    _fields_=[('interval',Timespec),('value',Timespec)]
class Sigevent(ctypes.Structure):
    _fields_=[('value',ctypes.c_void_p),('signo',ctypes.c_int),('notify',ctypes.c_int),('padding',ctypes.c_int*12)]

def read(path):return json.loads(Path(path).read_bytes())

def check_launch(directory):
    directory=Path(directory);request=read(directory/'request.json');lease=read(directory/'lease.json')
    if lease['token']!=request['token'] or time.monotonic()>=lease['until']:raise TimeoutError('Attempt lease expired')
    cancel=directory/'cancel.json'
    if cancel.exists():
        if read(cancel)['token']!=request['token']:raise ValueError('Cancellation identity differs')
        raise RuntimeError('Attempt cancelled')

def arm(directory,seconds):
    import platform
    assert sys.platform=='linux' and platform.machine()=='x86_64'
    directory=Path(directory);parent=os.getppid();lib=ctypes.CDLL(None,use_errno=True)
    lib.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    assert parent>1 and lib.prctl(1,9,0,0,0)==0
    if os.getppid()!=parent:raise RuntimeError('Supervisor exited before guard')
    check_launch(directory)
    # request.json is written once at worker admission. Count delayed launch
    # against the same budget; wall clock must not move backwards at admission.
    age=time.time()-(directory/'request.json').stat().st_mtime
    if not -5<=age<seconds:raise TimeoutError('Original admission deadline passed')
    deadline=time.monotonic()+seconds-max(0,age)
    rt=ctypes.CDLL('librt.so.1',use_errno=True);timer=ctypes.c_void_p();event=Sigevent();event.signo=9;event.notify=0
    assert ctypes.sizeof(event)==64
    rt.timer_create.argtypes=[ctypes.c_int,ctypes.POINTER(Sigevent),ctypes.POINTER(ctypes.c_void_p)]
    rt.timer_settime.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(Itimerspec),ctypes.c_void_p]
    if rt.timer_create(1,ctypes.byref(event),ctypes.byref(timer))!=0:raise OSError(ctypes.get_errno(),'timer_create')
    second=int(deadline);spec=Itimerspec(Timespec(0,0),Timespec(second,int((deadline-second)*1e9)))
    if rt.timer_settime(timer,1,ctypes.byref(spec),None)!=0:raise OSError(ctypes.get_errno(),'timer_settime')
    # No exec or scientific subprocess after arming. Kernel deadline and
    # parent-death SIGKILL remain active for this exact interpreter.
    check_launch(directory)
    return dict(mode='single-process-kernel-deadline',deadline_monotonic=deadline,admission_age_seconds=age)
