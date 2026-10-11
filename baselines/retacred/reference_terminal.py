"""Fixed GPU rows in a terminal; readable appended snapshots in redirected logs."""
from __future__ import annotations

import os
import shutil
import sys
import time


def duration(seconds):
    if seconds is None or seconds < 0 or seconds > 365*86400:
        return '--:--:--'
    seconds=int(seconds)
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


class Dashboard:
    def __init__(self, devices, total, *, stream=None, clock=time.monotonic):
        self.devices=list(devices);self.total=total
        self.stream=sys.stdout if stream is None else stream
        self.live=self.stream.isatty() and os.environ.get('TERM','')!='dumb'
        self.clock=clock;self.started=clock();self.height=0
        self.last_batch={};self.last_change={}

    def render(self, completed, rows, pending, *, status='RUNNING'):
        now=self.clock();width=max(20,shutil.get_terminal_size((140,24)).columns-1)
        by_gpu={row['gpu']:row for row in rows}
        lines=[f'REFERENCE SUITE | {completed}/{self.total} complete | {len(rows)} running | '
               f'{pending} pending | {duration(now-self.started)} | {status}']
        for gpu in self.devices:
            row=by_gpu.get(gpu)
            if row is None:
                lines.append(f'GPU {gpu} | waiting');continue
            identity=(gpu,row['task']);cursor=(row.get('phase'),row.get('batch'))
            if self.last_batch.get(identity)!=cursor:
                self.last_batch[identity]=cursor;self.last_change[identity]=now
            batch=row.get('batch');batches=row.get('batches')
            percent=min(1.,max(0.,batch/batches)) if isinstance(batch,(int,float)) and batches else 0.
            bar='#'*int(percent*12)+'-'*(12-int(percent*12))
            rate=row.get('batches_per_second')
            speed=f'{rate:.3f}/s' if isinstance(rate,(int,float)) else '--/s'
            memory=row.get('memory_allocated_bytes')
            vram=f'{memory/1024**3:.2f}G' if isinstance(memory,(int,float)) else '--'
            stale=now-self.last_change.get(identity,now)
            quiet=f' | no update {duration(stale)}' if stale>=60 else ''
            progress=f'{batch if batch is not None else "-"}/{batches or "-"}'
            if width<110:
                # Keep rate and memory visible on a typical 80-column SSH pane.
                task=row['task']
                task=task if len(task)<=22 else task[:19]+'...'
                lines.append(f'GPU {gpu} | {task} | {progress} {percent:4.0%} | {speed} | VRAM {vram}')
            else:
                lines.append(f'GPU {gpu} | {row["task"]} | {row.get("phase","initializing")} | '
                             f'[{bar}] {progress} {percent:5.1%} | {speed} | '
                             f'ETA {duration(row.get("eta_seconds"))} | VRAM {vram}{quiet}')
        lines=[line if len(line)<=width else line[:width-3]+'...' for line in lines]
        if self.live:
            if self.height:self.stream.write(f'\x1b[{self.height}A')
            self.stream.write(''.join('\r\x1b[2K'+line+'\n' for line in lines))
            self.height=len(lines)
        else:
            self.stream.write('\n'.join(lines)+'\n')
        self.stream.flush()

    def notice(self, message):
        # Leave the final GPU snapshot visible, then put a permanent outcome
        # below it. Never redraw over a failure reason or its log path.
        self.height=0
        self.stream.write(message+'\n');self.stream.flush()
