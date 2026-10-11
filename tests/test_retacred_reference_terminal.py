"""TTY repaint, non-TTY fallback and actionable subprocess failures."""
import io
import json
import types
import os

from baselines.retacred.reference_terminal import Dashboard


class Terminal(io.StringIO):
    def isatty(self):return True


def row():
    return {'gpu':'0','task':'ctl_drp_supervised_seed13','phase':'training',
            'batch':4,'batches':1150,'batches_per_second':.00314,
            'memory_allocated_bytes':5*1024**3,'eta_seconds':3000}


def test_tty_keeps_one_gpu_block_and_shows_nonzero_slow_rate(monkeypatch):
    monkeypatch.setenv('TERM','xterm-256color')
    stream=Terminal();dashboard=Dashboard(['0','1','2'],15,stream=stream)
    dashboard.render(0,[row()],12)
    first=stream.getvalue()
    assert first.count('GPU ')==3 and '\x1b[4A' not in first
    dashboard.render(0,[dict(row(),batch=5)],12)
    second=stream.getvalue()[len(first):]
    assert second.startswith('\x1b[4A')
    assert second.count('\x1b[2K')==4
    assert '0.003/s' in second and '5.00G' in second
    dashboard.notice('[FAILED] ctl_drp_supervised_seed13 | CUDA OOM')
    assert stream.getvalue().endswith('[FAILED] ctl_drp_supervised_seed13 | CUDA OOM\n')


def test_redirected_output_is_plain_and_missing_telemetry_is_unknown():
    stream=io.StringIO();dashboard=Dashboard(['0'],15,stream=stream)
    dashboard.render(0,[{'gpu':'0','task':'metaentail_seed13','batch':2,'batches':100}],14)
    assert '\x1b' not in stream.getvalue()
    assert '--/s' in stream.getvalue() and 'VRAM --' in stream.getvalue()


def test_narrow_terminal_keeps_rate_and_memory_visible(monkeypatch):
    from baselines.retacred import reference_terminal
    monkeypatch.setattr(reference_terminal.shutil,'get_terminal_size',lambda _:os.terminal_size((80,24)))
    stream=Terminal();dashboard=Dashboard(['0'],15,stream=stream)
    dashboard.render(0,[row()],12)
    line=stream.getvalue().splitlines()[-1]
    assert '0.003/s' in line and '5.00G' in line


def test_worker_failure_displays_exception_and_exact_log(tmp_path,monkeypatch,capsys):
    import sys
    from baselines.retacred import reference_complete as suite
    monkeypatch.delenv('QATTN_REFERENCE_STORAGE_ROOT',raising=False)
    original_sleep=suite.time.sleep
    monkeypatch.setattr(suite.time,'sleep',lambda _:original_sleep(.01))
    monkeypatch.setattr(suite,'command',lambda *a:[sys.executable,'-c',
        "print('torch.OutOfMemoryError: synthetic CUDA out of memory',flush=True);raise SystemExit(1)"])
    job={'id':'ctl_fixture_seed13','model':'fixture','seed':13,'directory':tmp_path/'task'}
    assert suite.queue([job],['0'],types.SimpleNamespace(run_dir=tmp_path),None,None)==1
    output=capsys.readouterr().out
    assert '[FAILED]' in output and 'synthetic CUDA out of memory' in output
    assert str(tmp_path/'logs/ctl_fixture_seed13.log') in output
    failed=json.loads((tmp_path/'RUN_FAILED.json').read_text())
    assert failed['failures'][0]['gpu']=='0' and failed['failures'][0]['exit_code']==1
