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
    pending=[f'pending{i}_seed13' for i in range(12)]
    dashboard.render([],[row()],pending)
    first=stream.getvalue()
    assert first.count('GPU ')==3 and '\x1b[12A' not in first
    assert 'PENDING (12)' in first and 'COMPLETED (0)' in first
    dashboard.render(['finished_seed13'],[dict(row(),batch=5)],pending[1:])
    second=stream.getvalue()[len(first):]
    assert second.startswith('\x1b[12A')
    assert second.count('\x1b[2K')==12
    assert '0.003/s' in second and '5.00G' in second
    assert 'PENDING (11)' in second and 'COMPLETED (1)' in second and 'finished_seed13' in second
    dashboard.notice('[FAILED] ctl_drp_supervised_seed13 | CUDA OOM')
    assert stream.getvalue().endswith('[FAILED] ctl_drp_supervised_seed13 | CUDA OOM\n')


def test_redirected_output_is_plain_and_missing_telemetry_is_unknown():
    stream=io.StringIO();dashboard=Dashboard(['0'],15,stream=stream)
    dashboard.render([],[{'gpu':'0','task':'metaentail_seed13','batch':2,'batches':100}],['next_seed13'])
    assert '\x1b' not in stream.getvalue()
    assert '--/s' in stream.getvalue() and 'VRAM --' in stream.getvalue()


def test_narrow_terminal_keeps_rate_and_memory_visible(monkeypatch):
    from baselines.retacred import reference_terminal
    monkeypatch.setenv('TERM','xterm-256color')
    monkeypatch.setattr(reference_terminal.shutil,'get_terminal_size',lambda _:os.terminal_size((80,24)))
    stream=Terminal();dashboard=Dashboard(['0'],15,stream=stream)
    dashboard.render([],[row()],['next_seed13'])
    line=stream.getvalue().replace('\r\x1b[2K','').splitlines()[1]
    assert '0.003/s' in line and '5.00G' in line


def test_queue_order_overflow_and_empty_areas_have_fixed_height(monkeypatch):
    from baselines.retacred import reference_terminal
    monkeypatch.setenv('TERM','xterm-256color')
    monkeypatch.setattr(reference_terminal.shutil,'get_terminal_size',lambda _:os.terminal_size((80,24)))
    stream=Terminal();dashboard=Dashboard(['0','1','2'],15,stream=stream)
    pending=[f'long_reference_model_{i}_seed13' for i in range(15)]
    completed=['first_completed_seed13','last_completed_seed13']
    dashboard.render(completed,[],pending)
    lines=stream.getvalue().replace('\r\x1b[2K','').splitlines()
    assert len(lines)==12
    assert 'PENDING (15)' in lines[4] and '01. long_reference_model_0_seed13' in lines[5]
    assert '+10 more' in lines[7]
    assert 'COMPLETED (2)' in lines[8]
    assert lines[9].index('last_completed_seed13')<lines[9].index('first_completed_seed13')
    assert all(len(line.replace('\r\x1b[2K',''))<=79 for line in lines)
    before=len(stream.getvalue())
    dashboard.render([],[],[])
    final=stream.getvalue()[before:]
    assert final.startswith('\x1b[12A') and final.count('\x1b[2K')==12
    assert final.count('(empty)')==2


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
