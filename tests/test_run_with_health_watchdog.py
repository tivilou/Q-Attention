import signal

import scripts.run_with_health_watchdog as watchdog


def test_watchdog_signal_handler_stops_child(monkeypatch):
    calls = []

    class Process:
        pass

    process = Process()

    def stop_process(value, reason):
        calls.append((value, reason))
        return 124

    monkeypatch.setattr(watchdog, "stop_process", stop_process)
    monkeypatch.setattr(watchdog.signal, "signal", lambda *_args: None)
    watchdog._install_signal_handlers(process)

    handler = None

    def capture(signum, callback):
        nonlocal handler
        if signum == signal.SIGTERM:
            handler = callback

    monkeypatch.setattr(watchdog.signal, "signal", capture)
    watchdog._install_signal_handlers(process)
    assert handler is not None

    try:
        handler(signal.SIGTERM, None)
    except SystemExit as exc:
        assert exc.code == 128 + signal.SIGTERM
    else:  # pragma: no cover
        raise AssertionError("signal handler did not exit")
    assert calls == [(process, "received_signal=SIGTERM")]
