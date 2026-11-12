"""
hardware/alarm_controller.py
AlarmController — drives audio feedback for pass/fail results.

Primary driver : winsound  (Windows built-in, zero extra install)
Fallback output: Coloured console message via ANSI / colorama

All beeps are executed in daemon threads so they never block the inspection loop.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers (ANSI — works in Windows Terminal / PowerShell 7)
# ---------------------------------------------------------------------------
try:
    import colorama  # type: ignore

    colorama.init(autoreset=True)
    _GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    _RED = colorama.Fore.RED + colorama.Style.BRIGHT
    _YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    _RESET = colorama.Style.RESET_ALL
except ImportError:
    # Fallback: raw ANSI codes (Windows Terminal supports these)
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _YELLOW = "\033[93m"
    _RESET = "\033[0m"

# ---------------------------------------------------------------------------
# winsound availability
# ---------------------------------------------------------------------------
try:
    import winsound as _ws

    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False
    logger.warning("winsound not available (non-Windows host) — audio disabled")


def _beep(freq_hz: int, duration_ms: int) -> None:
    """Emit a single beep; silently skips on non-Windows."""
    if _WINSOUND_AVAILABLE:
        _ws.Beep(freq_hz, duration_ms)
    else:
        logger.debug("BEEP skipped — winsound unavailable (%d Hz, %d ms)", freq_hz, duration_ms)


def _run_in_thread(target, *args) -> threading.Thread:
    """Spawn and start a daemon thread; return the thread object."""
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# AlarmController
# ---------------------------------------------------------------------------

class AlarmController:
    """
    Controls audible (and visual) alarms for the inspection station.

    Usage::

        with AlarmController() as alarm:
            # on pass
            alarm.beep_pass()
            # on failure
            alarm.alarm_ng("SIZE_MISMATCH")

    Thread safety: beep methods launch daemon threads and return immediately.
    The main inspection loop is never blocked.
    """

    # Beep pattern definitions: list of (freq_hz, duration_ms, gap_ms)
    _PATTERNS = {
        "SIZE_MISMATCH": [
            (500, 300, 80),
            (500, 300, 80),
            (500, 300, 0),
        ],
        "MISSING_REQUIRED_LABEL": [
            (400, 1000, 200),
            (400, 1000, 0),
        ],
        "SIZE_NOT_FOUND": [
            (600, 150, 60),
            (600, 150, 60),
            (600, 800, 0),
        ],
        "TEMPLATE_NOT_FOUND": [
            (450, 400, 100),
            (450, 400, 0),
        ],
        "LOW_CONFIDENCE": [
            (700, 200, 100),
            (700, 200, 0),
        ],
    }

    def __init__(self, enabled: bool = True) -> None:
        """
        Args:
            enabled: When *False* all sound is suppressed (useful for testing
                     in silent environments).
        """
        self.enabled = enabled
        self._active_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AlarmController":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Wait for any in-progress beep thread to finish."""
        if self._active_thread and self._active_thread.is_alive():
            self._active_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def beep_pass(self) -> None:
        """
        Emit a short, high-pitched beep indicating a PASS result.

        Pattern: 1 kHz, 100 ms.
        """
        self._print_status("PASS ✓", _GREEN)
        if self.enabled:
            self._active_thread = _run_in_thread(self._do_pass_beep)

    def alarm_ng(self, error_code: str) -> None:
        """
        Emit an error-specific alarm pattern.

        Args:
            error_code: One of the :class:`~src.core.models.ErrorCode` string
                values, e.g. ``"SIZE_MISMATCH"``.
        """
        self._print_status(f"FAIL ✗ [{error_code}]", _RED)
        if self.enabled:
            pattern = self._PATTERNS.get(
                error_code,
                [(440, 500, 0)],  # generic fallback: one 440 Hz beep
            )
            self._active_thread = _run_in_thread(self._play_pattern, pattern)

    def test_alarm(self) -> None:
        """
        Run every alarm pattern sequentially — use during station setup to
        verify speaker / buzzer wiring.
        """
        print(f"{_YELLOW}=== Alarm self-test ==={_RESET}")
        # Pass beep
        print("  PASS beep …")
        self._do_pass_beep()
        time.sleep(0.3)

        for code, pattern in self._PATTERNS.items():
            print(f"  {code} …")
            self._play_pattern(pattern)
            time.sleep(0.5)

        print(f"{_YELLOW}=== Self-test complete ==={_RESET}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _do_pass_beep() -> None:
        _beep(1000, 100)

    @staticmethod
    def _play_pattern(pattern: list[tuple[int, int, int]]) -> None:
        """Play a sequence of (freq, duration, gap) beep entries."""
        for freq, dur, gap in pattern:
            _beep(freq, dur)
            if gap:
                time.sleep(gap / 1000.0)

    @staticmethod
    def _print_status(label: str, colour: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] ALARM: {colour}{label}{_RESET}")


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with AlarmController() as ctrl:
        ctrl.test_alarm()
