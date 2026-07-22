"""
Unit tests for browser_service.utils.logging_setup — setup_logging.

Purpose: setup_logging is called before browser-use is imported so every child
         logger inherits UTF-8 handling. It clears the root logger's handlers
         and, on Windows, rebinds sys.stdout/sys.stderr. Both are process-wide
         side effects, so a regression here either silences logs or crashes on
         the first emoji in a log line.

Isolation: setup_logging must never see pytest's real streams or the real root
           logger — see isolated_logging below for why, and why the swap has to
           happen inside the test body rather than in a fixture. The log file
           path is redirected to tmp_path so no test writes into the repo's
           logs/ directory.

Tests:
  - Returns the root logger when no name is given, a named logger otherwise
  - Applies the requested level to the root logger
  - Replaces pre-existing root handlers rather than appending to them
  - Attaches a rotating file handler with the module's size/backup settings
  - A file handler that cannot be created is skipped, not fatal
  - Windows branch sets UTF-8 env vars and wraps stdout/stderr
  - Windows branch does not double-wrap an already-wrapped stream
  - POSIX branch leaves streams and environment alone
"""

import io
import logging
import os
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

import pytest

from browser_service.utils import logging_setup
from browser_service.utils.logging_setup import (
    BROWSER_USE_LOG_BACKUP_COUNT,
    BROWSER_USE_LOG_MAX_BYTES,
    setup_logging,
)


class FakeStream(io.TextIOWrapper):
    """A stdout/stderr stand-in owning a real, disposable byte buffer."""

    def __init__(self):
        self.raw_bytes = io.BytesIO()
        super().__init__(self.raw_bytes, encoding="utf-8", errors="replace")

    def written(self) -> str:
        self.flush()
        return self.raw_bytes.getvalue().decode("utf-8", errors="replace")


class _Isolated:
    """Handles for assertions inside an isolated_logging block."""

    def __init__(self, root, stdout, stderr):
        self.root = root
        self.stdout = stdout
        self.stderr = stderr


@contextmanager
def isolated_logging(platform="win32"):
    """Run setup_logging against a stand-in root logger and disposable streams.

    This is a context manager used *inside* test bodies, not a fixture, and that
    is deliberate. Under the default fd-level capture, pytest reassigns
    sys.stdout when it resumes capturing for the call phase — so a stream swap
    performed during fixture setup is silently reverted before the test runs.
    setup_logging would then wrap pytest's own tmpfile in a TextIOWrapper, and
    when that wrapper is garbage-collected it closes the tmpfile, killing the
    whole session with "I/O operation on closed file" at teardown. Swapping here
    happens after capture has resumed, so the swap actually holds.

    The root logger is a stand-in for a second reason: setup_logging calls
    root_logger.handlers.clear(), which on the real root would tear out pytest's
    LogCaptureHandler and the handler browser_service/__init__.py bound at
    import time.

    `platform` is explicit rather than inherited from the host so both the
    Windows and POSIX branches are exercised on every machine and in CI.
    """
    root = logging.Logger("isolated-root-stand-in")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    saved_env = {k: os.environ.get(k) for k in ("PYTHONIOENCODING", "PYTHONUTF8")}
    fake_out, fake_err = FakeStream(), FakeStream()

    def fake_get_logger(name=None):
        return root if name is None else logging.Logger(name)

    try:
        sys.stdout, sys.stderr = fake_out, fake_err
        with patch.object(sys, "platform", platform):
            with patch.object(logging, "getLogger", side_effect=fake_get_logger):
                yield _Isolated(root, fake_out, fake_err)
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            # Release the rotating log before tmp_path is torn down, or it keeps
            # a Windows file lock on a directory pytest is about to delete.
            if isinstance(handler, RotatingFileHandler):
                handler.close()
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def log_file(tmp_path):
    """Redirect the module's log path into tmp_path."""
    path = tmp_path / "logs" / "browser_use.log"
    with patch.object(logging_setup, "BROWSER_USE_LOG_FILE", str(path)):
        yield path


class TestReturnedLogger:
    """Tests for which logger comes back."""

    def test_returns_root_logger_by_default(self, log_file):
        """No logger_name means the caller gets the root logger itself."""
        with isolated_logging() as iso:
            assert setup_logging() is iso.root

    def test_returns_named_logger(self, log_file):
        """A logger_name returns that specific logger, not the root."""
        with isolated_logging() as iso:
            result = setup_logging(logger_name="browser_service.test_target")
            assert result.name == "browser_service.test_target"
            assert result is not iso.root

    def test_applies_requested_level(self, log_file):
        """The level lands on the root logger so children inherit it."""
        with isolated_logging() as iso:
            setup_logging(log_level=logging.DEBUG)
            assert iso.root.level == logging.DEBUG


class TestHandlerWiring:
    """Tests for the handlers attached to the root logger."""

    def test_replaces_existing_handlers(self, log_file):
        """Pre-existing handlers are cleared, not appended to."""
        with isolated_logging() as iso:
            sentinel = logging.NullHandler()
            iso.root.addHandler(sentinel)

            setup_logging()

            assert sentinel not in iso.root.handlers

    def test_attaches_rotating_file_handler(self, log_file):
        """The rotating handler uses the module's size and backup settings."""
        with isolated_logging() as iso:
            setup_logging()

            rotating = [h for h in iso.root.handlers if isinstance(h, RotatingFileHandler)]
            assert len(rotating) == 1
            assert rotating[0].maxBytes == BROWSER_USE_LOG_MAX_BYTES
            assert rotating[0].backupCount == BROWSER_USE_LOG_BACKUP_COUNT

    def test_creates_log_directory(self, log_file):
        """The parent directory is created rather than assumed to exist."""
        with isolated_logging():
            setup_logging()
        assert log_file.parent.is_dir()

    def test_file_handler_failure_is_not_fatal(self, log_file):
        """An unwritable log path degrades to console-only instead of raising."""
        with isolated_logging() as iso:
            with patch.object(logging_setup, "os") as fake_os:
                fake_os.makedirs.side_effect = OSError("read-only filesystem")
                fake_os.path.dirname.return_value = str(log_file.parent)
                logger = setup_logging()

            assert logger is iso.root
            assert [h for h in iso.root.handlers if isinstance(h, RotatingFileHandler)] == []
            assert "Could not initialize browser_use.log file handler" in iso.stderr.written()

    def test_stream_handler_always_present(self, log_file):
        """Console logging survives alongside the file handler."""
        with isolated_logging() as iso:
            setup_logging()
            assert any(
                isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
                for h in iso.root.handlers
            )


class TestWindowsBranch:
    """Tests for the Windows-only UTF-8 reconfiguration."""

    def test_sets_utf8_environment(self, log_file):
        """Both UTF-8 env vars are forced on the Windows path."""
        with isolated_logging(platform="win32"):
            setup_logging()
            assert os.environ["PYTHONIOENCODING"] == "utf-8"
            assert os.environ["PYTHONUTF8"] == "1"

    def test_wraps_unwrapped_streams(self, log_file):
        """A raw stream with a .buffer gets a UTF-8 TextIOWrapper."""

        class RawStream:
            def __init__(self):
                self.buffer = io.BytesIO()

        with isolated_logging(platform="win32"):
            sys.stdout, sys.stderr = RawStream(), RawStream()

            setup_logging()

            assert isinstance(sys.stdout, io.TextIOWrapper)
            assert isinstance(sys.stderr, io.TextIOWrapper)
            assert sys.stdout.encoding == "utf-8"
            assert sys.stdout.errors == "replace"

    def test_does_not_rewrap_textiowrapper(self, log_file):
        """An already-wrapped stream is left alone — no wrapper-on-wrapper."""
        with isolated_logging(platform="win32") as iso:
            already = iso.stdout

            setup_logging()

            assert sys.stdout is already

    def test_stream_reconfiguration_failure_propagates(self, log_file):
        """A detached buffer is not survivable, and the test records that.

        The try/except in setup_logging guards only the wrapping block; the
        handler-stream assignment further down reads .buffer again outside it.
        Asserting the real behaviour rather than the intended one — if that
        second read is ever guarded too, this test fails and says so.
        """

        class ExplodingStream:
            @property
            def buffer(self):
                raise ValueError("underlying buffer has been detached")

        with isolated_logging(platform="win32"):
            sys.stdout, sys.stderr = ExplodingStream(), ExplodingStream()

            with pytest.raises(ValueError, match="detached"):
                setup_logging()


class TestPosixBranch:
    """Tests for the non-Windows path."""

    def test_streams_and_environment_untouched(self, log_file):
        """On POSIX the streams are left as-is and no UTF-8 env is forced."""
        with isolated_logging(platform="linux") as iso:
            os.environ.pop("PYTHONUTF8", None)

            setup_logging()

            assert sys.stdout is iso.stdout
            assert "PYTHONUTF8" not in os.environ

    def test_uses_plain_stream_handler(self, log_file):
        """The POSIX branch attaches a StreamHandler bound to stdout as-is."""
        with isolated_logging(platform="linux") as iso:
            setup_logging()

            stream_handlers = [
                h
                for h in iso.root.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, RotatingFileHandler)
            ]
            assert len(stream_handlers) == 1
            assert stream_handlers[0].stream is iso.stdout
