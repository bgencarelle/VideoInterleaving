"""Launch and manage the optional desktop video-player preview."""
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys

from tools.v7_capture import video_source_loops


def ffplay_command(source, loop, executable='ffplay'):
    """Build a muted ffplay command with the sender's repeat policy."""
    command = [str(executable), '-hide_banner', '-loglevel', 'error', '-an']
    if loop:
        command.extend(('-loop', '0'))
    command.append(os.path.expanduser(str(source)))
    return command


@dataclass
class VideoPreview:
    process: object = None
    warning: str = ''

    def close(self):
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        except OSError:
            pass


def _platform_open_command(source, platform, which):
    value = os.path.expanduser(str(source))
    if platform == 'darwin':
        return ['open', value]
    if platform.startswith('win'):
        return None
    opener = which('xdg-open')
    if opener:
        return [opener, value]
    gio = which('gio')
    if gio:
        return [gio, 'open', value]
    raise RuntimeError('No desktop video player or file opener is available.')


def launch_video_preview(source, live=None, *, platform=None, which=None,
                         popen=None, startfile=None):
    """Open a video in a desktop player without blocking sender startup.

    ffplay is preferred because its loop option can follow the sender's file
    policy and its preview audio can be disabled. The platform-associated app
    is the fallback; its loop and audio behavior are application-controlled.
    """
    platform = sys.platform if platform is None else platform
    which = shutil.which if which is None else which
    popen = subprocess.Popen if popen is None else popen
    source = os.path.expanduser(str(source))
    loop = video_source_loops(source, live)
    ffplay = which('ffplay')
    if ffplay:
        process = popen(
            ffplay_command(source, loop, ffplay),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **({} if platform.startswith('win') else {'start_new_session': True}))
        return VideoPreview(process=process)

    if platform.startswith('win'):
        startfile = startfile or getattr(os, 'startfile', None)
        if startfile is None:
            raise RuntimeError('No system video-player launcher is available.')
        startfile(source)
    else:
        command = _platform_open_command(source, platform, which)
        popen(command, stdin=subprocess.DEVNULL,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
              **({} if platform.startswith('win') else {'start_new_session': True}))
    warning = ('System video player opened; repeat and audio settings are '
               'controlled by that application.')
    return VideoPreview(warning=warning)
