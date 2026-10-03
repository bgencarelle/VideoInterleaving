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


def launch_video_preview(source, live=None, *, platform=None, which=None,
                         popen=None):
    """Open a video in a desktop player without blocking sender startup.

    ffplay is used because its loop option can follow the sender's file policy
    and its preview audio can be disabled. If it is unavailable, do not launch
    an uncontrolled system player that may play audio.
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

    return VideoPreview(warning=(
        'Muted source preview unavailable: ffplay was not found, so no player '
        'was opened. Install ffplay to use the External player preview.'))
