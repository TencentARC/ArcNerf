# -*- coding: utf-8 -*-

import os
import os.path as osp
from subprocess import call

import cv2
import ffmpeg
import numpy as np


def read_video(path, bgr2rgb=False, downsample=1, max_count=None):
    """
    Read a video to image list

    Args:
        path: The path of video file.
        bgr2rgb: Whether to change the frame from bgr to rgb order
        downsample: downsample ratio. default is 1
        max_count: max number of frames to be extracted. None will extract all.
    Returns:
         a list of image.
    """
    assert osp.exists(path), 'No video file at {}'.format(path)

    img_list = []
    count = 0
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
        count += 1

        if max_count is not None and count >= max_count:
            break

    cap.release()

    # downsample
    img_list = img_list[::downsample]

    return img_list


def write_video(img_list, path, rgb2bgr=False, fps=30):
    """
    Write a video from image list

    Args:
        img_list: A list of img in np array. Should be in bgr order, if in rgb need to set rgb2bgr=True
        path: The path to write the video. .MP4 is tested now.
        rgb2bgr: Whether to change img from rgb order to bgr.
        fps: video fps to be saved.
    """
    assert len(img_list) > 0, 'No image to be recorded down'

    height, width = img_list[0].shape[0], img_list[0].shape[1]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in img_list:
        if rgb2bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    writer.release()


def extract_video(path, dst_folder, max_name_len=6, ext='.png', video_downsample=1, image_downsample=1, max_count=None):
    """Extract frames of a video to final folder. Frame will to write to xxxxx.jpg.

    Args:
        path: video path
        dst_folder: folder to write
        max_name_len: max len of name.
        video_downsample: video downsample factor
        image_downsample: if >1, will resize image_h and image_w by factor
        max_count: max number of frames to be extracted. None will extract all
        ext: .png or .jpg
    """
    assert osp.exists(path), 'No video file at {}'.format(path)

    idx = 0
    count = 0
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        # video downsample
        if idx % video_downsample != 0:
            idx += 1
            continue
        idx += 1

        name = str(count).zfill(max_name_len) + ext
        img_path = osp.join(dst_folder, name)
        h, w = img.shape[0], img.shape[1]

        if image_downsample > 1:
            img = cv2.resize(img, (int(w / image_downsample), int(h / image_downsample)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, img)

        count += 1
        if max_count is not None and count >= max_count:
            break

    cap.release()


def extract_video_ffmpeg(path, dst_folder, max_name_len=6, ext='.png', max_count=None):
    """Extract frames by ffmpeg directly. Faster than extract_video, but hard to handle all choice"""
    assert osp.exists(path), 'No video file at {}'.format(path)

    format_str = '%0{}d'.format(max_name_len)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path]
    command.extend(['{}/{}{}'.format(dst_folder, format_str, ext)])

    if max_count is not None:
        command.extend(['-vframes', max_count])

    call(command)


def get_video_metadata(path):
    """
    Get the metadata of a video.

    Args:
        path: video path

    Returns：
        - length: number of frames
        - width: frame width
        - height: frame height
        - fps: video fps
    """
    assert osp.exists(path), 'No video file at {}'.format(path)

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return length, width, height, fps


def ffmpeg_video_read(video_path, fps=None):
    """Video reader based on FFMPEG. Borrow from https://github.com/google/aistplusplus_api.
      This function supports setting fps for video reading. It is critical
      as AIST++ Dataset are constructed under exact 60 fps, while some of
      the AIST dance videos are not percisely 60 fps.

    Args:
        video_path: A video file.
        fps: Use specific fps for video reading. (optional)

    Returns:
        A `np.array` with the shape of [seq_len, height, width, 3]
    """
    assert os.path.exists(video_path), f'{video_path} does not exist!'
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise e('stdout: {}, stderr: {}'.format(e.stdout.decode('utf8'), e.stderr.decode('utf8')))
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    stream = ffmpeg.input(video_path)
    if fps:
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return out.copy()


def ffmpeg_video_write(data, video_path, fps=25):
    """Video writer based on FFMPEG. Borrow from https://github.com/google/aistplusplus_api.

    Args:
        data: A `np.array` with the shape of [seq_len, height, width, 3]
        video_path: A video file.
        fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
    _, height, width, _ = data.shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg.input('pipe:', framerate=fps, format='rawvideo', pix_fmt='rgb24',
                     s='{}x{}'.format(width,
                                      height)).output(video_path,
                                                      pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()


def is_video_ext(file):
    """Check whether a filename is an video file by checking extension."""
    return file.lower().endswith(('.mp4', '.avi', '.mov'))
