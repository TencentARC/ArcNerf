# -*- coding: utf-8 -*-

import os
import os.path as osp
import shutil
from shutil import copyfile, copytree, ignore_patterns


def remove_if_exists(file):
    """Remove file if it exists"""
    if os.path.exists(file):
        os.remove(file)


def remove_dir_if_exists(folder):
    """Remove directory with all files if it exists"""
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)


def copy_files(src_dir, dst_dir, subdir_name=None, file_names=None, ignore=ignore_patterns('*DS_Store', '.gitignore')):
    """Copy files in dir or by specified """
    os.makedirs(dst_dir, exist_ok=True)

    if subdir_name is None and file_names is None:
        copytree(src_dir, dst_dir, ignore=ignore)
    elif file_names is None:
        copytree(osp.join(src_dir, subdir_name), osp.join(dst_dir, subdir_name), ignore=ignore)
    elif subdir_name is None:
        if not isinstance(file_names, list):
            file_names = [file_names]
        for file_name in file_names:
            copyfile(osp.join(src_dir, file_name), osp.join(dst_dir, file_name))
    else:
        os.makedirs(osp.join(dst_dir, subdir_name), exist_ok=True)
        if not isinstance(file_names, list):
            file_names = [file_names]
        for file_name in file_names:
            copyfile(osp.join(src_dir, subdir_name, file_name), osp.join(dst_dir, subdir_name, file_name))


def replace_file(filename, src_words, dst_words, line_idx=None):
    """Replace src_words by dst_words in file.
     If line is set, only do it in that line. Line can be an num or a list
    """
    if not isinstance(src_words, list):
        src_words = [src_words]
    if not isinstance(dst_words, list):
        dst_words = [dst_words]
    if not isinstance(line_idx, list) and line_idx is not None:
        line_idx = [line_idx]
    assert len(src_words) == len(dst_words), 'Word pairs should have same number...'

    with open(filename, 'r') as f:
        text = f.readlines()

    with open(filename, 'w') as f:
        for idx, line in enumerate(text):
            if line_idx is None or (idx in line_idx):
                for src, dst in zip(src_words, dst_words):
                    line = line.replace(src, dst)
            f.write(line)


def scan_dir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
