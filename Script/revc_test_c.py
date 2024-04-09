#!/usr/bin/env python3
""" Tests for loading_data1.py """

from subprocess import getstatusoutput
import platform
import os

PRG = './loading_data_c.py'
RUN = f'python {PRG}' if platform.system() == 'Windows' else PRG

# --------------------------------------------------


def test_exists() -> None:
    """ Program exists """
    assert os.path.isfile(PRG)


def test_directory() -> None:
    """ Program exists """
    assert os.path.exists(os.getcwd())


def test_usage() -> None:
    """ Prints usage """
    for arg in ['-h', '--help']:
        rv_, out = getstatusoutput(f'{RUN} {arg}')
        assert rv_ == 0
        assert out.lower().startswith('usage:')
# --------------------------------------------------


def test_not_right() -> None:
    """ Trying first inputs to see if they fail or pass """
    files = ['not', 'here', 'eggs']
    for fi_ in files:
        retval = getstatusoutput(f'{RUN} {fi_}')
        assert retval != 0
# --------------------------------------------------
