import itertools
import multiprocessing
import runpy
import sys
from os import path as osp



def run_main(*args):
    # patch sys.args
    sys.argv = list(args)
    print("Args", args)
    target = args[0]
    # run_path has one difference with invoking Python from command-line:
    # if the target is a file (rather than a directory), it does not add its
    # parent directory to sys.path. Thus, importing other modules from the
    # same directory is broken unless sys.path is patched here.
    if osp.isfile(target):
        sys.path.insert(0, osp.dirname(target))
    runpy.run_path(target, run_name="__main__")


def run_main_subproc(args):
    # This test needs to be done in its own process as there is a potentially for
    # an OpenGL context clash otherwise
    mp_ctx = multiprocessing.get_context("spawn")
    proc = mp_ctx.Process(target=run_main, args=args)
    proc.start()
    proc.join()
    assert proc.exitcode == 0
