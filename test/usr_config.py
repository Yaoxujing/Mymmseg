import os
import sys

path = 'Mymmseg'

os.chdir(path)

sys.path.append(os.getcwd())

print('current path' ,os.getcwd())
print('sys path:' ,sys.path)

from mmmcv.utils import Config

cfg = Config.fromfile('test/config0.py')

print(cfg)
print("done")