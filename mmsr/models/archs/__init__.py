import importlib
import os.path as osp
import os
import mmcv

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in os.scandir(arch_folder)
    if v.is_file() and v.name.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'models.archs.{file_name}')
    for file_name in arch_filenames
]
