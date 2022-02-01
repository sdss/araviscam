# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['araviscam']

package_data = \
{'': ['*']}

install_requires = \
['PyGObject>=3.48.0', 'astropy', 'math', 'numpy', 'sdss-basecam>=0.5.0']

entry_points = \
{'console_scripts': ['run = BlackflyCam:main']}

setup_kwargs = {
    'name': 'sdss-araviscam',
    'version': '0.0.342',
    'description': 'Blackfly S GigE camera reader for SDSS-V/LVM',
    'long_description': '# araviscam\nFLIR Blackfly S GigE camera reader for SDSS-V LVM telescope\n\n## Purpose\nA python class subclassed from [sdss/basecam](https://github.com/sdss/basecam) to read monochrome images of [FLIR](https://www.flir.com) Blackfly S GigE cameras.\nIt uses the [Aravis](https://github.com/AravisProject/aravis) C-library which is an interface to [GenICam](https://www.emva.org/standards-technology/genicam/) cameras. As such it might also be used to read a more general class of GenICam cameras.\n\nDevelopped for the guider of the Local Volume Mapper (LVM) of the 5th generation of the telescopes of the [Sloan Digital Sky survey](https://en.wikipedia.org/wiki/Sloan_Digital_Sky_Survey) (SDSS-V).\n\n## See Also\n\n* [this project](https://github.com/sdss/araviscam)\n* [baslerCam](https://github.com/sdss/baslercam)\n* [mantacam](https://github.com/sdss/mantacam)\n* [flicamera](https://github.com/sdss/flicamera)\n',
    'author': 'Richard J. Mathar',
    'author_email': 'mathar@mpia-hd.mpg.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://wiki.sdss.org',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.5',
}


setup(**setup_kwargs)
