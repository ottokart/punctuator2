from __future__ import print_function
import os

from setuptools import setup, find_packages

import punctuator

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(CURRENT_DIR, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except TypeError:
    with open(os.path.join(CURRENT_DIR, 'README.md')) as f:
        long_description = f.read()


def get_reqs(*fns):
    lst = []
    for fn in fns:
        for package in open(os.path.join(CURRENT_DIR, fn)).readlines():
            package = package.strip()
            if not package:
                continue
            lst.append(package.strip())
    return lst


setup(
    name="punctuator",
    version=punctuator.__version__,
    packages=find_packages(),
    author="Chris Spencer",
    author_email="chrisspen@gmail.com",
    description="Adds punctuation to a block of text.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/chrisspen/punctuator2",
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # 'Development Status :: 6 - Mature',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={'console_scripts': [
        'punctuator.py = punctuator.punc:command_line_runner',
    ]},
    zip_safe=False,
    install_requires=get_reqs('requirements.txt'),
    tests_require=get_reqs('requirements-test.txt'),
)
