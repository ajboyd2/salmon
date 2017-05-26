from setuptools import setup, find_packages

setup(
    name="salmon",
    version="0.2.0",

    description="A symbolic statistical modeling tool.",

    url="https://github.com/ajboyd2/salmon",

    author="Alex Boyd",
    author_email="ajboyd@calpoly.edu",

    license="GPLv3",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='modeling symbolic regression',

    packages=find_packages()
)
