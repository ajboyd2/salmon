from setuptools import setup, find_packages

setup(
    name="salmon-lm",
    version="1.2",

    description="A symbolic algebra based linear regression tool.",

    url="https://github.com/ajboyd2/salmon",
    download_url="https://github.com/ajboyd2/salmon/archive/v_100.tar.gz",

    author="Alex Boyd",
    author_email="alexjb@uci.edu",

    license="GPLv3",

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='modeling symbolic regression',
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.25.0",
        "matplotlib>=3.7.1",
        "scipy>=1.10.1",
        "ordered-set>=4.1.0",
    ],
    packages=find_packages(exclude=("test",)),
)
