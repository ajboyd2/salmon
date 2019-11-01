from setuptools import setup, find_packages

setup(
    name="salmon",
    packages = ["salmon"],
    version="1.0.0",

    description="A symbolic algebra based linear regression tool.",

    url="https://github.com/ajboyd2/salmon",
    download_url="https://github.com/ajboyd/salmon/archive/v_100.tar.gz",



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
        "pandas",
        "numpy",
        "unittest",
        "matplotlib",
        "scipy",
    ],
    packages=find_packages()
)
