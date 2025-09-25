from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='mo_epgg',
    version='1.0.1',
    description='Multi-objective evolutionary population games gym environments and algorithms',
    author='Nicole Orzan',
    author_email='orzan.nicole@gmail.com',
    url='https://github.com/nicoleorzan/marl-mo',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)