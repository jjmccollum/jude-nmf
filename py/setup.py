from setuptools import setup, find_packages

setup(
    name='jude-nmf',
	version='1.0.0',
    packages=find_packages(),
    description='Non-negative matrix factorization applied to Wasserman\'s collation of the epistle of Jude.',
    author='Joey McCollum',
    license='MIT',
    author_email='jjmccollum@vt.edu',
    url='https://github.com/jjmccollum/jude-nmf',
	python_requires='>=3.5',
    install_requires=[
        'pandas',
        'sklearn',
        'scipy',
        'nimfa'
    ],
	classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
