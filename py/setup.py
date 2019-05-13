from distutils.core import setup

setup(
    name='jude-nmf',
    packages=[],
    description='Non-negative matrix factorization applied to Wasserman\'s collation of the epistle of Jude.',
    author='Joey McCollum',
    license='MIT',
    author_email='jjmccollum@vt.edu',
    url='https://github.com/jjmccollum/jude-nmf',
    requires=[
        'pandas',
        'sklearn',
        'scipy',
        'nimfa'
    ]
)
