"""
Setup configuration for browser-service package
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='browser-service',
    version='1.0.0',  # Update version in browser_service/__init__.py too
    author='monkscode',  # Update with your name
    author_email='1dhruvilvyas@gmail.com',  # Update with your email
    description='Browser automation service with AI-powered element identification and locator generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/monkscode/browser-service',  # Update with your repo URL
    project_urls={
        'Bug Reports': 'https://github.com/monkscode/browser-service/issues',
        'Source': 'https://github.com/monkscode/browser-service',
        'Documentation': 'https://github.com/monkscode/browser-service#readme',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Internet :: WWW/HTTP :: Browsers',
        'License :: OSI Approved :: MIT License',  # Update if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-asyncio>=0.21',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
            'isort>=5.12',
        ],
        'test': [
            'pytest>=7.0',
            'pytest-asyncio>=0.21',
            'pytest-cov>=4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Add if you want CLI commands
            # 'browser-service=browser_service.cli:main',
        ],
    },
    include_package_data=True,
    keywords='browser automation testing locators selenium playwright robot-framework ai',
    zip_safe=False,
)
