from setuptools import setup, find_packages


setup(
    name='chat_in_a_nutshell',
    version='1.6.0',
    author='David Novák',
    description="A terminal-based script for interacting with OpenAI's and Anthropic's API.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thedavidnovak/chat-in-a-nutshell',
    packages=find_packages(),
    py_modules=['chat_provider'],
    entry_points={
        'console_scripts': [
            'ch=chat_provider:main',
        ],
    },
    python_requires='>=3.11',
    install_requires=[
        'httpx~=0.28.1',
        'openai~=1.62.0',
        'anthropic~=0.49.0',
        'mcp~=1.9.0',
        'aiofiles~=24.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-mock>=3.10.0',
            'flake8>=5.0.0',
            'flake8-quotes>=3.3.0',
            'autopep8>=2.0.0',
            'black>=23.0.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)