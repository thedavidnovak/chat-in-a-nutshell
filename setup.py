from setuptools import setup, find_packages


setup(
    name='chat_in_a_nutshell',
    version='1.3.0',
    author='David Novák',
    description="A terminal-based script for interacting with OpenAI's API.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thedavidnovak/chat-in-a-nutshell',
    packages=find_packages(),
    py_modules=['chat_openai'],
    entry_points={
        'console_scripts': [
            'ch=chat_openai:main',
        ],
    },
    python_requires='>=3.7.1',
    install_requires=[
        'openai~=1.35.7'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)