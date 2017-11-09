from distutils.core import setup

setup(
    name='simple_rl',
    packages=['simple_rl', 'simple_rl.utils', 'simple_rl.mdp', 'simple_rl.mdp.oomdp', \
              'simple_rl.planning', 'simple_rl.agents', 'simple_rl.agents.bandits', 'simple_rl.agents.func_approx',
              'simple_rl.experiments', 'simple_rl.tasks', \
              'simple_rl.tasks.chain', 'simple_rl.tasks.random', 'simple_rl.tasks.grid_world',
              'simple_rl.tasks.four_room', \
              'simple_rl.tasks.taxi', 'simple_rl.mdp.markov_game', 'simple_rl.tasks.gym', \
              'simple_rl.tasks.grid_game', 'simple_rl.tasks.rock_paper_scissors', 'simple_rl.tasks.prisoners', \
              'simple_rl.abstraction', 'simple_rl.abstraction.action_abs', 'simple_rl.abstraction.state_abs'],
    scripts=['simple_rl/run_experiments.py'],
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    version='0.76',
    description='A simple framework for experimenting with Reinforcement Learning in Python 2.7',
    author='David Abel',
    author_email='david_abel@brown.edu',
    url='https://github.com/david-abel/simple_rl',
    download_url='https://github.com/david-abel/simple_rl/tarball/v0.76',
    keywords=['Markov Decision Process', 'MDP', 'Reinforcement Learning'],
    classifiers=[],
)
