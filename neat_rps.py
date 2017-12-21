
from __future__ import print_function
import os
import neat

from rock_paper_stuff import play_games
from rock_paper_stuff.player import RandomPlayer, SimplePlayer

from ann_player import ANNPlayer


OTHER_PLAYERS = [SimplePlayer("Bob"), SimplePlayer("Cathy")]


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    player = ANNPlayer(net, "ANN")

    # Play games, get avg rank
    results = play_games(50, OTHER_PLAYERS + [player])
    wins = sum([1 for x in results[player.name] if x == 1])
    return wins


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)

    # winner = p.run(eval_genomes, 300)
    pe = neat.ParallelEvaluator(8, eval_genome)
    winner = p.run(pe.evaluate, 300)

    # Show output of the most fit genome against training data.
    winning_net = neat.nn.FeedForwardNetwork.create(winner, config)
    winning_player = ANNPlayer(winning_net, "ANN")
    results = play_games(100, [winning_player] + OTHER_PLAYERS)
    print("Winner sample game results:")
    print(results[winning_player.name])
    print(sum([1 for x in results[winning_player.name] if x == 1]))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    run(config_path)
