import click
import os 
import sys 
import json 
sys.path.append(os.path.realpath("..")) # Add hippocampalseq to path

import hippocampalseq as hse
import hippocampalseq.utils as hseu

@click.command()
@click.option("--results-path", default="../results/")
@click.option("--rats", multiple=True, type=click.Choice(hseu.RAT_NAMES), default=hseu.RAT_NAMES)
def main(
        results_path,
        rats,
    ):
    with open(os.path.join(results_path, 'results.json'), 'r') as f:
        model_params = json.load(f)

    for rat in rats:
        for session in os.listdir(os.path.join(results_path, rat)):
            pass

if __name__ == '__main__':
    main()