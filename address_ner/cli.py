# -*- coding: utf-8 -*-
import os
import click
import logging

from address_ner import train
from address_ner import data
from address_ner import pull_data
from address_ner import eval


@click.group()
def cli():
    """
    main address-ner model
    """
    pass


cli.command("pull")(pull_data.run)
cli.command("data")(data.run)
cli.command("train")(train.run)
cli.command("eval")(eval.run)

if __name__ == "__main__":
    cli()
