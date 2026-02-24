import typer

app = typer.Typer(help="emb: Embed, index, and search text corpora", invoke_without_command=True)

@app.callback()
def main(ctx: typer.Context):
    """emb: Embed, index, and search text corpora."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

@app.command()
def version():
    """Show version."""
    from emb import __version__
    print(f"emb {__version__}")

if __name__ == "__main__":
    app()
