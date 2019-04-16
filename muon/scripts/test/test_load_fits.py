
import click
from astropy.io import fits


@click.group(invoke_without_command=True)
@click.argument('fname')
def main(fname):
    f = fits.open(fname)
    print(f)

    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
