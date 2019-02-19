from astropy.io import fits
import click
import code


@click.group(invoke_without_command=True)
@click.argument('fits_file')
def main(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data

        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
