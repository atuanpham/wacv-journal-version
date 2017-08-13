import click
import numpy as np
from configs.development import SimpleConfig
from src.data.preprocessor import Preprocessor
from src.models.unet import Unet

_config = SimpleConfig


@click.group()
def cli():
    pass


@click.command('preprocess')
@click.option('--train-path', 'train_image_path',
              type=click.Path(), default=_config.RAW_TRAIN_IMAGE_PATH)
@click.option('--label-path', 'train_label_path',
              type=click.Path(), default=_config.RAW_TRAIN_LABEL_IMAGE_PATH)
@click.option('--test-path', 'test_image_path',
              type=click.Path(), default=_config.RAW_TEST_IMAGE_PATH)
@click.option('--processed-train-path', 'processed_train_path',
              type=click.Path(), default=_config.PROCESSED_TRAIN_IMAGE_PATH)
@click.option('--processed-label-path', 'processed_train_label_path',
              type=click.Path(), default=_config.PROCESSED_TRAIN_LABEL_PATH)
@click.option('--processed-test-path', 'processed_test_path',
              type=click.Path(), default=_config.PROCESSED_TEST_PATH)
@click.option('--aug-size', 'aug_size', type=click.INT, default=_config.AUG_SIZE)
def process_data(train_image_path, train_label_path, test_image_path,
                 processed_train_path, processed_train_label_path, processed_test_path, aug_size):

    preprocessor = Preprocessor(train_image_path, train_label_path, test_image_path, aug_size, _config.AUG_CONFIGS)

    click.echo('Start processing data.')
    preprocessor.do_preprocess()
    click.echo('Data have been processed.')

    preprocessor.save(processed_train_path, processed_train_label_path, processed_test_path)
    click.echo('Processed data have been saved to: {}'.format(_config.PROCESSED_DATA_DIR))


@click.command()
@click.option('--processed-train-path', 'processed_train_path',
              type=click.Path(), default=_config.PROCESSED_TRAIN_IMAGE_PATH)
@click.option('--processed-train-label-path', 'processed_train_label_path',
              type=click.Path(), default=_config.PROCESSED_TRAIN_LABEL_PATH)
@click.option('--image-width', 'image_width', type=click.INT)
@click.option('--image-height', 'image_height', type=click.INT)
def train(processed_train_path, processed_train_label_path, image_width, image_height):

    try:
        train_data = np.load(processed_train_path)
        train_label_data = np.load(processed_train_label_path)
    except FileNotFoundError:
        click.echo('File not found!')
        return 1

    unet = Unet(image_width, image_height)
    unet.train(train_data, train_label_data)


# Add commands
cli.add_command(process_data)
cli.add_command(train)


if __name__ == '__main__':
    cli()
