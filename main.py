import os
import errno
import click
import numpy as np
from configs.ibsr import IBSRConfig
from src.data.preprocessor import Preprocessor
from src.models.unet import Unet
from src.data.utils import DataUtils

_config = IBSRConfig


@click.group()
def cli():
    
    # Initiate essential directories
    try:
        os.makedirs(_config.PROCESSED_TRAIN_DATA_DIR)
        os.makedirs(_config.PROCESSED_TEST_DATA_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@click.command('preprocess')
@click.option('--train-dir', 'raw_train_data_dir',
              type=click.Path(), default=_config.RAW_TRAIN_DIR)
@click.option('--test-dir', 'raw_test_data_dir',
              type=click.Path(), default=_config.RAW_TEST_DIR)
@click.option('--processed-train-dir', 'processed_train_dir',
              type=click.Path(), default=_config.PROCESSED_TRAIN_DATA_DIR)
@click.option('--processed-test-dir', 'processed_test_dir',
              type=click.Path(), default=_config.PROCESSED_TEST_DATA_DIR)
def process_data(raw_train_data_dir, raw_test_data_dir, processed_train_dir, processed_test_dir):

    preprocessor = Preprocessor(raw_train_data_dir=raw_train_data_dir,
                                raw_test_data_dir=raw_test_data_dir,
                                processed_train_data_dir=processed_train_dir,
                                processed_test_data_dir=processed_test_dir,
                                postfix_data_file=_config.POSTFIX_DATA_FILE,
                                postfix_mask_data_file=_config.POSTFIX_MASK_DATA_FILE, transpose=[1, 2, 0, 3])

    click.echo('Start processing data.')
    preprocessor.do_preprocess()
    click.echo('Data have been processed.')


@click.command()
@click.option('--weights-path', 'weights_path', type=click.Path(), default=_config.WEIGHTS_PATH)
@click.option('--image-width', 'image_width', type=click.INT)
@click.option('--image-height', 'image_height', type=click.INT)
def train(weights_path,image_width, image_height):
    data_utils = DataUtils(_config.PROCESSED_TRAIN_DATA_DIR, _config.PROCESSED_TEST_DATA_DIR)
    data, mask = data_utils.get_train_data()
    unet = Unet(image_width, image_height)
    unet.train(data, mask, weights_path, _config.EPOCHS)


@click.command()
@click.option('--weights-path', 'weights_path', type=click.Path(), default=_config.WEIGHTS_PATH)
@click.option('--data-path', 'data_path', type=click.Path())
@click.option('--predictions-path', 'predictions_path', type=click.Path())
def predict(weights_path, data_path, predictions_path):
    data = np.load(data_path)
    unet = Unet()
    predictions = unet.predict(data=data, weights_path=weights_path)
    np.save(predictions_path, predictions)


@click.command()
@click.option('--weights-path', 'weights_path', type=click.Path(), default=_config.WEIGHTS_PATH)
def evaluate(weights_path):
    data_utils = DataUtils(_config.PROCESSED_TRAIN_DATA_DIR, _config.PROCESSED_TEST_DATA_DIR)
    test_data, test_mask = data_utils.get_test_data()

    unet = Unet()
    score, acc = unet.evaluate(test_data, test_mask, weights_path)

    click.echo('Test score: {}'.format(score))
    click.echo('Test accuracy: {}'.format(acc))


# Add commands
cli.add_command(process_data)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == '__main__':
    cli()
