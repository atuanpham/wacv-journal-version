import click
import numpy as np
from configs.ibsr import IBSRConfig
from src.data.preprocessor import Preprocessor
from src.models.unet import Unet

_config = IBSRConfig


@click.group()
def cli():
    pass


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
                                postfix_mask_data_file=_config.POSTFIX_MASK_DATA_FILE)

    click.echo('Start processing data.')
    preprocessor.do_preprocess()
    click.echo('Data have been processed.')


# @click.command()
# @click.option('--processed-train-path', 'processed_train_path',
#               type=click.Path(), default=_config.PROCESSED_TRAIN_IMAGE_PATH)
# @click.option('--processed-train-label-path', 'processed_train_label_path',
#               type=click.Path(), default=_config.PROCESSED_TRAIN_LABEL_PATH)
# @click.option('--weights-path', 'weights_path', type=click.Path(), default=_config.WEIGHTS_PATH)
# @click.option('--image-width', 'image_width', type=click.INT)
# @click.option('--image-height', 'image_height', type=click.INT)
# def train(processed_train_path, processed_train_label_path, weights_path,image_width, image_height):
# 
#     try:
#         train_data = np.load(processed_train_path)
#         train_label_data = np.load(processed_train_label_path)
#     except FileNotFoundError:
#         click.echo('File not found!')
#         return 1
# 
#     unet = Unet(image_width, image_height)
#     unet.train(train_data, train_label_data, weights_path, _config.EPOCHS)


# Add commands
cli.add_command(process_data)
# cli.add_command(train)


if __name__ == '__main__':
    cli()
