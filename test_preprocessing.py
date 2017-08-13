from configs.development import SimpleConfig
from src.data.preprocessor import Preprocessor

_config = SimpleConfig


if __name__ == '__main__':

    preprocessor = Preprocessor(
        _config.RAW_TRAIN_IMAGE_PATH,
        _config.RAW_TRAIN_LABEL_IMAGE_PATH,
        _config.RAW_TEST_IMAGE_PATH,
        _config.AUG_SIZE,
        _config.AUG_CONFIGS
    )

    print('Preprocessing data...')
    preprocessor.do_preprocess()
    print('Finish preprocess.')

    preprocessor.save(
        _config.PROCESSED_TRAIN_IMAGE_PATH,
        _config.PROCESSED_TRAIN_LABEL_PATH,
        _config.PROCESSED_TEST_PATH
    )
