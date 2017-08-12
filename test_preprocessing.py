from src.preprocessor import Preprocessor


if __name__ == '__main__':
    raw_train_data_path = './raw_data/train-volume.tif'
    raw_label_data_path = './raw_data/train-labels.tif'
    preprocessed_train_path = './preprocessed_data/train_data.npy'
    preprocessed_label_path = './preprocessed_data/label_data.npy'

    preprocessor = Preprocessor(raw_train_data_path, raw_label_data_path, aug_size=3)

    print('Preprocessing data...')
    preprocessor.do_preprocess()
    print('Finish preprocess.')

    preprocessor.save(preprocessed_train_path, preprocessed_label_path)
