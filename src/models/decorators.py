class UnetDecorator(object):

    @staticmethod
    def load_model(func):

        def wrapper(unet, *args, **kwargs):
            if unet.model is None:
                unet.model = unet._get_unet_model()
            return func(unet, *args, **kwargs)

        return wrapper

    @staticmethod
    def load_weights(func):

        def wrapper(unet, *args, **kwargs):
            if unet.model is None:
                raise ModelError('Make sure that model was be loaded before loading weights.')

            if unet.weights_path is None:
                raise ModelError('Weights path is not defined.')

            if unet.weights_loaded == False:
                unet.model.load_weights(unet.weights_path)
                unet.weights_loaded = True

            return func(unet, *args, **kwargs)

        return wrapper

