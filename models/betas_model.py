from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dense, Flatten, MaxPool2D

from base_model import BaseModel


class BetasModel(BaseModel):
    def __init__(self, input_shape=(1024, 1024, 3), output_dims=10,
                 kernel_size=3, bn=True):
        super(BetasModel, self).__init__()
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        self.kernel_size = (kernel_size, kernel_size)
        self.bn = bn
        self.build_model()

    def build_model(self):
        x = Input(shape=self.input_shape, name='image')

        self.inputs.append(x)

        filters = 8

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPool2D()(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Downsampling
        d1 = conv2d(x, filters, bn=False)
        d2 = conv2d(d1, filters * 2)
        d3 = conv2d(d2, filters * 4)
        d4 = conv2d(d3, filters * 8)
        d5 = conv2d(d4, filters * 8)
        d6 = conv2d(d5, filters * 8)
        d7 = conv2d(d6, filters * 8)

        x = Flatten()(d7)
        x = Dense(self.output_dims, name='betas')(x)

        self.outputs.append(x)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)


if __name__ == "__main__":
    model = BetasModel(input_shape=(1024, 1024, 3))
    model.summary()
