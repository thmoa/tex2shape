from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Dropout, LeakyReLU, BatchNormalization

from models.base_model import BaseModel


class Tex2ShapeModel(BaseModel):
    def __init__(self, input_shape=(512, 512, 3), output_dims=6,
                 kernel_size=3, dropout_rate=0, bn=True, final_layer=None):
        super(Tex2ShapeModel, self).__init__()
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        self.kernel_size = (kernel_size, kernel_size)
        self.dropout_rate = dropout_rate
        self.bn = bn
        self.final_layer = final_layer
        self.build_model()

    def build_model(self):
        x = Input(shape=self.input_shape, name='image')

        self.inputs.append(x)

        x = self._unet_core(x)
        x = Conv2D(self.output_dims, self.kernel_size, padding='same', name='output')(x)

        if self.final_layer:
            x = self.final_layer(x)

        self.outputs.append(x)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _unet_core(self, d0):

        filters = 64

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if self.dropout_rate:
                u = Dropout(self.dropout_rate)(u)
            if self.bn:
                u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Downsampling
        d1 = conv2d(d0, filters, bn=False)
        d2 = conv2d(d1, filters * 2)
        d3 = conv2d(d2, filters * 4)
        d4 = conv2d(d3, filters * 8)
        d5 = conv2d(d4, filters * 8)
        d6 = conv2d(d5, filters * 8)
        d7 = conv2d(d6, filters * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, filters * 8)
        u2 = deconv2d(u1, d5, filters * 8)
        u3 = deconv2d(u2, d4, filters * 8)
        u4 = deconv2d(u3, d3, filters * 4)
        u5 = deconv2d(u4, d2, filters * 2)
        u6 = deconv2d(u5, d1, filters)

        u7 = UpSampling2D(size=2)(u6)

        return u7


if __name__ == "__main__":
    model = Tex2ShapeModel()
    model.summary()
