#  install required lib
import tensorflow as tf
import tf2onnx
import onnx
import keras

# defining ctcLayer to load the .h5 model
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

loaded_model = tf.keras.models.load_model('/bestfcn.h5',custom_objects={'CTCLayer': CTCLayer}) # calling custom layer to load the model
final_model = tf.keras.models.Model(
    loaded_model.get_layer(name="image").input,loaded_model.get_layer(name="dense2").output) 

# onnx conversion
onnx_model,_ = tf2onnx.convert.from_keras(final_model,opset=11) # change the opset
onnx.save(onnx_model,'fcn.onnx')
print('successfully converted..')
