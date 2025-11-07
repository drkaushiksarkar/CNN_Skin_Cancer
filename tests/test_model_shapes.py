import tensorflow as tf
from cnn_skin_cancer.model import build_model, compile_model

def test_model_output_shape():
    model = build_model(num_classes=9, img_height=180, img_width=180, dropout=0.1)
    model = compile_model(model, {"name":"RMSprop","lr":1e-4})
    x = tf.random.uniform((2,180,180,3))
    y = model(x)
    assert y.shape == (2,9)
    s = tf.reduce_sum(y, axis=1).numpy()
    assert (abs(s - 1.0) < 1e-5).all()
