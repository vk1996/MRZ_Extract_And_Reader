import tensorflow as tf
from glob import glob

files=sorted(glob('models/*.h5'))

quant=True

for modelpath in files:

  model=tf.keras.models.load_model(modelpath)
  # Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  if quant:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save the model.
  if quant:
    tflitepath=modelpath.replace('.h5','_quant.tflite')
  else:
    tflitepath=modelpath.replace('.h5','.tflite')
  with open(tflitepath, 'wb') as f:
    f.write(tflite_model)