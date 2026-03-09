from keras.models import load_model

model = load_model("asl_model_final.keras")

model.save("asl_model_tf.h5")

print("Model converted successfully")