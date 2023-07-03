import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("./pre_trained_model")
embeddings = embed(["cat is on the floor", "dog is on the floor"])
print(embeddings)

sim = tf.keras.losses.cosine_similarity(embeddings[0], embeddings[1], axis=0)
print(sim)
print(sim.numpy())