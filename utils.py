import tensorflow as tf
import tensorflow_hub as hub


def similarity_value(s1, s2):
    embed = hub.load("./pre_trained_model")
    embeddings = embed([s1, s2])
    sim = tf.keras.losses.cosine_similarity(
        embeddings[0], embeddings[1], axis=0)

    print(sim.numpy())

    if sim.numpy() <= -0.6:
        msg = f"Cosine Similarity is {sim.numpy()}, this indicates high similarity"
    elif sim.numpy() >= 0.6:
        msg = f"Cosine Similarity is {sim.numpy()}, this indicates high dissimilarity"
    else:
        msg = f"Cosine Similarity is {sim.numpy()}, can't decide much with this value"
    return msg