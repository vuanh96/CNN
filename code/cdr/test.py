import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
# Write before session run
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # Or write inside session
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
writer.close()  # close the writer when you’re done using it

# python3 [yourprogram].py
# tensorboard --logdir="./graphs" --port 6006 # Hoặc bất kỳ port nào khác http://localhost:6006/