#%%
import cv2
from glob import glob
import pylab
import numpy as np
filenames = glob("./pokemon/*")
pokes = []
# for i in np.random.choice(filenames, 6):
for i in filenames:
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    img[img[..., -1] <= 0] = 0
    img = img[..., :3][..., ::-1]
    pokes.append(img)
    pylab.imshow(img)
    pylab.show()

for i, v in enumerate(pokes):
    if v.shape != (32, 32, 3):
        pokes[i] = cv2.resize(pokes[i], (32, 32))
pokes = np.array(pokes)


#%%
tf.reset_default_graph()

inp = tf.keras.layers.Input((32, 32, 3))
x = inp
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation=tf.nn.relu)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation=tf.nn.relu)(x)
x = tf.keras.layers.MaxPool2D()(x)
encoder = tf.keras.layers.Conv2D(16, 3, padding="same", activation=tf.nn.relu)(x)

x = encoder
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation=tf.nn.relu)(x)
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation=tf.nn.relu)(x)
decoder = tf.keras.layers.Conv2D(3, 3, padding="same", activation=tf.nn.sigmoid)(x)

loss = tf.losses.mean_squared_error(inp, decoder)
opt = tf.train.AdamOptimizer().minimize(loss)

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
train_indices = list(range(len(pokes)))
batch_size = 100
check = np.random.choice(train_indices, 17)
for epoch in range(1000):
    np.random.shuffle(train_indices)

    losses = []
    for batch in zip(*[iter(train_indices)]* batch_size):
        _, lv = sess.run([opt, loss], feed_dict={inp: pokes[batch,] / 255})
        losses.append(lv)

    # print(epoch, ":", np.mean(losses), np.max(losses), np.min(losses))
    if epoch % 10 == 0:
        d = sess.run(decoder, feed_dict={inp: pokes[check,] / 255})
        pylab.imshow(d[0])
        pylab.show()

#%%
class Shuffle:
    def __getitem__(self, s):
        self.s = s
        return self.shuffle

    def shuffle(self, array):
        array = list(array)
        bet = array[self.s]
        np.random.shuffle(bet)
        array[self.s] = bet
        return array
shuffle = Shuffle()

#%%
check = np.random.choice(train_indices, 17)
for i in range(10):
    check = np.random.choice(train_indices, 17)
    e = sess.run(encoder, feed_dict={inp: pokes[check,] / 255})
    # e[..., i] += np.random.normal(scale=1.0, size=e.shape[1:-1])
    # e[1] = (e[i] + e[1]) / 2

    sp = 4
    start = np.random.randint(16 - sp)
    index = shuffle[start : start + sp](range(16))
    # index1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 11, 14, 15]
    index2 = [0, 4, 2, 3, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # e = (e[..., index1] + e[..., index2] + e[..., index]) / 3
    # e = (e[..., index2] + e[..., index1]) / 2
    e = e[..., index2]
    p = 9
    for _ in range(1):
        pylab.subplot(2, 2, 1)
        pylab.imshow(e[p, ..., 1], cmap="gray")
        pylab.subplot(2, 2, 2)
        pylab.imshow(e[p, ..., 2], cmap="gray")
        pylab.subplot(2, 2, 3)
        pylab.imshow(e[p, ..., 3], cmap="gray")
        pylab.subplot(2, 2, 4)
        pylab.imshow(e[p, ..., 4], cmap="gray")
        pylab.show()

    d = sess.run(decoder, feed_dict={encoder: e})
    cv2.imwrite(f"ahiru/ahiru_{i}.png", d[1])
    pylab.title(index)
    pylab.imshow(d[1])
    pylab.show()
