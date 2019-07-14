import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from .nn import energy_distance, discretized_mix_logistic_loss, sample_from_discretized_mix_logistic, adam_updates
from .model import model_spec
import time

SEED = 1
LOSS_FUNCTION = 'energy distance'
INIT_BATCH_SIZE = 16
BATCH_SIZE = 16
NUM_GPU = 1
CONDITIONAL = False
NUM_CLASS = 10
NUM_RESNET = 5
NUM_FILTER = 160
NUM_LOGISTIC_MIX = 10
RESNET_NON_LIN = 'concat_elu'
DROPOUT = 0.5
POLYAK_DECAY = 0.9995
SAVE_DIR = '.'
LEARNING_RATE = 0.0001
MAX_EPOCHS = 10



rng = np.random.RandomState(SEED)
tf.set_random_seed(SEED)

if LOSS_FUNCTION == 'energy distance':
    loss_func = energy_distance
elif LOSS_FUNCTION == 'mix logistic loss':
    loss_func = discretized_mix_logistic_loss
elif LOSS_FUNCTION == 'new function':
    loss_func = newfunc
else:
    raise Exception("Unknown loss function")


data_train, data_test = download_mnist()
obs_shape = data_train[0][0].shape
assert len(obs_shape) == 3


x_init = tf.placeholder(tf.float32, shape=(INIT_BATCH_SIZE,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape = (INIT_BATCH_SIZE,) + obs_shape) for i in range(NUM_GPU)]


if CONDITIONAL:
    y_init = tf.placeholder(tf.int32, shape=(INIT_BATCH_SIZE,))
    h_init = tf.one_hot(y_init, NUM_CLASS)
    y_sample = np.split(np.mod(np.arrange(BATCH_SIZE*NUM_GPU), NUM_CLASS), NUM_GPU)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), NUM_CLASS) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(BATCH_SIZE,)) for i in range(NUM_GPU)]
    hs = [tf.one_hot(ys[i], NUM_CLASS) for i in range(NUM_GPU)]

else:
    h_init = None
    h_sample =[None] * NUM_GPU
    hs = h_sample


model_opt = {
    "nr_resnet": NUM_RESNET,
    "nr_filter": NUM_FILTER,
    "nr_logistic_mix": NUM_LOGISTIC_MIX,
    "resnet_nonlinearity": RESNET_NON_LIN,
    "energy_distance": energy_distance
}
model = tf.make_template("model", model_spec)

init_pass = model(x_init, h_init, init=True, dropout_p=DROPOUT, **model_opt)

all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=POLYAK_DECAY)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

grads = []
loss_gen = []
loss_gen_test = []
new_x_gen = []
for i in range(NUM_GPU):
    with tf.device('/gpu:%d' % i):
        out = model(xs[i], hs[i], ema=None, dropout_p=DROPOUT, **model_opt)
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out))

        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

        out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out))

        out = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        if energy_distance:
            new_x_gen.append(out[0])
        else:
            new_x_gen.append(sample_from_discretized_mix_logistic(out, NUM_LOGISTIC_MIX))

tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,NUM_GPU):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    optimizer = tf.group(adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

bits_per_dim = loss_gen[0]/(NUM_GPU*np.log(2.)*np.prod(obs_shape)*BATCH_SIZE)
bits_per_dim_test = loss_gen_test[0]/(NUM_GPU*np.log(2.)*np.prod(obs_shape)*BATCH_SIZE)


def sample_from_model(sess):
    x_gen = [np.zeros((BATCH_SIZE,) + obs_shape, dtype=np.float32) for i in range(NUM_GPU)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(NUM_GPU)})
            for i in range(NUM_GPU):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, NUM_GPU)
        feed_dict = {xs[i]: x[i] for i in range(NUM_GPU)}
        if y is not None:
            y = np.split(y, NUM_GPU)
            feed_dict.update({ys[i]: y[i] for i in range(NUM_GPU)})
    return feed_dict

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
test_bpd = []
lr = LEARNING_RATE
with tf.Session() as sess:
    for epoch in range(MAX_EPOCHS):
        begin = time.time()

        # init
        if epoch == 0:
            train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            if args.load_params:
                ckpt_file = SAVE_DIR + '/params_' + data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            else:
                print('initializing the model...')
                sess.run(initializer)
                feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True)  # manually retrieve exactly init_batch_size examples
                sess.run(init_pass, feed_dict)
            print('starting training')

        # train for one epoch
        train_losses = []
        for d in train_data:
            feed_dict = make_feed_dict(d)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({ tf_lr: lr })
            l,_ = sess.run([bits_per_dim, optimizer], feed_dict)
            train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test data
        test_losses = []
        for d in test_data:
            feed_dict = make_feed_dict(d)
            l = sess.run(bits_per_dim_test, feed_dict)
            test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            # generate samples from the model
            sample_x = []
            for i in range(args.num_samples):
                sample_x.append(sample_from_model(sess))
            sample_x = np.concatenate(sample_x,axis=0)
            img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
            np.savez(os.path.join(args.save_dir,'%s_sample%d.npz' % (args.data_set, epoch)), sample_x)

            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))






