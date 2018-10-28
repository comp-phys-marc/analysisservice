import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


def normalize(array):
    return (array - array.mean()) / array.std()


def plot_machine_states(machine_digital_state, machine_analog_phase):
    plt.plot(machine_digital_state, machine_analog_phase, "bx")
    plt.ylabel("Phase")
    plt.xlabel("Memory")
    plt.savefig(sys.stdout.buffer)
    plt.show()


def plot_training_optimization(machine_digital_state,
                               machine_analog_phase,
                               train_machine_digital_state,
                               train_machine_analog_phase,
                               train_machine_digital_state_norm,
                               tf_machine_analog_phase_offset,
                               tf_size_factor,
                               fit_size_factor,
                               fit_plot_idx,
                               fit_machine_analog_phase_offsets,
                               sess):

    # get values used to normalized data so we can denormalize data back to its original scale
    train_machine_digital_state_mean = train_machine_digital_state.mean()
    train_machine_digital_state_std = train_machine_digital_state.std()

    train_machine_analog_phase_mean = train_machine_analog_phase.mean()
    train_machine_analog_phase_std = train_machine_analog_phase.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_machine_digital_state, train_machine_analog_phase, 'go', label='Training data')
    plt.plot(train_machine_digital_state_norm * train_machine_digital_state_std + train_machine_digital_state_mean,
             (sess.run(tf_size_factor) * train_machine_digital_state_norm + sess.run(
                 tf_machine_analog_phase_offset)) * train_machine_analog_phase_std + train_machine_analog_phase_mean,
             label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots()
    line, = ax.plot(machine_digital_state, machine_analog_phase)

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_machine_digital_state, train_machine_analog_phase, 'go', label='Training data')

    def animate(i):
        line.set_xdata(
            train_machine_digital_state_norm * train_machine_digital_state_std + train_machine_digital_state_mean)
        line.set_ydata((fit_size_factor[i] * train_machine_digital_state_norm + fit_machine_analog_phase_offsets[
            i]) * train_machine_analog_phase_std + train_machine_analog_phase_mean)
        return line,

    def initAnim():
        line.set_ydata(np.zeros(shape=machine_analog_phase.shape[0]))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                  interval=1000, blit=True)
    plt.savefig(sys.stdout.buffer)
    plt.show()


def train_regression_model(machine_digital_state, machine_analog_phase):

    plot_machine_states(machine_digital_state, machine_analog_phase)

    num_train_samples = len(machine_digital_state)

    train_machine_digital_state = np.asarray(machine_digital_state[:num_train_samples])
    train_machine_analog_phase = np.asanyarray(machine_analog_phase[:num_train_samples:])

    train_machine_digital_state_norm = normalize(train_machine_digital_state)
    train_machine_analog_phase_norm = normalize(train_machine_analog_phase)

    tf_machine_digital_state = tf.placeholder("float", name="machine_digital_state")
    tf_machine_analog_phase = tf.placeholder("float", name="price")

    tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
    tf_machine_analog_phase_offset = tf.Variable(np.random.randn(), name="machine_analog_phase_offset")

    # Non-Linear Regression
    tf_machine_analog_phase_pred = tf.add(tf.multiply(tf_size_factor,
                                                      tf_machine_digital_state), tf_machine_analog_phase_offset)

    # Loss Function
    tf_cost = tf.reduce_sum(tf.pow(tf_machine_analog_phase_pred - tf_machine_analog_phase, 2)) / (2 * num_train_samples)

    # Optimizer Learning Rate
    learning_rate = 0.1

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        display_every = 2
        num_training_iter = 50

        fit_num_plots = math.floor(num_training_iter / display_every)
        fit_size_factor = np.zeros(fit_num_plots)
        fit_machine_analog_phase_offsets = np.zeros(fit_num_plots)
        fit_plot_idx = 0

        for iteration in range(num_training_iter):

            for (x, y) in zip(train_machine_digital_state_norm, train_machine_analog_phase_norm):
                sess.run(optimizer, feed_dict={tf_machine_digital_state: x, tf_machine_analog_phase: y})

            if (iteration + 1) % display_every == 0:
                c = sess.run(tf_cost, feed_dict={tf_machine_digital_state: train_machine_digital_state_norm,
                                                 tf_machine_analog_phase: train_machine_analog_phase_norm})
                print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                      "size_factor=", sess.run(tf_size_factor), "machine_analog_phase_offset=",
                      sess.run(tf_machine_analog_phase_offset))
                fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
                fit_machine_analog_phase_offsets[fit_plot_idx] = sess.run(tf_machine_analog_phase_offset)
                fit_plot_idx = fit_plot_idx + 1

        print("Optimization Finished!")
        training_cost = sess.run(tf_cost, feed_dict={tf_machine_digital_state: train_machine_digital_state_norm,
                                                     tf_machine_analog_phase: train_machine_analog_phase_norm})
        print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "machine_analog_phase_offset=",
              sess.run(tf_machine_analog_phase_offset), '\n')

        plot_training_optimization(machine_digital_state,
                                   machine_analog_phase,
                                   train_machine_digital_state,
                                   train_machine_analog_phase,
                                   train_machine_digital_state_norm,
                                   tf_machine_analog_phase_offset,
                                   tf_size_factor,
                                   fit_size_factor,
                                   fit_plot_idx,
                                   fit_machine_analog_phase_offsets,
                                   sess)

