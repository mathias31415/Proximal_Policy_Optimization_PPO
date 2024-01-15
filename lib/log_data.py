# Description: This file contains the function to log all metrics to tensorboard.
import tensorflow as tf

# define all metrics to log
def log_metrics(summary_writer, epoch, terminations, total_timesteps, critic_loss, actor_loss, avg_epoch_return, actor_learning_rate, critic_learning_rate, epsilon, gamma):
    with summary_writer.as_default():
        tf.summary.scalar('epoch', epoch, step=epoch)
        tf.summary.scalar('terminations per epoch', terminations, step=epoch)
        tf.summary.scalar('total_timesteps', total_timesteps, step=epoch)
        tf.summary.scalar('critic_loss', critic_loss, step=epoch)
        tf.summary.scalar('actor_loss', actor_loss, step=epoch)
        tf.summary.scalar('avg_epoch_return', avg_epoch_return, step=epoch)
        tf.summary.scalar('critic_learning_rate', critic_learning_rate, step=epoch)
        tf.summary.scalar('actor_learning_rate', actor_learning_rate, step=epoch)
        tf.summary.scalar('discount_factor_gamma', gamma, step=epoch)
        tf.summary.scalar('clip_range_epsilon', epsilon, step=epoch)