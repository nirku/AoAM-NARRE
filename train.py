def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch,batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,

        deep.dropout_keep_prob: dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return loss, accuracy, mae, u_a, i_a


def dev_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae, preds = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae, preds]


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)