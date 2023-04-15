def cal_pr_index(labels, predicts):
    a_tp = 0
    a_fp = 0
    a_fn = 0
    o_tp = 0
    o_fp = 0
    o_fn = 0
    n_tp = 0
    n_fp = 0
    n_fn = 0
    for i in range(len(labels)):
        label = labels[i]
        predict = predicts[i]
        if label == 0:
            if predict == 0:
                a_tp += 1
            else:
                a_fn += 1
        else:
            if predict == 0:
                a_fp += 1
        if label == 1:
            if predict == 1:
                o_tp += 1
            else:
                o_fn += 1
        else:
            if predict == 1:
                o_fp += 1
        if label == 2:
            if predict == 2:
                n_tp += 1
            else:
                n_fn += 1
        else:
            if predict == 2:
                n_fp += 1
    return {'a_tp': a_tp, 'a_fp': a_fp, 'a_fn': a_fn, 'o_tp': o_tp, 'o_fp': o_fp, 'o_fn': o_fn, 'n_tp': n_tp, 'n_fp': n_fp, 'n_fn': n_fn}