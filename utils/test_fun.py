def cal_pr_index(predicts):
    tp = 0
    fn = 0
    for i in range(len(predicts)):
        predict = predicts[i]
        if predict == 2:
            tp += 1
        else:
            fn += 1
    return tp, fn