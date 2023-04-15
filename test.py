import torch
from mlp_model.mlp_mixer import MLP_Mixer
from dataset import get_test_loader
from tqdm import tqdm
from utils.test_fun import cal_pr_index

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型对象
mlp_net = MLP_Mixer(image_size=60, patch_size=10, dim=512, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048, dropout=0.2)
mlp_net.to(device)


# 加载模型
mlp_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

# 验证数据集名称
test_names = ['nt3', 'nt4', 'mt', 'ngf', 'cntf', 'ln']

with torch.no_grad():
    mlp_net.eval()
    for name in test_names:
        tps = 0
        fns = 0
        test_loader = get_test_loader(batch_size=128, test_name=name)
        for batch in tqdm(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mlp_net(inputs)
            predicts = torch.argmax(outputs, dim=1)
            tp, fn = cal_pr_index(predicts)
            tps += tp
            fns += fn
        print(name)
        print('tps = ', tps)
        print('fns = ', fns)
        print('recall = ', tps / (tps + fns))
