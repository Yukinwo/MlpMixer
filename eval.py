import torch
from mlp_model.mlp_mixer import MLP_Mixer
from dataset import get_val_loader
from tqdm import tqdm
from utils.eval_fun import cal_pr_index

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型对象
mlp_net = MLP_Mixer(image_size=60, patch_size=10, dim=512, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048, dropout=0.2)

mlp_net.to(device)

# 加载模型
mlp_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

# 验证数据加载器
val_loader = get_val_loader(batch_size=128)

# 验证集总样本数量
val_count = 0
# 预测正确的总样本数量
true_count = 0
# 用于计算三种细胞的pr曲线的指标
pr_params = {'a_tp': 0, 'a_fp': 0, 'a_fn': 0, 'o_tp': 0, 'o_fp': 0, 'o_fn': 0, 'n_tp': 0, 'n_fp': 0, 'n_fn': 0}

with torch.no_grad():
    mlp_net.eval()
    for batch in tqdm(val_loader):
        inputs, labels, files = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mlp_net(inputs)
        predicts = torch.argmax(outputs, dim=1)
        # 计算总准确率
        batch_true_count = (labels == predicts).sum().item()
        true_count += batch_true_count
        val_count += len(labels)
        # 计算当前批次的pr曲线指标
        pr_param = cal_pr_index(labels, predicts)
        for key, value in pr_param.items():
            pr_params[key] += value
print(val_count)
print(true_count)
print('总准确率= ', true_count / val_count)
print(pr_params)
