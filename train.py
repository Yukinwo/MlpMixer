import torch
from dataset import get_train_loader, get_val_loader
from mlp_model.mlp_mixer import MLP_Mixer
import torch.optim as optim
import os
from tqdm import tqdm
from torch.optim import lr_scheduler

# 总训练轮数
epochs = 10
# 初始学习率
lr = 0.1

# 定义模型对象
mlp_net = MLP_Mixer(image_size=60, patch_size=10, dim=512, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048, dropout=0.2)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6, 8], gamma=0.1)

# 获取数据加载器
train_loader = get_train_loader(batch_size=32)
val_loader = get_val_loader(batch_size=128)

# 打开训练日志文件
train_log_f = open('./log/train-loss.log', 'w')
# 打开验证日志文件
val_log_f = open('./log/val-loss.log', 'w')

# 计算模型参数量
params_total = sum([param.nelement() for param in mlp_net.parameters()]) / 1e6
print(params_total)
train_log_f.write('参数量 = {:.2f} M \n'.format(params_total))
val_log_f.write('参数量 = {:.2f} M \n'.format(params_total))

# 判断是否支持gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 在支持gpu的情况下将模型载入gpu
mlp_net.to(device)

# 判断是否存在模型保存路径
if not os.path.exists('./trained_model'):
    os.mkdir('./trained_model')
model_path = './trained_model'

# 最佳验证loss
best_val_loss = 999

# 训练模型
for epoch in range(epochs):
    loader = iter(train_loader)
    mlp_net.train()
    for step, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = mlp_net(inputs)
        print(outputs.shape, labels.shape)
        print(outputs)
        print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=mlp_net.parameters(), max_norm=1)
        optimizer.step()
        # 每5步打印一次训练loss
        if step % 5 == 0:
            loss_str = f'epoch: {epoch} step: {step} train_loss: {loss}  ' \
                       f'lr: {lr_scheduler.get_last_lr()} \n'
            train_log_f.write(loss_str)
            print(loss_str)
    # 每训练一轮计算一次验证集loss
    mlp_net.eval()
    v_loader = iter(val_loader)
    val_losses = []
    with torch.no_grad():
        for val_step, val_data in enumerate(tqdm(v_loader)):
            inputs, labels = val_data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mlp_net(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss)
    val_loss = torch.tensor(val_losses).mean().detach().item()
    val_loss_str = f'epoch: {epoch} val_loss: {val_loss}  lr: {lr_scheduler.get_last_lr()} \n'
    val_log_f.write(val_loss_str)
    print(val_loss_str)
    #调整学习率
    lr_scheduler.step()
    # 保存在验证集上loss最低的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(mlp_net.state_dict(), os.path.join(model_path, 'best_model.pth'))

# 最后再保存一次模型
torch.save(mlp_net.state_dict(), os.path.join(model_path, 'final_model.pth'))

# 关闭日志文件
train_log_f.close()
val_log_f.close()


