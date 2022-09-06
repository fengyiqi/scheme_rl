import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

cov2_1 = nn.Conv2d(1, 1, 3, padding=1).double()
cov2_2 = nn.Conv2d(1, 1, 3, padding=1).double()
cov2_3 = nn.Conv2d(1, 1, 3, padding=1).double()
maxpool = nn.MaxPool2d(2, 2).double()

free_env.reset()
accumulated_reward = 0
for t in range(20):
    results = free_env.step(0, q_cq=[6, 1])
    vor = results[0]
    vor = torch.tensor(vor)
    vor = vor.unsqueeze(0).unsqueeze(0)
    # [batch_size, input_channels, input_height, input_width]
    cov2_vor_1 = F.relu(cov2_1(vor))
    max_vor_1 = maxpool(cov2_vor_1)

    cov2 = nn.Conv2d(1, 1, 3, padding=1).double()
    cov2_vor_2 = F.relu(cov2_2(max_vor_1))
    max_vor_2 = maxpool(cov2_vor_2)

    cov2 = nn.Conv2d(1, 1, 3, padding=1).double()
    cov2_vor_3 = F.relu(cov2_3(max_vor_2))
    max_vor_3 = maxpool(cov2_vor_3)

    # linear_4 = torch.flatten(max_vor_4, 1)
    # linear_4 = linear(linear_4)
    accumulated_reward += results[1]
    print(f"{format((t+1)*0.1, '.2f')} s, reward: {format(results[1], '.4f')}, accumulated reward: {format(accumulated_reward, '.4f')}")
    plt.figure(figsize=(32, 4), dpi=50)
    plt.subplot(161)
    plt.imshow(cov2_vor_1.squeeze().squeeze().detach().numpy())
    plt.colorbar()
    plt.subplot(162)
    plt.imshow(max_vor_1.squeeze().squeeze().detach().numpy())
    plt.colorbar()
    plt.subplot(163)
    plt.imshow(cov2_vor_2.squeeze().squeeze().detach().numpy())
    plt.colorbar()
    plt.subplot(164)
    plt.imshow(max_vor_2.squeeze().squeeze().detach().numpy())
    plt.colorbar()
    plt.subplot(165)
    plt.imshow(cov2_vor_3.squeeze().squeeze().detach().numpy())
    plt.colorbar()
    plt.subplot(166)
    plt.imshow(max_vor_3.squeeze().squeeze().detach().numpy())
    plt.colorbar()

    plt.show()