import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("cpu")


def cnn_net_version_vgglike_full(n_input_channels):
    return nn.Sequential(
        # 64*64
        nn.Conv2d(n_input_channels, 16, (3, 3), padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, (3, 3), padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 32*32
        nn.Conv2d(16, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 16*16
        nn.Conv2d(32, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 8*8
        nn.Conv2d(64, 128, (3, 3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 4*4
        nn.Flatten(),
    )

def cnn_net_version_vgglike_full_rti(n_input_channels):
    return nn.Sequential(
        # transfer 64 * 256 to 64*64
        nn.Conv2d(n_input_channels, 8, (3, 3), padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 4, (3, 3), padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d((4, 1)),
        # 64*64
        nn.Conv2d(4, 16, (3, 3), padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, (3, 3), padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 32*32
        nn.Conv2d(16, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 16*16
        nn.Conv2d(32, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 8*8
        nn.Conv2d(64, 128, (3, 3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 4*4
        nn.Flatten(),
    )


def cnn_net_version_vgglike_3d(n_input_channels):
    return nn.Sequential(
        # 16*16*16
        nn.Conv3d(n_input_channels, 16, (3, 3, 3), padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(),
        nn.Conv3d(16, 16, (3, 3, 3), padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(),
        nn.MaxPool3d(2),
        # 8 * 8 * 8
        nn.Conv3d(16, 32, (3, 3, 3), padding=1),
        nn.BatchNorm3d(32),
        nn.ReLU(),
        nn.Conv3d(32, 32, (3, 3, 3), padding=1),
        nn.BatchNorm3d(32),
        nn.ReLU(),
        # nn.MaxPool3d(2),
        # 8*8*8
        nn.Conv3d(32, 64, (3, 3, 3), padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.Conv3d(64, 64, (3, 3, 3), padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.MaxPool3d(2),
        # 4*4*4
        nn.Conv3d(64, 128, (3, 3, 3), padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.Conv3d(128, 128, (3, 3, 3), padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.MaxPool3d(2),
        # 2*2*2
        nn.Flatten(),
    )


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = cnn_net_version_vgglike_full(n_input_channels)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU()

        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        y = self.linear(self.cnn(observations))
        return y

class CustomCNNRTI(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNNRTI, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = cnn_net_version_vgglike_full_rti(n_input_channels)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        print(n_flatten)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU()

        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        y = self.linear(self.cnn(observations))
        return y


class CustomCNN3D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN3D, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print(n_input_channels)
        self.cnn = cnn_net_version_vgglike_3d(n_input_channels)

        #         Compute shape by doing one forward pass
        print(torch.as_tensor(observation_space.sample()[None]).shape)
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        print(n_flatten)
        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, 4096),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(4096),
        #     nn.Linear(4096, features_dim),
        #     nn.ReLU()
        #
        # )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.BatchNorm1d(4096),
            nn.Linear(1024, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU()

        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        y = self.linear(self.cnn(observations))
        return y


class CNNNet(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        #         self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1)
        #         self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(2, 2, 3, padding=1)
        #         self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 64, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(states, device=device, dtype=torch.float64)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)

        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
