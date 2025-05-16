import math
from abc import ABC, abstractmethod

import torch


class MuZeroNetwork:
    '''
    这个应该是公共的特征提取层
    '''
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape, # 观察空间shape
                config.stacked_observations, # todo
                len(config.action_space), # 动作空间大小
                config.encoding_size, # 输出的特征编码的大小
                config.fc_reward_layers, # todo
                config.fc_value_layers, # todo
                config.fc_policy_layers, # todo
                config.fc_representation_layers, # 中间隐藏层的大小
                config.fc_dynamics_layers, # todo
                config.support_size, # todo
            )
        elif config.network == "resnet":
            # 对于图像数据，使用resnet来提取特征
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        '''
        获取网络的权重，并将权重转移到CPU上
        '''
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    '''
    全链接特征提取层
    '''
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        # todo 根据后文件，感觉是预测均值和方差吧？
        self.full_support_size = 2 * support_size + 1

        '''
        observation_shape[0]
        * observation_shape[1]
        * observation_shape[2] = 这里的应该是观察空间展平后的大小

        * (stacked_observations + 1)： 不懂是什么
        stacked_observations * observation_shape[1] * observation_shape[2]：这个应该是保留的前32帧的图像信息

        # 这里构建的应该是一个全连接的特征提取层，并且使用了DataParallel来支持多GPU训练
        '''
        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        # 这里继续上一步的特征提取层，集合了动作空间，构建对应的特征全链接提取层
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )

        # 这里是奖励的全链接提取层
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        # 构建预测的动作
        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        # 构建预测状态的价值
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        '''
        根据提取的环境特征预测动作和动作的价值
        '''
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        '''
        observation：环境观察

        return 返回提取的特征嵌入 并归一化
        '''
        # 提取观察特征
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        '''
        observation：环境观察
        初始化推理，应该时最开始的时候模型预测的动作、价值、奖励，环境特征嵌入
        '''
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        '''
        创建初始奖励向量，将中间位置设为1，其他位置为0
        这是为了保持一致性，因为初始状态没有真实的奖励值
        '''
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    '''
    这里的卷积核大小是3x3，步长是1，填充是1，那么输入和输出的大小是一样的
    '''
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        # 尺寸不变化，通道数减半
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )

        # 尺寸减半，通道数翻倍回到out_channels
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        # 尺寸不变化，通道数也不变化
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        # 尺寸减半，通道数不变化
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 尺寸不变，通道数不变化
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        # 尺寸减半，通道数不变化
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    '''
    普通的CNN特征提取层
    '''
    def __init__(self, in_channels, out_channels, h_w):
        '''
        h_w: 目标下采样后的大小 这里得注意一下得和代码中的尺寸一致，否则会出错
        '''
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2 # 计算中间通道数
        # 第一层卷积核大小是h_w[0] * 2，步长是4，填充是2，大概率原始的代码中这里的尺寸会减半
        # 第二层卷积核大小是5，步长是1，填充是2：尺寸不变化
        # 第三层卷积核大小是3，步长是2，填充是1：尺寸减半
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels, # 卷积层采输出的通道数
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            # 如果有设置下采样，那么就选择使用哪种下采样的方法
            # observation_shape[0]：表示单帧图像的通道数
            # observation_shape[0] * (stacked_observations + 1)：表示所有堆叠的帧的总通道数，其中 + 1表示当前帧
            # + stacked_observations : 为每个历史帧添加额外的信息通道 todo
            # 以下两个特征提取层最后
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                # DownsampleCNN 多传入了一个目标下采样后的大小尺寸
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        # 这里应该要设置else吧，毕竟有downsample时，forward中会有不同的处理
        # 如果不进行下采样，那么就直接使用卷积层恒等特征提取，可能针对比较简答的游戏
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        # 最后进行一次resnet，通道数不变，大小不变
        # todo 了解这边的尺寸变化
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        '''
        num_blocks: 残差块的数量
        num_channels：resnet输出的卷积通道数
        reduced_channels_value：输出的价值通道数
        reduced_channels_policy：输出的动作通道数
        block_output_size_value：卷积后输入到全链接层的展平后的大小
        block_output_size_policy：卷积后输入到全链接层的展平后的大小
        fc_value_layers：全链接层的层数
        fc_policy_layers：全链接层的层数
        action_space_size：动作的输出维度
        full_support_size：价值的输出维度
        '''
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        '''
        observation_shape,      # 观察空间的形状，例如(3, 96, 96)表示96x96的RGB图像
        stacked_observations,   # 堆叠的观察帧数，保存历史状态信息
        action_space_size,     # 动作空间大小，即可选动作数量
        num_blocks,            # ResNet中残差块的数量
        num_channels,          # 卷积层的通道数
        reduced_channels_reward,  # 奖励预测头的通道数
        reduced_channels_value,   # 价值预测头的通道数
        reduced_channels_policy,  # 策略预测头的通道数
        fc_reward_layers,      # 奖励预测全连接层的配置
        fc_value_layers,       # 价值预测全连接层的配置
        fc_policy_layers,      # 策略预测全连接层的配置
        support_size,          # 输出分布的支持范围大小 todo
        downsample,           # 下采样方法，可选"resnet"或"CNN"或者None或者false不进行下采样
        '''
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        # downsample todo 这里可以不选择下采样？
        # math.ceil(observation_shape[1] / 16) 应该是下采样16倍后的尺寸，所以这里初始的图像大小能被16整除
        # reduced_channels_reward 表示下采样后奖励的特征通道数
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        ## reduced_channels_value 表示下采样后价值的特征通道数
        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        # reduced_channels_policy 表示下采样后动作策略的特征通道数
        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        # 构建Resnet特征提取层
        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        # todo 这里的作用
        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        # todo 这里的作用 看代码，预测的是动作和价值
        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state):
        '''
        注意，这里输出的value时一个多维的数据，而不是一个标量值
        '''
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        '''
        observation：环境观察

        这里的作用是提取环境的特征，并将特征值归一化
        '''
        # 提取obs的特征
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        # encoded_state.view 进行内部feature map展平，shape变成(batch_size, channels, w * h = S)
        # min_encoded_state 获取每个通道中的最小值的特征
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1], # 应该是通道数
                encoded_state.shape[2] * encoded_state.shape[3], # 这里应该是尺寸面积，展平？
            )
            .min(2, keepdim=True)[0] # 类似torch.max 找到维度为2的最小值，并返回，比如encoded_state.view 后的 shape 为 [32, 256, 36]  (batch_size, channels, H*W) -> 输出 shape: [32, 256, 1]
            .unsqueeze(-1) # 在增加维度，最后的shape=[32, 256, 1, 1]
        )
        # 这里是获取特征中每个通道的最大值
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        # 特征最大值和最小值差值
        scale_encoded_state = max_encoded_state - min_encoded_state
        # 如果特征差值过小则对应的通道增加1e-5
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        # 归一化特征
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        '''
        observation：环境观察
        初始化推理，应该时最开始的时候模型预测的动作、价值、奖励，环境特征嵌入
        '''
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        '''
        创建初始奖励向量，将中间位置设为1，其他位置为0
        这是为了保持一致性，因为初始状态没有真实的奖励值
        '''
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    '''
    input_size: 输入层大小
    layer_sizes: 隐藏层大小列表
    output_size: 输出层大小
    output_activation: 输出层激活函数
    activation: 隐藏层激活函数
    '''

    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    # 构建MLP层
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar 将预测的分布转换为概率分布
    probabilities = torch.softmax(logits, dim=1)
    '''
    torch.tensor([x for x in range(-support_size, support_size + 1)])
    假设 support_size = 300
    生成范围为 [-300, 300] 的向量
    输出形状: [601]

    .expand(probabilities.shape)
    假设 probabilities.shape = [32, 601]（批量大小为32）
    将支持向量扩展到与概率分布相同的形状
    输出形状: [32, 601]
    '''
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )

    # 有点像C51的强化学习，计算概率分布的期望值
    # todo 调试这里看维度变化
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    # 使用特定公式进行反向变换，恢复原始数值范围
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
