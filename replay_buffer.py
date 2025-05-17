import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"] # 记录游戏的数量
        self.num_played_steps = initial_checkpoint["num_played_steps"] # 记录所有游戏记录的步数总和
        # total_samples 有多少个样本已经在 buffer 中，初始化时为0
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        '''
        game_histroy: 一轮游戏的所有数据
        '''
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                # todo 避免加载为只读数组
                # 已有优先级，则直接复制
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                # 没有优先级，则计算优先级
                # todo 后续再看
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1 # 记录游戏周期的数量
        self.num_played_steps += len(game_history.root_values) # 记录游戏的步数
        self.total_samples += len(game_history.root_values) # 记录所有游戏的样本数总和

        # 限制重放缓冲区的长度，限制为 replay_buffer_size
        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            # 更新已有的游戏周期个数以及总样本的个数
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        '''
        获取游戏训练的batch
        '''
        (
            index_batch, # 每个采样的游戏id和采样位置
            observation_batch, # 每个采样位置开始的观察帧堆叠
            action_batch, # 采样位置开始时到结束位置的执行动作list
            reward_batch, # 采样位置开始时到结束位置的获取奖励list
            value_batch, # 采样位置开始时到结束位置的动作价值list
            policy_batch, # 采样位置开始时到结束位置的每个位置的动作概率分布（依据访问次数确定）
            gradient_scale_batch, # # 计算梯度缩放因子
        ) = ([], [], [], [], [], [], [])
        # 这里的 weight_batch 是用来做优先级经验回放的
        weight_batch = [] if self.config.PER else None

        # game_id: 游戏的 id
        # game_history: 每个游戏周期的所有样本数据
        # game_prob: 游戏样本的优先级，对于没有优先级的游戏，值为 None
        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            # 对应游戏生命周期采样的位置，以及对应位置的优先级
            game_pos, pos_prob = self.sample_position(game_history)
            
            # 返回以game_pos为起点的连续样本数据
            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )
            
            # 记录该样本的游戏id、样本位置
            index_batch.append([game_id, game_pos])
            # 获取对应位置的且进行了历史帧堆叠的观察，这边就不是一个列表了，而是一个向量包含历史帧信息
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            # 取展开步数和剩余动作数的较小值
            # 对每个动作重复相同的缩放因子
            # todo 看看后续是什么使用的吧
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                # self.total_samples 总共的样本数量
                # game_prob：当前游戏的优先级
                # pos_prob：样本的优先级
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            # 归一化
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.


        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs) 
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        '''
        Buffer 结构

        self.num_played_games: 总共玩过的游戏数量
        len(self.buffer): 当前缓冲区中的游戏数量
        game_index: 在当前缓冲区中随机选择的索引
        ID 计算目的

        维持游戏 ID 的连续性
        考虑了缓冲区大小限制导致的旧游戏被删除的情况

        # 假设:
        num_played_games = 1000  # 总共玩了1000局
        buffer_size = 100       # 缓冲区限制为100局
        game_index = 50        # 随机选中第50局
        
        # 计算:
        game_id = 1000 - 100 + 50 = 950

        最早的900局已被删除
        缓冲区中保存了901-1000这100局
        选中的是缓冲区中第50个游戏，实际对应第950局
        '''
        game_id = self.num_played_games - len(self.buffer) + game_index

        # 游戏的id，指定游戏周期的所有数据，游戏的优先级
        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        '''
        n_games: 采样的游戏数量，传入的是btach_size,意思就是采集 batch_size 个游戏数据？
        '''
        if self.config.PER and not force_uniform:
            game_id_list = [] # 记录游戏的 id，一个游戏周期应该算一个id
            game_probs = [] # 记录每轮游戏的优先级
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            # 归一化每轮游戏的优先级
            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)
            # 游戏的 id 和优先级组成一个字典
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            # 随机选择 n_games 个游戏周期样本数据
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)
        else:
            # 没有优先级则随机选择 n_games 个游戏周期样本数据
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        # 返回选择的游戏训练周期数据的列表
        # 看起来应该就是一个游戏周期的连续训练数据
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        这个方法的作用 todo
        game_history: 游戏周期的所有数据
        force_uniform: 是否强制使用均匀采样

        return: 返回采集的样本位置和优先级
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            # 计算每个样本的优先级比例
            position_probs = game_history.priorities / sum(game_history.priorities)
            # 根据优先级比例随机选择一个样本位置
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            # 获取对应样本位置的优先级
            position_prob = position_probs[position_index]
        else:
            # 随机选择一个样本位置
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        # 更新置顶id的游戏周期的所有数据
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        '''
        game_history: 游戏周期的所有数据
        index: 采样的样本位置

        返回指定周期内的价值
        '''
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        # 计算计算value展开的步数
        bootstrap_index = index + self.config.td_steps
        # 确保采样的样本位置在游戏周期的范围内
        if bootstrap_index < len(game_history.root_values):
            # 获取根节点的值 todo 这里的reanalysed_predicted_root_values时什么？
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            # 计算最后一步的价值，如果是双人游戏则取反
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )
            # 利用n步dqn的计算方式，得到最后一步的折扣价值
            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            # 如果超出范围，则取最后一步的价值取0
            value = 0

        # 遍历指定周期内的奖励
        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            # 通过bellman方程计算价值整个指定周期内的价值
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * self.config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        game_history: 游戏周期的所有数据
        state_index: 采样的样本位置

        return: 返回以state_index为起点的 num_unroll_steps + 1 个样本的价值、奖励、访问次数比率和动作
        """
        # target_values：存储每个采样位置的价值
        # target_rewards：存储每个采样位置的奖励
        # target_policies：获取采样位置的每个动作的访问次数比率
        # action：存储每个采样位置的动作
        target_values, target_rewards, target_policies, actions = [], [], [], []
        # 从采样的位置开始，向后采样 num_unroll_steps + 1 个样本
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            # 计算指定采样位置的价值
            value = self.compute_target_value(game_history, current_index)

            # 没有超过游戏周期的范围
            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            # 达到了游戏周期的末尾
            elif current_index == len(game_history.root_values):
                # 达到了末尾则value为0
                target_values.append(0)
                # 这里的奖励按照实际的奖励来
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                # 如果达到了游戏周期的末尾，则访问次数均匀分布，即每个动作的访问次数相同，概率也相同概率值 = 1/动作空间大小
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                # 实际在末尾执行的动作
                actions.append(game_history.action_history[current_index])
            # 超过了游戏周期的末尾
            else:
                # States past the end of games are treated as absorbing states
                # 超过末尾则价值奖励均为0
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy # 如果达到了游戏周期的末尾，则访问次数均匀分布，即每个动作的访问次数相同，概率也相同概率值 = 1/动作空间大小
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                # 这里的动作随机选择，很显然超过的部分没有任何意义
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network 这里也有一个相同的模型 todo 作用是什么？
        # 设置为了 eval 模式，难道是类似 test？
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        '''
        类是用来更新回放缓冲区中的价值估计
        '''
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):  
            # 获取最新的模型权重
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            # 随机采集一个游戏周期的样本数据
            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                # 遍历每一个游戏周期的样本数据，将对应的观察进行堆叠
                observations = numpy.array(
                    [
                        game_history.get_stacked_observations(
                            i,
                            self.config.stacked_observations,
                            len(self.config.action_space),
                        )
                        for i in range(len(game_history.root_values))
                    ]
                )

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )

                # 进行模型的前向推理，获取每个样本的价值，并将向量调整为标量
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                # 将得到的一个样本周期内的所有样本的价值设置为这个样本周期的价值
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            # 记录多少个游戏周期被重新分析过
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
