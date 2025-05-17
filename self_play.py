import math
import time

import numpy
import ray
import torch

import models


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed) # 游戏的实例，比如breakout

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network 
        # 构建了相同的MuZeroNetwork，并且设置为评估模式
        # todo 感觉应该是用来采集样本的
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        '''
        shared_storage: ray引用对象的共享存储区，todo 感觉和模型权重有关系
        replay_buffer: ray引用对象的重放缓冲区
        test_mode: 是否是测试模式，默认是False，影响到action的选择
        '''

        # 从参数共享存储区对象获取当前训练步数 和 是否已经中断训练的标识
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):  
            # 更新模型缓存
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                # 训练模式，随机选择动作 visit_softmax_temperature_fn返回的是一个softmax的温度，标签平滑？todo
                # temperature_threshold感觉是温度阈值，不能小于的作用吧 todo
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                # 将一轮游戏的记录存储在重放缓冲区中
                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # 测试模型，从这里来看，test模式选择动作是最大最好的动作
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                # 保存测试过程中的游戏记录（分数、步数等）
                # 存储到checkpoints中
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    # 多人游戏
                    # 则分开记录不同玩家的分数
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            # 粗略的延迟等待训练进程上来，如果不行就加入更加精细的while循环
            # self_play_delay是一个超参数，表示的是自对弈的延迟时间
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            # 这里的self.config.ratio是一个超参数，表示的是自对弈和训练的比例
            # 具体看readme
            if not test_mode and self.config.ratio:
                # 这里的动作是保证训练和环境采集的平衡
                # 如果采集的步数小于训练的步数，则继续采集
                # 如果采集的步数大于训练的步数，则继续训练，采集进行sleep等待
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)
        
        # 采集结束，关闭游戏
        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.

        render：是否渲染采集样本时的画面
        opponent：对手，主要针对竞技游戏，类似下棋这种游戏的对手是p1 和 p2 切换的游戏
        """
        # 游戏样本存储缓冲区
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            # 没有结束、采集的样本小于max_moves（也就是游戏的步数）
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                # 这里校验观察的shape，必须是像素空间
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                # 获取历史帧信息（包含观察和动作）和当前环境观察
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action 计算动作的方式
                # self表示类似下棋的游戏，需要采用MCTS的方式，这样才能进行的下去
                # muzero_player == self.game.to_play判断的是当前的玩家是否和游戏的玩家一致，说明也是符合下棋类的角色切换游戏
                if opponent == "self" or muzero_player == self.game.to_play():
                    # 这里使用了MCTS 主要用于pvp的游戏
                    # 利用模型对环境的学习，模拟N步的执行，从中找到最好奖励的动作，选择进行执行（有点像贪心算法）
                    # 返回搜索的根节点和搜索树的信息
                    root, mcts_info = MCTS(self.config).run(
                        self.model, # 模型
                        stacked_observations, # 历史帧堆叠
                        self.game.legal_actions(), # 返回所有的合法动作
                        self.game.to_play(), # 当前的游戏玩家id
                        True, # 是否给动作增加噪音
                    )

                    # 根据搜索树得到下一个要执行的动作
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    # 打印搜索树的信息
                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    # 这边应该就是属于pve的游戏，就比较简单了
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                # 环境执行动作
                observation, reward, done = self.game.step(action)

                if render:
                    # 渲染游戏画面
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        # 返回一轮游戏的样本记录
        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        opponent: 对手类型
        "self":
        表示自对弈模式，即 MuZero 和自己对弈
        使用 MCTS (蒙特卡洛树搜索) 来选择动作
        主要用于类似围棋、象棋等双人对弈游戏
        "human":
        表示人类玩家作为对手
        MuZero 会给出动作建议，但最终动作由人类选择
        用于人机交互测试场景
        "expert":
        表示使用预定义的专家系统作为对手
        调用 game.expert_agent() 来获取动作
        用于与已有的强AI系统对比测试
        "random":
        表示使用随机策略的对手
        从合法动作中随机选择一个动作
        用于基准测试或简单评估
        """
        if opponent == "human":
            # 但是在这里hunman使用的是已有的模型进行模拟
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            # 这里需要依赖游戏本身支持生成最好的动作
            return self.game.expert_agent(), None
        elif opponent == "random":
            # 这里就很简答了，随机选择一个动作
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.

        node: 搜索树的起始节点
        temperature: 温度参数，控制动作选择的随机性
        """
        # 获取当前节点所有子节点被访问的次数
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        # 或者当前节点的所有子节点的动作
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            # 如果温度为0，则选择访问次数最多的动作
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            # 如果温度为无穷大，则随机选择一个动作
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            # 根据被访问的次数计算动作的概率分布选择动作
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.

        model, # 模型
        observation, # 历史帧堆叠和当前帧信息
        legal_actions # 返回所有的合法动作
        to_play # 当前的游戏玩家id
        add_exploration_noise  # 是否给动作增加噪音
        override_root_with：传入已构建的搜索树节点，避免重新构建搜索树，提高搜索效率
        """
        # 首先构建搜索树
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            # 构建搜索根节点
            root = Node(0)
            # 将观察帧迁移到对应的device上
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            # 将观察传给模型获取预测的动作、奖励、隐藏状态
            # 初始化模型的推理状态
            (
                root_predicted_value, # 价值
                reward, # 奖励
                policy_logits, # 动作的概率分布
                hidden_state, # 特征状态嵌入
            ) = model.initial_inference(observation)
        
            # 将价值分布（Q值分布）转换为标量值，也就是期望值
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            # 将奖励分布转换为标量值，也就是期望值
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            # 将预测的动作、奖励、隐藏状态传给根节点
            # 这里的hidden_state是一个tensor，表示的是当前的状态嵌入
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            # 给根节点每个动作添加噪音
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        # 这里的min_max_stats是一个类，主要是用来记录搜索树的最大值和最小值 todo
        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        # num_simulations 这里是一个超参数，表示的是模拟搜索的次数
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play # 当前的玩家
            node = root # 当前的节点
            search_path = [node] # 搜索路径 记录从当前node搜索到叶子节点的路径
            current_tree_depth = 0 # 记录当前的搜索深度

            # 如果当前的节点还有可以执行的动作，就继续搜索
            while node.expanded():
                current_tree_depth += 1 # 探索深度加1
                # 获取当前节点的下一个要执行的动作以及动作对应的子节点
                action, node = self.select_child(node, min_max_stats)
                # 将下一个节点添加到搜索路径中
                search_path.append(node)

                # Players play turn by turn
                # 动作执行完毕，切换
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            # 看mdn的注释，表示的是当前节点的父节点
            parent = search_path[-2]
            # action: 表示到达当前节点执行的动作
            # hidden_state: 表示当父节点的隐藏状态
            # 得到下一个状态的Q价值，执行当前动作得到的奖励，下一个状态的动作logits，下一个动作的Q价值
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            # 将Q价值分布和奖励分布转换为标量值，也就是期望值
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            # 更新链路的价值
            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            # 探索数的最大深度
            max_tree_depth = max(max_tree_depth, current_tree_depth)

        # 记录探索的最大深度和根节点的价值
        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        # root：表示搜索树的根节点
        # extra_info：表示搜索树的额外信息: 记录探索的最大深度和根节点的价值
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        node: 当前节点
        min_max_stats: 记录搜索树的最大值和最小值 todo

        returns: 选择的动作，获取对应动作的子节点
        """
        # for action, child in node.children.items(): 获取当前节点的所有子节点（也就是能够执行的动作）
        # 计算每个子节点的UCB分数，获取最大的UCB分数
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        # 从所有最大UCB分数中的动作随机选择一个
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        parent: 父节点
        child: 子节点
        min_max_stats: 记录搜索树的最大值和最小值 todo
        """
        # pb_c_base和pb_c_init是两个超参数，表示的是UCB的探索系数
        # 这里的pb_c是一个超参数，表示的是UCB的探索系数 todo 在哪里用到
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        # 计算探索强度
        '''
        基于父节点访问次数和子节点访问次数的比率
        与网络预测的先验概率相乘
        访问次数少的节点会获得更高的探索分数
        '''
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q 如果子节点探索过
            # 则使用子节点的价值来计算
            # child.reward: 子节点的奖励 todo 什么时候设置的？
            # self.config.discount：折扣系数
            # child.value(): 当前子节点的价值 针对单人游戏
            # -child.value()：如果是双人游戏，则使用负的子节点价值，因为如果下一步对手的更具备优势则扣除的优势更多，确实符合设计
            # normalize：将当前的奖励和折扣后的价值进行归一化
            # value_score: 子节点的价值
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            # 如果子节点没有探索过
            # 则子节点的的价值分数为0
            value_score = 0
        
        # 探索分数+价值分数=UCB分数
        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        该方法主要适用于更新每个节点的价值和访问次数
        更新整个链路中的最大价值和最小价值
        这样才能够保证每个节点的价值都是最新的，找到最具有价值的节点

        search_path: 搜索路径
        value: 价值，当前节点的Q价值
        to_play: 玩家的id
        min_max_stats: 记录搜索树的最大值和最小值 记录的是搜索树中的bellman方程计算得到的最大值和最小值
        会在这里更新搜索树的最大值和最小值
        """
        if len(self.config.players) == 1:
            # 只有一个玩家的情况
            # 搜索路径是从叶子节点到顶节点
            for node in reversed(search_path):
                node.value_sum += value # 这里统计的是当前节点后续所有的Q价值总和
                node.visit_count += 1 # 被访问的次数
                min_max_stats.update(node.reward + self.config.discount * node.value()) # 更新当前搜索树中的bellman方程计算得到的最大值和最小值

                # 这里的value时真正的bellman方程计算得到的价值，因为value就是当前节点的Q价值
                # 而上面的value和value_sum以及被访问的次数有关系，如果访问次数少，价值大则大
                # 如果访问次数多但是价值小则小
                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            # 两个玩家的游戏
            for node in reversed(search_path):
                # 这里计算value的时候要注意，如果上一个节点不是当前玩家则增加的价值为负号，理由
                # 就是如果对方的回合那么肯定是对方的价值增加了，而当前玩家的价值减少了
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                # 这里的node之所以是负数，是因为value通常指的是下一个状态的价值
                # 而下一个状态肯定是对方的状态，所以要取反
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                # 这里value之所以不是负数就是因为这个的value已经考虑到了负数
                # 而奖励才需要根据当前玩家的id来决定是正数还是负数，因为对方得到了分数
                # 那么对于另一个玩家就是不利的，需要扣除
                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0 # 当前节点的访问次数
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {} # 动作和对应的节点
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        # 这里表示当前节点是否有子节点，而自节点的子节点就是动作
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            # 如果访问的次数是0则价值为0
            return 0
        # 如果存在访问次数，则返回当前节点的价值
        # 这里的value_sum是当前节点的价值总和
        # visit_count是当前节点的访问次数
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        将当前节点计算出来的所有可能的动作创建子节点添加到当前节点中
        并得到当前节点获取的奖励和价值

        actions: 所有可能的动作
        to_play: 玩家的id
        reward：奖励的期望值
        policy_logits: 动作概率分布
        hidden_state: 观察特征提取潜入值
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        # 这里就是很明显的将动作和对应的概率分布进行映射
        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        # 为每个动作制作一个节点，用于后续的动作选择
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        # 或者当前节点的能够执行的所有动作
        actions = list(self.children.keys())
        # 这里的dirichlet_alpha是一个超参数，表示的是噪音的强度 todo
        # numpy.random.dirichlet([dirichlet_alpha] * len(actions))：生成一个dirichlet分布的噪音
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        # frac是一个加权超参数，用于控制噪音的强度
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            # todo 这里prior的作用？难道是影响到离散动作的概率分布吗
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    记录一个生命周期内的游戏执行的状态信息，连续存储
    """

    def __init__(self):
        self.observation_history = [] # 环境观察
        self.action_history = [] # 执行的动作
        self.reward_history = [] # 奖励
        self.to_play_history = [] # 记录玩家的id
        self.child_visits = [] # 记录一轮游戏中，每次状态的子节点的访问次数占比
        self.root_values = [] # 记录一轮游戏中的每次执行的动作的价值
        self.reanalysed_predicted_root_values = None # 记录一个样本周期内每个观察的预测价值总和，这里是使用最新的模型对整体的价值进行重新估计，提高更加准确的价值估计
        # For PER 训练数据的优先级
        self.priorities = None # todo
        self.game_priority = None # todo

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        # 存储游戏过程中的搜索树的统计信息
        # root：表示搜索树的根节点
        # action_space：表示所有的动作空间
        # 这里的action_space是一个离散动作空间
        if root is not None:
            # 统计所有子节点的访问次数
            sum_visits = sum(child.visit_count for child in root.children.values())
            # 统计所有动作的访问次数在所有动作中的占比，如果不在子节点中则为0
            # todo 作用
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            # 
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.

        index： 我看到有传入-1
        num_stacked_observations：需要堆叠的历史帧长度num_stacked_observations
        action_space_size：动作空间的维度
        """
        # Convert to positive index 
        # 如果传入-1，则表示获取缓冲区最后一个数据
        index = index % len(self.observation_history)
        
        # 获取最后一个观察obs
        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                # self.observation_history[past_observation_index： 获取对应索引的观察数据
                # self.action_history[past_observation_index + 1]：获取当前观察下执行的动作
                # 将观察和观察下执行的动作合并起来作为历史帧信息
                # / action_space_size是将动作值归一化道0/1之间，标准化
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                # 如果不足stacked_observations长度的帧信息则已0填充
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            # 将历史帧信息填充到缓冲区
            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        # 返回对接好的历史帧和当前帧信息 todo 查看这里的shape
        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
