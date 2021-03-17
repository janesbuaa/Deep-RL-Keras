import sys
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats

class DDQN:
    """ Deep Q-Learning Main Algorithm          深度Q学习主要算法
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization      初始化
        """
        # Environment and DDQN parameters       环境和DDQN参数
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.buffer_size = 20000
        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks      建立演员和评论家网络
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay   用于经验重播的内存缓冲区
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action          应用epsilon-greedy策略选择下一步操作
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer            从缓冲区采样的批次训练Q网络
        """
        # Sample experience from memory buffer (optionally with PER)    来自内存缓冲区的示例体验（可选配PER）
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN     在批次样本里应用Bellman方程来训练我们的DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree                   更新PER Sum树
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch                                批量训练
        self.agent.fit(s, q)
        # Decay epsilon                                 衰变epsilon
        self.epsilon *= self.epsilon_decay


    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm                DDQN主要训练算法
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode                             重设episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)                      演员选择动作（遵循政策）
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal     检索新状态，奖励以及该状态是否为终端
                new_state, r, done, _ = env.step(a)
                # Memorize for experience replay                                    保存经验重播
                self.memorize(old_state, a, r, done, new_state)
                # Update current state                                              更新当前状态
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network                 训练DDQN并将权重转移到目标网络
                if(self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            # Gather stats every episode for plotting                   收集每个情节的统计数据以进行绘图
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard                            为Tensorboard导出结果
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score             显示分数
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer                           将经验存储在内存缓冲区中
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(self.agent.predict(new_state))
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        if(self.with_per):
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
