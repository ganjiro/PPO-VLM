import argparse
import functools
import os
import random
import signal
import sys
import time
from distutils.util import strtobool
from typing import Optional

import torch.nn as nn
import torch.optim as optim
from torch import int8
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from envpool.vizdoom.registration import register_custom_folder

import torch
import numpy as np
import cv2
import envpool  # Assuming envpool is being used for the environment
import imageio

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import torch
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

from utils.ewc import EWC


def sp_module(current_module, init_module, shrink_factor, epsilon):
    use_device = next(current_module.parameters()).device
    init_params = list(init_module.to(use_device).parameters())
    for idx, current_param in enumerate(current_module.parameters()):
        current_param.data *= shrink_factor
        current_param.data += epsilon * init_params[idx].data


def shrink_perturb(model, factor=1e-6):
    new_model = Agent(None)
    sp_module(model, new_model, 1 - factor, factor)


def save_frames_as_gif(frames, filename="episode_recording.gif"):
    """z
    Saves a list of frames as a GIF file.

    Args:
        frames: list of numpy arrays representing frames
        filename: output filename
    """
    if len(frames) > 0:
        processed_frames = []
        for frame in frames:
            # Ensure frame is in HWC format (height, width, channels)
            if frame.shape[0] == 3:  # If in CHW format
                frame = np.transpose(frame, (1, 2, 0))

            # Ensure values are in uint8 range [0, 255]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            processed_frames.append(frame)
        imageio.mimsave(filename, processed_frames, fps=120)
        print(f"Saved recording to {filename}")
    else:
        print("No frames were recorded")


def log_matrix_with_heatmap(matrix, tasks, name, vmin=0, vmax=1, cmap='YlOrRd'):
    """
    Create and log a heatmap visualization of a matrix to wandb

    Args:
        matrix: numpy array containing the matrix data
        tasks: list of task names for axes labels
        name: name for the wandb log
        vmin: minimum value for color scaling
        vmax: maximum value for color scaling
        cmap: matplotlib colormap name
    """
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(matrix,
                xticklabels=tasks,
                yticklabels=tasks,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                annot=True,  # Show values in cells
                fmt='.2f',  # Format for cell values
                cbar_kws={'label': 'Score'})

    plt.title(f'{name} Heatmap')
    plt.xlabel('Target Task')
    plt.ylabel('Source Task')
    plt.tight_layout()

    # Log to wandb
    wandb.log({f"{name}_heatmap": wandb.Image(plt)})
    plt.close()


def test(model, test_envs, env_names, global_step, save_gif=False, trackmatrix=False):
    print(f"Testing - Global Steps: {global_step} Time {time.time() - start_time}")
    model.eval()

    with torch.no_grad():
        for i, test_env in enumerate(test_envs):

            next_obs, _ = test_env.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            episode_rewards = np.zeros(10)
            episode_len = np.zeros(10)

            sum_len = 0
            sum_reward = 0
            count_done = 0
            kills = 0

            frames = []
            ep0_end = False

            for _ in range(0, 1250):
                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                # Execute game step
                next_obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())

                next_done = torch.tensor(terminated | truncated).to(int8).numpy()
                if not ep0_end: frames.append(next_obs[0][-3:])

                episode_rewards += reward
                episode_len += 1

                if next_done[0]:
                    ep0_end = True

                count_done += sum(next_done)
                sum_reward += sum(episode_rewards[next_done == 1])
                kills += sum(info["KILLCOUNT_TOTAL"][next_done == 1])
                sum_len += sum(episode_len[next_done == 1])
                episode_rewards[next_done == 1] = 0
                episode_len[next_done == 1] = 0

                # Process next observation
                next_obs = torch.Tensor(next_obs).to(device)
                if count_done >= 10: break

            mean_return = sum_reward / count_done
            mean_len = sum_len / count_done
            kills = kills / count_done
            success = (kills - 3.5) / 26.5

            print(
                f"{env_names[i]} - global_step={global_step}, mean_episodic_return={mean_return:.2f}, mean_episodic_len={mean_len}, Kills={kills:.2f}, Success={success:.2f}")
            writer.add_scalar(f"{env_names[i]}/episode_len", mean_len, global_step)
            writer.add_scalar(f"{env_names[i]}/reward", mean_return, global_step)
            writer.add_scalar(f"{env_names[i]}/kills", kills, global_step)
            writer.add_scalar(f"{env_names[i]}/success", success, global_step)
            # try:
            if save_gif: save_frames_as_gif(frames=frames, filename=f"gifs/{env_names[i]}_{global_step}.gif")
            # except:
            #     print("Error saving image")

            if trackmatrix:
                results_matrix[current_task][i] = kills
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_vanilla",
                        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=9,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=99,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--offline", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-vizdoom",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="girolamomacaluso",
                        help="the entity (team) of wandb's project")

    parser.add_argument("--env-floder", type=str, default=os.getcwd() + '/run_and_gun',
                        help="Folder with custom maps")

    parser.add_argument("--s-p", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this use shrink and perturb")
    parser.add_argument("--ewc", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this use elastic weight consolidation")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=32,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--updates-per-env", type=int, default=500,
                        help="the number of steps to run in each environment")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4),  # Adjusted for RGB input
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.actor = nn.Linear(256, 12)
        self.critic = nn.Linear(256, 1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    tasks = ["Obstacles-v1", "Green-v1", "Resized-v1", "Monsters-v1", "Default-v1", "Red-v1", "Blue-v1", "Shadows-v1"]
    current_task = 0
    register_custom_folder(args.env_floder)

    infinite_ammo = False
    terminate_on_ammo_depletion = True
    initial_ammo = 100
    max_episode_steps = 1250

    batch_size = args.num_envs

    dict_envs = dict(zip(tasks, [None for _ in range(len(tasks))]))
    dict_test_envs = dict(zip(tasks, [None for _ in range(len(tasks))]))

    print("Creating main environments...")
    envs_cont = []
    for task in tasks:
        env = envpool.make(
            task,
            env_type="gymnasium",
            num_envs=args.num_envs,
            batch_size=batch_size,
            use_combined_action=True,
            max_episode_steps=max_episode_steps,
            infinite_ammo=infinite_ammo,
            terminate_on_ammo_depletion=terminate_on_ammo_depletion,
            initial_ammo=initial_ammo,
        )
        print(task)
        envs_cont.append(env)

    envs = envs_cont[current_task]
    print("Creating test environments...")
    test_envs = []
    for task in tasks:
        env = envpool.make(
            task,
            env_type="gymnasium",
            num_envs=10,
            use_combined_action=True,
            max_episode_steps=max_episode_steps,
            infinite_ammo=infinite_ammo,
            terminate_on_ammo_depletion=terminate_on_ammo_depletion,
            initial_ammo=initial_ammo,
        )
        print(task)
        test_envs.append(env)

    # Wandb setup
    if args.track:
        import wandb
        mode = "offline" if args.offline else "online"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            mode=mode,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print(f"First task! #{current_task + 1}: {tasks[current_task]}")

    agent = Agent(envs).to(device)
    if args.ewc:
        ewc = EWC(agent, ewc_lambda=250)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    results_matrix = np.zeros([len(test_envs), len(test_envs)])

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device).to(torch.int64)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device).to(torch.bool)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize environments
    envs.async_reset()
    # next_done = torch.zeros(batch_size).to(device)
    # Start training
    global_step = 0
    start_time = time.time()
    num_updates = args.updates_per_env * len(tasks)

    # Initialize reward normalization variables
    running_mean = 0.0
    running_variance = 0.0
    count = 1e-8  # Small initial value to prevent division by zero

    for update in range(0, num_updates):

        if (update % args.updates_per_env == 0) and update != 0:
            test_time = time.time()
            test(agent, test_envs, tasks, global_step, True, True)
            print(f"Tested! Time elaplesed {time.time() - test_time}")
            print()
            start_time = time.time()

            current_task += 1
            current_task %= len(tasks)

            envs = envs_cont[current_task]
            if args.ewc:
                ewc.update_task_weights(
                    task_id=tasks[current_task],
                    obs=torch.Tensor(next_obs).to(device)
                )

            envs.async_reset()
            # next_done = torch.zeros(batch_size).to(device)
            print(f"Next task! #{current_task + 1}: {tasks[current_task]}")

        if update % 20 == 0 and update % args.updates_per_env != 0:
            test_time = time.time()
            test(agent, test_envs, tasks, global_step, False)
            print(f"Tested! Time elaplesed {time.time() - test_time}")
            print()
            start_time = time.time()

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        episode_rewards = np.zeros(args.num_envs)
        episode_lenghts = np.zeros(args.num_envs)

        episode_step = np.zeros(args.num_envs)

        # Rollout
        for step in range(args.num_steps):
            global_step += batch_size

            # Receive state from environments
            next_obs, reward, term, trunc, info = envs.recv()
            env_ids = info["env_id"]

            # Store current observation
            obs[step][env_ids] = torch.Tensor(next_obs).to(device)

            # Get actions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs[step][env_ids])
                values[step][env_ids] = value.flatten()

            actions[step][env_ids] = action
            logprobs[step][env_ids] = logprob

            # Store rewards and dones
            rewards[step][env_ids] = torch.tensor(reward).to(device)
            dones[step][env_ids] = torch.tensor(term | trunc).to(device)

            # Send actions to environments
            envs.send(action.cpu().numpy(), env_ids)
            episode_step[env_ids] += 1

        # Advantage computation
        with torch.no_grad():

            next_value = agent.get_value(obs[-1]).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = ~ dones[-1]
                        nextvalues = next_value
                    else:
                        nextnonterminal = ~ dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = ~ dones[-1]
                        next_return = next_value
                    else:
                        nextnonterminal = ~ dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.ewc:
                    ewc_loss = ewc.compute_ewc_loss()
                    loss += ewc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.s_p:
                    shrink_perturb(agent)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Log training metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.ewc:
            writer.add_scalar("losses/ewc", ewc_loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    test(agent, test_envs, tasks, global_step, True, True)

    # columns = [f"Task {i}" for i in range(results_matrix.shape[0])]

    results_matrix_20 = (results_matrix - 3.5) / 16.5
    results_matrix_20 = np.clip(results_matrix_20, 0, 1)
    results_matrix_tough = (results_matrix - 3.5) / 26.5
    results_matrix_tough = np.clip(results_matrix_tough, 0, 1)

    table = wandb.Table(data=results_matrix.tolist(), rows=tasks, columns=tasks)
    table_paper = wandb.Table(data=results_matrix_20.tolist(), rows=tasks, columns=tasks)
    table_tough = wandb.Table(data=results_matrix_tough.tolist(), rows=tasks, columns=tasks)

    log_matrix_with_heatmap(results_matrix, tasks, "Raw Results",
                            vmin=np.min(results_matrix), vmax=np.max(results_matrix))
    log_matrix_with_heatmap(results_matrix_20, tasks, "Paper Results")
    log_matrix_with_heatmap(results_matrix_tough, tasks, "Tough Results")

    # Calculate metrics
    average_accuracy_20 = np.mean(results_matrix_20[-1])  # Mean accuracy of final row
    average_incremental_accuracy_20 = np.mean([results_matrix_20[i, :i + 1].mean() for i in range(len(tasks))])
    forgetting_20 = np.max(results_matrix_20, axis=0) - results_matrix_20[-1]
    average_forgetting_20 = forgetting_20[:-1].mean()
    forward_transfer_20 = np.mean([results_matrix_20[i - 1, i] for i in range(1, len(tasks))])
    backward_transfer_20 = np.mean([results_matrix_20[-1, i] - results_matrix_20[i, i] for i in range(len(tasks) - 1)])

    average_accuracy_tough = np.mean(results_matrix_tough[-1])  # Mean accuracy of final row
    average_incremental_accuracy_tough = np.mean([results_matrix_tough[i, :i + 1].mean() for i in range(len(tasks))])
    forgetting_tough = np.max(results_matrix_tough, axis=0) - results_matrix_tough[-1]
    average_forgetting_tough = forgetting_tough[:-1].mean()
    forward_transfer_tough = np.mean([results_matrix_tough[i - 1, i] for i in range(1, len(tasks))])
    backward_transfer_tough = np.mean(
        [results_matrix_tough[-1, i] - results_matrix_tough[i, i] for i in range(len(tasks) - 1)])

    wandb.log({"Kills": table})
    wandb.log({"Result Matrix- Paper": table_paper})
    wandb.log({"Result Matrix - tough": table_tough})
    wandb.log({
        "Paper/Average Accuracy": average_accuracy_20,
        "Paper/Average Incremental Accuracy": average_incremental_accuracy_20,
        "Paper/Forgetting": forgetting_20,
        "Paper/Average Forgetting": average_forgetting_20,
        "Paper/Forward Transfer": forward_transfer_20,
        "Paper/Backward Transfer": backward_transfer_20,
        "Tough/Average Accuracy": average_accuracy_tough,
        "Tough/Average Incremental Accuracy": average_incremental_accuracy_tough,
        "Tough/Forgetting": forgetting_tough,
        "Tough/Average Forgetting": average_forgetting_tough,
        "Tough/Forward Transfer": forward_transfer_tough,
        "Tough/Backward Transfer": backward_transfer_tough,
    })

    envs.close()
    writer.close()
    for test_env in test_envs:
        test_env.close()

    torch.save(agent, f"models/{args.exp_name}.pth")
