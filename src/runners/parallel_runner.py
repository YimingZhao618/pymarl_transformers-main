from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
from runners.episode_runner import EpisodeRunner
import numpy as np
import torch as th
import copy
import traceback
import json
import hashlib

from .frozen_qmix_selector import FrozenQMIXSelector


class RecoverableEnvError(RuntimeError):
    """Discard the incomplete vector episode and retry all eight clients."""


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self._closed = False
        self.ep_runner = None
        self.recovery_protocol = bool(getattr(args, "recovery_protocol", False))
        self.failure_active = False
        self.eval_seed_base = int(getattr(args, "eval_seed_base", 180000))
        self.eval_episode_cursor = 0
        self.eval_wins = 0
        self.eval_episodes = 0
        self.last_eval_win_rate = None
        self.eval_removed_slots = []
        self.eval_configs = []
        self.eval_manifest_configs = None
        self.eval_manifest_path = getattr(args, "eval_manifest_path", "")
        self.eval_manifest_sha256 = None
        if self.recovery_protocol and self.eval_manifest_path:
            with open(self.eval_manifest_path, "rb") as stream:
                raw = stream.read()
            manifest = json.loads(raw.decode("utf-8"))
            self.eval_manifest_configs = manifest["configurations"]
            if len(self.eval_manifest_configs) != args.test_nepisode:
                raise ValueError("Held-out manifest must contain exactly {} configurations".format(args.test_nepisode))
            if manifest.get("map_name") != args.env_args["map_name"] or \
                    manifest.get("n_agents") != args.env_args["capability_config"]["n_units"] or \
                    manifest.get("n_enemies") != args.env_args["capability_config"]["n_enemies"] or \
                    manifest.get("eval_seed_base") != self.eval_seed_base:
                raise ValueError("Held-out manifest does not match the requested SMACv2 map/team")
            self.eval_manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.selector_sha256 = None
        self.discarded_episode_batches = 0

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg)),
                                                   self.recovery_protocol, getattr(args, "failure_selector_path", "")))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        try:
            self.parent_conns[0].send(("get_env_info", None))
            self.env_info = self._recv(self.parent_conns[0])
        except Exception:
            self.close_env()
            raise
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        # initialize also an episode runner just for the animations
        if not self.recovery_protocol:
            dummy_args = copy.copy(args)
            dummy_args.batch_size_run = 1
            self.ep_runner = EpisodeRunner(dummy_args, logger=None)

    def _recv(self, connection):
        result = connection.recv()
        if isinstance(result, dict) and "__worker_error__" in result:
            raise RecoverableEnvError(result["__worker_error__"])
        return result

    def begin_evaluation(self):
        self.eval_episode_cursor = 0
        self.eval_wins = 0
        self.eval_episodes = 0
        self.last_eval_win_rate = None
        self.eval_removed_slots = []
        self.eval_configs = []

    def set_failure_active(self, active):
        self.failure_active = bool(active)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        # setup also the episode runner
        if self.ep_runner is not None:
            self.ep_runner.setup(scheme, groups, preprocess, mac)

    def get_env_info(self):
        return self.env_info

    def save_replay(self, path):
        if self.ep_runner is None:
            raise RuntimeError("Rendering is not supported in recovery protocol")
        self.ep_runner.save_replay(path)

    def save_animation(self, path):
        if self.ep_runner is None:
            raise RuntimeError("Rendering is not supported in recovery protocol")
        self.ep_runner.save_animation(path)

    def close_env(self):
        if self._closed:
            return
        self._closed = True
        for parent_conn in self.parent_conns:
            try:
                parent_conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.ps:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        if self.ep_runner is not None:
            self.ep_runner.close_env()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset the envs
        for index, parent_conn in enumerate(self.parent_conns):
            episode_seed = None
            episode_config = None
            if self.recovery_protocol and test_mode:
                episode_seed = self.eval_seed_base + self.eval_episode_cursor + index
                if self.eval_manifest_configs is not None:
                    episode_config = self.eval_manifest_configs[self.eval_episode_cursor + index]
            parent_conn.send(("reset", {
                "failure_active": self.failure_active,
                "episode_seed": episode_seed,
                "episode_config": episode_config,
            } if self.recovery_protocol else None))
        if self.recovery_protocol and test_mode:
            self.eval_episode_cursor += self.batch_size

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        reset_data = []
        first_error = None
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            try:
                data = self._recv(parent_conn)
            except RecoverableEnvError as exc:
                first_error = first_error or exc
                continue
            reset_data.append(data)
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        if first_error is not None:
            raise first_error

        self.batch.update(pre_transition_data, ts=0)
        if self.recovery_protocol:
            if self.failure_active:
                for data in reset_data:
                    if data["selector_sha256"] != self.selector_sha256:
                        raise RuntimeError("Worker selector checksum differs from the recorded frozen selector")
            if test_mode:
                self.eval_removed_slots.extend(data["removed_agent_id"] for data in reset_data)
                self.eval_configs.extend(data["episode_config"] for data in reset_data)
            self.batch.update({
                "participating_mask": [data["participating_mask"] for data in reset_data],
                "removed_agent_id": [[data["removed_agent_id"]] for data in reset_data],
            })

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False,render=False, save_animation=False, benchmark_mode=False):

        if self.recovery_protocol:
            last_error = None
            for attempt in range(int(getattr(self.args, "env_batch_retry_limit", 3))):
                eval_cursor = self.eval_episode_cursor
                removed_count = len(self.eval_removed_slots)
                config_count = len(self.eval_configs)
                try:
                    return self._run_once(test_mode, render, save_animation, benchmark_mode)
                except RecoverableEnvError as exc:
                    self.eval_episode_cursor = eval_cursor
                    del self.eval_removed_slots[removed_count:]
                    del self.eval_configs[config_count:]
                    last_error = exc
                    self.discarded_episode_batches += 1
                    self.logger.console_logger.warning(
                        "Discarding incomplete environment batch (attempt %s): %s", attempt + 1, exc)
            raise RuntimeError("Environment recovery exhausted after {} attempts".format(
                getattr(self.args, "env_batch_retry_limit", 3))) from last_error
        return self._run_once(test_mode, render, save_animation, benchmark_mode)

    def _run_once(self, test_mode=False, render=False, save_animation=False, benchmark_mode=False):

        if test_mode and (render or save_animation or benchmark_mode):
            return self.ep_runner.run(test_mode=True, render=render, save_animation=save_animation, benchmark_mode=benchmark_mode)

        self.reset(test_mode=test_mode)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            if all(terminated):
                break

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            first_error = None

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    try:
                        data = self._recv(parent_conn)
                    except RecoverableEnvError as exc:
                        first_error = first_error or exc
                        continue
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            if first_error is not None:
                raise first_error

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # The recovery metrics use per-episode terminal info. The upstream
        # get_stats() result is unused, so avoid an extra fragile SC2 RPC.
        if not self.recovery_protocol:
            for parent_conn in self.parent_conns:
                parent_conn.send(("get_stats", None))
            for parent_conn in self.parent_conns:
                self._recv(parent_conn)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        if self.recovery_protocol and test_mode:
            self.eval_wins += sum(int(info.get("battle_won", False)) for info in final_env_infos)
            self.eval_episodes += self.batch_size
            self.last_eval_win_rate = self.eval_wins / self.eval_episodes

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, recovery_protocol=False, selector_path=""):
    env = None
    selector = None
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        try:
            if cmd == "close":
                if env is not None:
                    env.close()
                remote.close()
                break
            if env is None:
                env = env_fn.x()
            if cmd == "step":
                reward, terminated, env_info = env.step(data)
                remote.send({
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                })
            elif cmd == "reset":
                reset_args = data or {}
                if recovery_protocol:
                    env.reset(episode_seed=reset_args.get("episode_seed"),
                              episode_config=reset_args.get("episode_config"))
                    removed_agent_id = -1
                    if reset_args.get("failure_active", False):
                        if selector is None:
                            info = env.get_env_info()
                            selector = FrozenQMIXSelector(selector_path, info["n_agents"], info["state_shape"])
                        initial_state = env.get_state()
                        initial_mask = env.get_agent_alive_mask()
                        removed_agent_id, _ = selector.select(initial_state, initial_mask)
                        env.remove_agent(removed_agent_id)
                    mask = [float(i != removed_agent_id) for i in range(env.n_agents)]
                    remote.send({
                        "state": env.get_state(),
                        "avail_actions": env.get_avail_actions(),
                        "obs": env.get_obs(),
                        "participating_mask": [[value] for value in mask],
                        "removed_agent_id": removed_agent_id,
                        "selector_sha256": selector.sha256 if selector is not None else None,
                        "episode_config": _json_safe(env.env.episode_config) if reset_args.get("episode_seed") is not None else None,
                    })
                else:
                    env.reset()
                    remote.send({
                        "state": env.get_state(),
                        "avail_actions": env.get_avail_actions(),
                        "obs": env.get_obs(),
                    })
            elif cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "get_stats":
                remote.send(env.get_stats())
            else:
                raise NotImplementedError(cmd)
        except Exception:
            remote.send({"__worker_error__": traceback.format_exc()})
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

