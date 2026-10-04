import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json
import hashlib
import atexit

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners.frozen_qmix_selector import FrozenQMIXSelector
from utils.recovery_metrics import summarize_recovery


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    print("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    print("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner, test_episodes=10):

    save_dir = os.path.join(args.local_results_path, 'renders', f'{args.unique_token}')
    if args.save_replay or args.save_animation:
        os.makedirs(save_dir, exist_ok=True)

    infos = []
    for i in range(test_episodes):
        episode_info = runner.run(
            test_mode=True,
            render=args.save_replay, 
            save_animation=args.save_animation and (not args.evaluate or i%args.animation_interval_evaluation==0), 
            benchmark_mode=True
        )
        for e_i in episode_info: 
            e_i["episode"] = i+1
        if args.save_replay:
            runner.save_replay(os.path.join(save_dir, f'render_episode_{runner.t_env}_{i}'))
        if args.save_animation and (not args.evaluate or i%args.animation_interval_evaluation==0):
            runner.save_animation(os.path.join(save_dir, f'animation_episode_{runner.t_env}_{i}'))
        infos.extend(episode_info)

    if "utracking" in args.env_args.get("key", ""):
        import pandas as pd
        exp_name = f"{args.env_args['num_agents']}v{args.env_args['num_landmarks']}_{args.env_args['movement']}_{args.env_args['difficulty']}"
        bench_dir = os.path.join(args.local_results_path, 'benchmarks', exp_name)
        os.makedirs(bench_dir, exist_ok=True)
        df = pd.DataFrame(infos)
        df.to_csv(os.path.join(bench_dir, f'{args.name}.csv'), index=False)

    #runner.close_env()


def run_sequential(args, logger):

    recovery_protocol = bool(getattr(args, "recovery_protocol", False))
    if recovery_protocol:
        if args.env != "sc2_v2" or args.runner != "parallel" or args.batch_size_run != 8:
            raise ValueError("Recovery protocol requires SMACv2 and eight parallel environments")
        if args.checkpoint_path or args.evaluate or args.save_replay or args.save_animation:
            raise ValueError("Recovery run must be continuous from scratch; replay/evaluate modes are separate")
        if args.test_nepisode != 32:
            raise ValueError("Recovery protocol requires 32 greedy evaluation episodes")
        if int(args.env_batch_retry_limit) < 1 or int(args.post_failure_epsilon_anneal_time) < 1:
            raise ValueError("Environment retry and recovery annealing limits must be positive")
        if not getattr(args, "failure_selector_path", ""):
            raise ValueError("Specify failure_selector_path to a shared frozen QMIX mixer checkpoint")
        candidate = args.failure_selector_path
        candidate = os.path.join(candidate, "mixer.th") if os.path.isdir(candidate) else candidate
        if not os.path.isfile(candidate):
            raise FileNotFoundError("Shared frozen selector mixer is missing: {}".format(candidate))
        failure_t_env = int(args.failure_t_env)
        recovery_budget = int(args.recovery_budget)
        if failure_t_env < 1 or recovery_budget < 1 or int(args.t_max) != failure_t_env + recovery_budget:
            raise ValueError("Set t_max = failure_t_env + recovery_budget, both positive")

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    atexit.register(runner.close_env)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    logger.console_logger.info(f"Env info: {env_info}")
    audit = None
    audit_path = None
    if recovery_protocol:
        selector = FrozenQMIXSelector(args.failure_selector_path,
                                      env_info["n_agents"], env_info["state_shape"])
        runner.selector_sha256 = selector.sha256
        audit = {
            "method": args.name,
            "map_name": args.env_args["map_name"],
            "n_agents": env_info["n_agents"],
            "n_enemies": args.env_args["capability_config"]["n_enemies"],
            "seed": args.seed,
            "selector_checkpoint": selector.path,
            "selector_sha256": selector.sha256,
            "selection_rule": "argmax_i mean_h abs(W1_i,h(s0)); initial complete-team state",
            "eval_seed_base": runner.eval_seed_base,
            "eval_manifest_path": runner.eval_manifest_path or None,
            "eval_manifest_sha256": runner.eval_manifest_sha256,
            "eval_episodes": args.test_nepisode,
            "failure_t_env_nominal": failure_t_env,
            "recovery_budget": recovery_budget,
            "recovery_epsilon": {
                "start": args.post_failure_epsilon_start,
                "finish": args.post_failure_epsilon_finish,
                "anneal_time": args.post_failure_epsilon_anneal_time,
            },
            "evaluations": [],
            "complete": False,
        }
        audit_dir = os.path.join(args.local_results_path, "recovery_audit")
        os.makedirs(audit_dir, exist_ok=True)
        audit_path = os.path.join(audit_dir, args.unique_token + ".json")

        def save_audit():
            temporary_path = audit_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as output:
                json.dump(audit, output, ensure_ascii=False, indent=2)
            os.replace(temporary_path, audit_path)

        save_audit()

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if recovery_protocol:
        scheme["participating_mask"] = {
            "vshape": (1,), "group": "agents", "episode_const": True,
        }
        scheme["removed_agent_id"] = {
            "vshape": (1,), "dtype": th.long, "episode_const": True,
        }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

    if args.evaluate or args.save_replay:
        runner.log_train_stats_t = runner.t_env
        evaluate_sequential(args, runner, test_episodes=args.test_nepisode)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_animation_T = -args.animation_interval - 1
    last_log_T = 0
    model_save_time = 0
    failure_start_t_env = None
    post_failure_batches = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while True:

        if not recovery_protocol and runner.t_env > args.t_max:
            break

        if recovery_protocol and failure_start_t_env is None and runner.t_env >= failure_t_env:
            failure_start_t_env = runner.t_env
            audit["failure_start_t_env_actual"] = failure_start_t_env
            runner.set_failure_active(True)
            mac.action_selector.activate_post_failure(
                failure_start_t_env, args.post_failure_epsilon_start,
                args.post_failure_epsilon_finish, args.post_failure_epsilon_anneal_time)
            save_audit()
            logger.console_logger.info("Persistent physical-removal regime starts at t_env=%s", failure_start_t_env)
        if recovery_protocol and failure_start_t_env is not None \
                and runner.t_env >= failure_start_t_env + recovery_budget:
            break

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        if recovery_protocol and failure_start_t_env is not None:
            post_failure_batches += 1

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        force_first_post_test = recovery_protocol and post_failure_batches == 1
        force_final_post_test = recovery_protocol and failure_start_t_env is not None \
            and runner.t_env >= failure_start_t_env + recovery_budget
        if force_first_post_test or force_final_post_test or (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            if recovery_protocol:
                runner.begin_evaluation()
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            if recovery_protocol:
                win_rate = runner.last_eval_win_rate
                if runner.eval_episodes != args.test_nepisode or win_rate is None:
                    raise RuntimeError("Recovery evaluation did not finish all 32 episodes")
                if runner.eval_manifest_configs is None:
                    manifest = {
                        "map_name": args.env_args["map_name"],
                        "n_agents": env_info["n_agents"],
                        "n_enemies": args.env_args["capability_config"]["n_enemies"],
                        "eval_seed_base": runner.eval_seed_base,
                        "configurations": runner.eval_configs,
                    }
                    manifest_dir = os.path.join(args.local_results_path, "eval_manifests")
                    os.makedirs(manifest_dir, exist_ok=True)
                    manifest_path = os.path.join(manifest_dir, args.unique_token + ".json")
                    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
                    with open(manifest_path, "wb") as output:
                        output.write(manifest_bytes)
                    runner.eval_manifest_configs = runner.eval_configs
                    runner.eval_manifest_path = os.path.abspath(manifest_path)
                    runner.eval_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                    audit["eval_manifest_path"] = runner.eval_manifest_path
                    audit["eval_manifest_sha256"] = runner.eval_manifest_sha256
                elif runner.eval_configs != runner.eval_manifest_configs:
                    raise RuntimeError("Held-out evaluation configurations changed during recovery")
                audit["evaluations"].append({
                    "t_env": runner.t_env,
                    "failure_active": runner.failure_active,
                    "win_rate": win_rate,
                    "removed_slots": list(runner.eval_removed_slots),
                })
                if runner.failure_active:
                    logger.log_stat("test_post_failure_battle_won_mean", win_rate, runner.t_env)
                    logger.log_stat("post_failure_steps", runner.t_env - failure_start_t_env, runner.t_env)
                logger.log_stat("failure_active", float(runner.failure_active), runner.t_env)
                audit["discarded_episode_batches"] = runner.discarded_episode_batches
                save_audit()

        # save an animation
        if ((runner.t_env - last_animation_T) / args.animation_interval >= 1.0) and (
            args.save_animation
        ):
            last_animation_T = runner.t_env
            evaluate_sequential(args, runner, test_episodes=1)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if recovery_protocol:
        audit["metrics"] = summarize_recovery(audit["evaluations"],
                                               failure_start_t_env, recovery_budget,
                                               nominal_failure_t=failure_t_env)
        audit["complete"] = runner.t_env >= failure_start_t_env + recovery_budget
        audit["end_t_env"] = runner.t_env
        for name, value in audit["metrics"].items():
            if value is not None:
                logger.log_stat(name, value, runner.t_env)
        save_audit()
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
