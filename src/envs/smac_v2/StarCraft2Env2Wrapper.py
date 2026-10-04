"""SMACv2 adapter for TransfQMix's entity-based agent and mixer inputs."""

import numpy as np
import random

from pysc2.lib import protocol

from .official.wrapper import StarCraftCapabilityEnvWrapper
from .official.distributions import Distribution
from .official.starcraft2 import CannotResetException


class StarCraft2Env2Wrapper(StarCraftCapabilityEnvWrapper):
    """Expose SMACv2 observations/state as padded entity-token matrices.

    TransfQMix's SMAC agent expects enemy entities first (so attack-action
    outputs align with enemy slots), followed by allied entities. Its mixer
    expects the global state as one row per ally/enemy. Native SMACv2 returns
    flattened feature vectors, so this adapter performs that conversion.
    """

    def __init__(self, **kwargs):
        self.removal_max_attempts = int(kwargs.pop("removal_max_attempts", 5))
        self.kill_unit_step_mul = int(kwargs.pop("kill_unit_step_mul", 2))
        self.reset_max_attempts = int(kwargs.pop("reset_max_attempts", 3))
        if min(self.removal_max_attempts, self.kill_unit_step_mul, self.reset_max_attempts) < 1:
            raise ValueError("SMACv2 removal and reset limits must be positive")
        self.initially_removed_agent_ids = set()
        super().__init__(**kwargs)
        worker_seed = int(kwargs.get("seed") or 0)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32))
        self._seed_distributions(worker_seed)

    def _distribution_rngs(self):
        seen = set()
        result = []

        def visit(item):
            if id(item) in seen:
                return
            seen.add(id(item))
            for name, value in sorted(vars(item).items()):
                if isinstance(value, np.random.Generator):
                    result.append((item, name))
                elif isinstance(value, Distribution):
                    visit(value)

        for _, distribution in sorted(self.env_key_to_distribution_map.items()):
            visit(distribution)
        return result

    def _seed_distributions(self, seed):
        sequence = np.random.SeedSequence(seed)
        for item, name in self._distribution_rngs():
            setattr(item, name, np.random.default_rng(sequence.spawn(1)[0]))

    def _sample_reset_config(self):
        reset_config = {}
        for distribution in self.env_key_to_distribution_map.values():
            reset_config.update(distribution.generate())
        return reset_config

    def _game_ended(self):
        status = getattr(getattr(self.env, "_controller", None), "_status", None)
        return status is not None and str(status).lower().endswith("ended")

    def reset(self, episode_seed=None, episode_config=None):
        """Reset to an in-game controller; optional seed fixes eval composition."""
        self.initially_removed_agent_ids = set()
        if episode_seed is not None:
            rngs = self._distribution_rngs()
            states = [(item, name, item.__dict__[name]) for item, name in rngs]
            self._seed_distributions(int(episode_seed))
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(int(episode_seed))
            np.random.seed(int(episode_seed) % (2 ** 32))
        else:
            states = None
        try:
            last_error = None
            for _ in range(self.reset_max_attempts):
                try:
                    if self._game_ended():
                        self.env.full_restart()
                    if episode_config is None:
                        if episode_seed is not None:
                            self._seed_distributions(int(episode_seed))
                            random.seed(int(episode_seed))
                            np.random.seed(int(episode_seed) % (2 ** 32))
                        reset_config = self._sample_reset_config()
                    else:
                        reset_config = dict(episode_config)
                        for key in ("ally_start_positions", "enemy_start_positions"):
                            if key in reset_config:
                                reset_config[key] = dict(reset_config[key])
                                reset_config[key]["item"] = np.asarray(reset_config[key]["item"])
                    result = self.env.reset(reset_config)
                    if self._game_ended():
                        raise RuntimeError("SMACv2 reset returned an ended controller")
                    return result
                except (CannotResetException, protocol.ProtocolError, protocol.ConnectionError, RuntimeError) as exc:
                    last_error = exc
                    try:
                        self.env.full_restart()
                    except Exception as restart_exc:
                        last_error = restart_exc
            raise RuntimeError("SMACv2 reset failed after {} attempts: {}".format(self.reset_max_attempts, last_error))
        finally:
            if states is not None:
                for item, name, generator in states:
                    setattr(item, name, generator)
                random.setstate(py_state)
                np.random.set_state(np_state)

    def get_agent_alive_mask(self):
        return [float(self.env.agents[i].health > 0) for i in range(self.env.n_agents)]

    def remove_agent(self, agent_id):
        """Delete exactly one allied unit before the first policy step."""
        agent_id = int(agent_id)
        if not 0 <= agent_id < self.env.n_agents:
            raise ValueError("Invalid agent slot {}".format(agent_id))
        unit = self.env.agents[agent_id]
        if unit.health <= 0 or self._game_ended():
            raise RuntimeError("Cannot remove slot {} from inactive game".format(agent_id))
        tag = int(unit.tag)
        alive_before = {i for i, ally in self.env.agents.items() if ally.health > 0}
        try:
            self.env._kill_units([tag])
            for attempt in range(1, self.removal_max_attempts + 1):
                if self._game_ended():
                    raise RuntimeError("SC2 game ended during pre-episode removal")
                self.env._controller.step(self.kill_unit_step_mul)
                self.env._obs = self.env._controller.observe()
                self.env.update_units()
                raw_tags = {int(u.tag) for u in self.env._obs.observation.raw_data.units}
                if tag in raw_tags or self.env.agents[agent_id].health > 0:
                    continue
                alive_after = {i for i, ally in self.env.agents.items() if ally.health > 0}
                unexpected = sorted((alive_before - alive_after) - {agent_id})
                if unexpected:
                    raise RuntimeError("Removing slot {} also removed slots {}".format(agent_id, unexpected))
                self.env.death_tracker_ally[agent_id] = 1
                self.env.last_action[agent_id] = 0
                self.initially_removed_agent_ids.add(agent_id)
                return {"agent_id": agent_id, "unit_tag": tag, "confirmation_frames": attempt * self.kill_unit_step_mul}
        except (protocol.ProtocolError, protocol.ConnectionError) as exc:
            raise RuntimeError("SC2 removal protocol error: {}".format(exc)) from exc
        raise RuntimeError("SC2 did not confirm removal of slot {} tag {}".format(agent_id, tag))

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        if self.initially_removed_agent_ids:
            info = dict(info)
            total = int(info.get("dead_allies", 0))
            info["total_dead_allies"] = total
            info["dead_allies"] = max(0, total - len(self.initially_removed_agent_ids))
            info["initial_removed_allies"] = len(self.initially_removed_agent_ids)
        return reward, terminated, info

    def _entity_feature_sizes(self):
        _, enemy_size = self.env.get_obs_enemy_feats_size()
        _, ally_size = self.env.get_obs_ally_feats_size()
        return enemy_size, ally_size

    def _obs_entity_shape(self):
        enemy_size, ally_size = self._entity_feature_sizes()
        return max(enemy_size, ally_size) + (2 if self.env.obs_own_pos else 0) + 2

    def get_obs_agent(self, agent_id):
        env = self.env
        if env.agents[agent_id].health <= 0:
            return np.zeros(self.get_obs_size(), dtype=np.float32)
        raw = np.asarray(env.get_obs_agent(agent_id), dtype=np.float32)
        n_enemies = env.n_enemies
        n_agents = env.n_agents
        n_enemy_feats, n_ally_feats = self._entity_feature_sizes()
        move_size = env.get_obs_move_feats_size()
        n_allies = n_agents - 1
        own_size = env.get_obs_own_feats_size()

        cursor = move_size
        enemy_end = cursor + n_enemies * n_enemy_feats
        enemies = raw[cursor:enemy_end].reshape(n_enemies, n_enemy_feats)
        cursor = enemy_end
        ally_end = cursor + n_allies * n_ally_feats
        allies = raw[cursor:ally_end].reshape(n_allies, n_ally_feats)
        own = raw[ally_end:ally_end + own_size]

        if env.stochastic_attack or env.stochastic_health or env.conic_fov or env.obs_last_action:
            raise ValueError("Entity adapter supports the published team/position SMACv2 settings only")
        feat_dim = self._obs_entity_shape()
        core_dim = max(n_enemy_feats, n_ally_feats)
        entities = np.zeros((n_agents + n_enemies, feat_dim), dtype=np.float32)
        entities[:n_enemies, :n_enemy_feats] = enemies
        entities[:n_enemies, -2] = 0.0  # not an ally
        entities[:n_enemies, -1] = 0.0  # not self

        ally_ids = [i for i in range(n_agents) if i != agent_id]
        for ally_row, ally_id in enumerate(ally_ids):
            entities[n_enemies + ally_id, :n_ally_feats] = allies[ally_row]
            entities[n_enemies + ally_id, -2] = 1.0

        self_row = np.zeros(feat_dim, dtype=np.float32)
        self_row[:4] = (1.0, 0.0, 0.0, 0.0)
        own_cursor = 0
        ally_cursor = 4
        if env.obs_own_health:
            self_row[ally_cursor] = own[own_cursor]
            own_cursor += 1
            ally_cursor += 1
            if env.shield_bits_ally:
                self_row[ally_cursor] = own[own_cursor]
                own_cursor += 1
                ally_cursor += 1
        if env.obs_own_pos:
            self_row[core_dim:core_dim + 2] = own[own_cursor:own_cursor + 2]
            own_cursor += 2
        if env.unit_type_bits:
            self_row[ally_cursor:ally_cursor + env.unit_type_bits] = own[
                own_cursor:own_cursor + env.unit_type_bits]

        self_row[-2] = 1.0
        self_row[-1] = 1.0
        entities[n_enemies + agent_id] = self_row
        if env.map_type in ("MMM", "terran_gen") and \
                env.agents[agent_id].unit_type == env.medivac_id:
            # SMACv2 encodes Medivac heal actions by ALLY slot rather than
            # enemy slot. The unchanged TransfQMix entity-action head reads
            # its first n_enemies tokens as action targets, so put allies
            # first for Medivacs (the benchmark uses equal team sizes).
            if n_agents != n_enemies:
                raise ValueError("Medivac entity-action alignment requires equal team sizes")
            entities = np.concatenate((entities[n_enemies:], entities[:n_enemies]), axis=0)
        return entities.reshape(-1)

    def get_obs(self):
        return [self.get_obs_agent(agent_id) for agent_id in range(self.env.n_agents)]

    def get_obs_size(self):
        return (self.env.n_agents + self.env.n_enemies) * self._obs_entity_shape()

    def get_state(self):
        state = self.env.get_state_dict()
        allies = np.asarray(state["allies"], dtype=np.float32)
        enemies = np.asarray(state["enemies"], dtype=np.float32)
        feat_dim = max(allies.shape[-1], enemies.shape[-1] + 1) + 1

        entities = np.zeros(
            (self.env.n_agents + self.env.n_enemies, feat_dim), dtype=np.float32
        )
        # Match the original TransfQMix SMAC entity-state order:
        # health, x, y, cooldown/energy, remaining features, ally flag.
        entities[:self.env.n_agents, :3] = allies[:, [0, 2, 3]]
        entities[:self.env.n_agents, 3] = allies[:, 1]
        entities[:self.env.n_agents, 4:allies.shape[-1]] = allies[:, 4:]
        entities[:self.env.n_agents, -1] = (allies[:, 0] > 0).astype(np.float32)
        entities[self.env.n_agents:, :3] = enemies[:, :3]
        entities[self.env.n_agents:, 4:enemies.shape[-1] + 1] = enemies[:, 3:]
        for agent_id in self.initially_removed_agent_ids:
            entities[agent_id] = 0.0
        return entities.reshape(-1)

    def get_state_size(self):
        feat_dim = max(
            self.env.get_ally_num_attributes(), self.env.get_enemy_num_attributes() + 1
        ) + 1
        return (self.env.n_agents + self.env.n_enemies) * feat_dim

    def get_env_info(self):
        info = super().get_env_info()
        state_feat_dim = max(
            self.env.get_ally_num_attributes(), self.env.get_enemy_num_attributes() + 1
        ) + 1
        info.update(
            {
                "state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_entities": self.env.n_agents + self.env.n_enemies,
                "n_entities_state": self.env.n_agents + self.env.n_enemies,
                "obs_entity_feats": self._obs_entity_shape(),
                "state_entity_feats": state_feat_dim,
                "n_normal_actions": self.env.n_actions_no_attack,
            }
        )
        return info
