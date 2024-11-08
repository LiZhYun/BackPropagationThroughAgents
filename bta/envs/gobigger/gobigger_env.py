from typing import Any, List, Union, Optional, Tuple
import time
import copy
import math
from collections import OrderedDict
from collections.abc import Sequence, Mapping
import collections.abc as container_abcs
import cv2
import numpy as np
# from ding.envs import BaseEnv
# from ding.envs.common.env_element import EnvElement, EnvElementInfo
# from ding.torch_utils import to_tensor, to_ndarray, to_list
# from ding.utils import ENV_REGISTRY
from gobigger.server import Server
from gobigger.render import EnvRender
import torch
import re
import os
import sys
import gym
from gym.spaces import Dict, Discrete, Box, Tuple

int_classes = int
string_classes = str
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

def default_collate(batch: Sequence) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Put each data field into a tensor with outer dimension batch size.
    Example:
        >>> # a list with B tensors shaped (m, n) -->> a tensor shaped (B, m, n)
        >>> a = [torch.zeros(2,3) for _ in range(4)]
        >>> default_collate(a).shape
        torch.Size([4, 2, 3])
        >>>
        >>> # a list with B lists, each list contains m elements -->> a list of m tensors, each with shape (B, )
        >>> a = [[0 for __ in range(3)] for _ in range(4)]
        >>> default_collate(a)
        [tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0])]
        >>>
        >>> # a list with B dicts, whose values are tensors shaped :math:`(m, n)` -->>
        >>> # a dict whose values are tensors with shape :math:`(B, m, n)`
        >>> a = [{i: torch.zeros(i,i+1) for i in range(2, 4)} for _ in range(4)]
        >>> print(a[0][2].shape, a[0][3].shape)
        torch.Size([2, 3]) torch.Size([3, 4])
        >>> b = default_collate(a)
        >>> print(b[2].shape, b[3].shape)
        torch.Size([4, 2, 3]) torch.Size([4, 3, 4])
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, directly concatenate into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.shape == (1,):
            # reshape (B, 1) -> (B)
            return torch.cat(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(
                        elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret

class GoBiggerEnv(gym.Env):

    def __init__(self, server_cfg=None, **kwargs):
        self.server_cfg = server_cfg
        self._cfg = server_cfg
        self._player_num_per_team = server_cfg.player_num_per_team
        self._team_num = server_cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = server_cfg.match_time
        self._map_height = server_cfg.map_height
        self._map_width = server_cfg.map_width
        self._spatial = server_cfg.spatial
        self._train = server_cfg.train
        self._last_team_size = None
        self._init_flag = False
        self._speed = server_cfg.speed
        self.device = server_cfg.device
        self.n_actions = 3
        self.max_player_num = self._player_num_per_team
        self.max_ball_num = self.server_cfg.max_ball_num
        self.max_food_num = self.server_cfg.max_food_num
        self.max_spore_num = self.server_cfg.max_spore_num
        self.direction_num = self.server_cfg.direction_num
        self.spatial_x = 64
        self.spatial_y = 64
        self.step_mul = self.server_cfg.step_mul
        self.second_per_frame = self.server_cfg.second_per_frame
        self.action_num = self.direction_num * 2 + 3
        self.use_render = self.server_cfg.use_render
        self.init_server()
        self.get_space()
        self.setup_action()

    def setup_action(self):
        theta = math.pi * 2 / self.direction_num
        self.x_y_action_List = [[0.3 * math.cos(theta * i), 0.3 * math.sin(theta * i), 0] for i in
                                range(self.direction_num)] + \
                               [[math.cos(theta * i), math.sin(theta * i), 0] for i in
                                range(self.direction_num)] + \
                               [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    
    def generate_action_mask(self, can_eject, can_split, ):
        action_mask = torch.zeros(size=(self.action_num,), dtype=torch.bool)
        if not can_eject:
            action_mask[self.direction_num * 2 + 1] = True
        if not can_split:
            action_mask[self.direction_num * 2 + 2] = True
        return action_mask

    def transform_action(self, action_idx):
        return self.x_y_action_List[int(action_idx)]
        
    def step(self, actions):
        for i in range(self.step_mul):
            if i==0:
                done = self.server.step(actions=actions)
            else:
                done = self.server.step(actions=None)
        obs_raw = self.server.obs()
        global_state, player_states, info = obs_raw
        obs_raw = [global_state, player_states]
        obs, share_obs, processed_share_obs_list = self.preprocess_obs(obs_raw)
        total_score = [global_state['leaderboard'][i] \
                        for i in range(len(global_state['leaderboard']))]
        assert len(self.last_total_score) == len(total_score)
        reward = [total_score[0] - self.last_total_score[0] for i in range(self._player_num_per_team)]
        self.last_total_score = total_score
        return processed_share_obs_list, obs, obs_raw, reward, done, info

    def reset(self):
        self.server.reset()
        obs_raw = self.server.obs()
        global_state, player_states, info = obs_raw
        obs_raw = [global_state, player_states]
        obs, share_obs, processed_share_obs_list = self.preprocess_obs(obs_raw)
        self.last_total_score = [global_state['leaderboard'][i] \
                                for i in range(len(global_state['leaderboard']))]
        return processed_share_obs_list, obs, obs_raw
    
    def get_space(self):
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.server.reset()
        obs_raw = self.server.obs()
        global_state, player_states, info = obs_raw
        obs_raw = [global_state, player_states]
        obs, share_obs, processed_share_obs_list = self.preprocess_obs(obs_raw)
        self.last_total_score = [global_state['leaderboard'][i] \
                                for i in range(len(global_state['leaderboard']))]
        
        for idx in range(self._player_num_per_team):
            self.action_space.append(Discrete(self.action_num))
            obs_space = Dict(
                {
                    "scalar_info": Dict(
                        {
                            "view_x": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["view_x"].shape,)),
                            "view_y": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["view_y"].shape,)),
                            "view_width": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["view_width"].shape,)),
                            "score": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["score"].shape,)),
                            "team_score": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["team_score"].shape,)),
                            "time": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["time"].shape,)),
                            "rank": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["rank"].shape,)),
                            "last_action_type": Box(-np.inf, np.inf, (*obs[idx]["scalar_info"]["last_action_type"].shape,)),
                        }
                    ),
                    "team_info": Dict(
                        {
                            "alliance": Box(-np.inf, np.inf, (*obs[idx]["team_info"]["alliance"].shape,)),
                            "view_x": Box(-np.inf, np.inf, (*obs[idx]["team_info"]["view_x"].shape,)),
                            "view_y": Box(-np.inf, np.inf, (*obs[idx]["team_info"]["view_x"].shape,)),
                            "player_num": Box(-np.inf, np.inf, (*obs[idx]["team_info"]["player_num"].shape,)),
                        }
                    ),
                    "ball_info": Dict(
                        {
                            "alliance": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["alliance"].shape,)),
                            "score": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["score"].shape,)),
                            "radius": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["radius"].shape,)),
                            "rank": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["rank"].shape,)),
                            "x": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["x"].shape,)),
                            "y": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["y"].shape,)),
                            "next_x": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["next_x"].shape,)),
                            "next_y": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["next_y"].shape,)),
                            "ball_num": Box(-np.inf, np.inf, (*obs[idx]["ball_info"]["ball_num"].shape,)),
                        }
                    ),
                    "spatial_info": Dict(
                        {
                            'food_x': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["food_x"].shape,)),
                            'food_y': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["food_y"].shape,)),
                            'spore_x': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["spore_x"].shape,)),
                            'spore_y': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["spore_y"].shape,)),
                            'ball_x': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["ball_x"].shape,)),
                            'ball_y': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["ball_y"].shape,)),
                            'food_num': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["food_num"].shape,)),
                            'spore_num': Box(-np.inf, np.inf, (*obs[idx]["spatial_info"]["spore_num"].shape,)),
                        }
                    )
                }
            )
            self.observation_space.append(obs_space)
            share_obs_space = Dict(
                {
                    "scalar_info": Dict(
                        {
                            "view_x": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["view_x"].shape,)),
                            "view_y": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["view_y"].shape,)),
                            "view_width": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["view_width"].shape,)),
                            "score": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["score"].shape,)),
                            "team_score": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["team_score"].shape,)),
                            "time": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["time"].shape,)),
                            "rank": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["rank"].shape,)),
                            "last_action_type": Box(-np.inf, np.inf, (*share_obs["scalar_info"]["last_action_type"].shape,)),
                        }
                    ),
                    "team_info": Dict(
                        {
                            "alliance": Box(-np.inf, np.inf, (*share_obs["team_info"]["alliance"].shape,)),
                            "view_x": Box(-np.inf, np.inf, (*share_obs["team_info"]["view_x"].shape,)),
                            "view_y": Box(-np.inf, np.inf, (*share_obs["team_info"]["view_x"].shape,)),
                            "player_num": Box(-np.inf, np.inf, (*share_obs["team_info"]["player_num"].shape,)),
                        }
                    ),
                    "ball_info": Dict(
                        {
                            "alliance": Box(-np.inf, np.inf, (*share_obs["ball_info"]["alliance"].shape,)),
                            "score": Box(-np.inf, np.inf, (*share_obs["ball_info"]["score"].shape,)),
                            "radius": Box(-np.inf, np.inf, (*share_obs["ball_info"]["radius"].shape,)),
                            "rank": Box(-np.inf, np.inf, (*share_obs["ball_info"]["rank"].shape,)),
                            "x": Box(-np.inf, np.inf, (*share_obs["ball_info"]["x"].shape,)),
                            "y": Box(-np.inf, np.inf, (*share_obs["ball_info"]["y"].shape,)),
                            "next_x": Box(-np.inf, np.inf, (*share_obs["ball_info"]["next_x"].shape,)),
                            "next_y": Box(-np.inf, np.inf, (*share_obs["ball_info"]["next_y"].shape,)),
                            "ball_num": Box(-np.inf, np.inf, (*share_obs["ball_info"]["ball_num"].shape,)),
                        }
                    ),
                    "spatial_info": Dict(
                        {
                            'food_x': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["food_x"].shape,)),
                            'food_y': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["food_y"].shape,)),
                            'spore_x': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["spore_x"].shape,)),
                            'spore_y': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["spore_y"].shape,)),
                            'ball_x': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["ball_x"].shape,)),
                            'ball_y': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["ball_y"].shape,)),
                            'food_num': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["food_num"].shape,)),
                            'spore_num': Box(-np.inf, np.inf, (*share_obs["spatial_info"]["spore_num"].shape,)),
                        }
                    )
                }
            )
            self.share_observation_space.append(share_obs_space)

    def close(self):
        self.server.close()

    def seed(self, seed):
        self.server.seed(seed)

    def get_team_infos(self):
        assert hasattr(self, 'server'), "Please call `reset()` first"
        return self.server.get_team_infos()

    def init_server(self):
        self.server = Server(cfg=self.server_cfg)
    
    def preprocess_obs(self, env_obs):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        processed_obs_list = []
        processed_share_obs_list = []
        for game_player_id in range(self._player_num_per_team):
            last_action_type = self.direction_num * 2
            env_player_obs, share_output_obs = self.transform_obs(env_obs, game_player_id=game_player_id, padding=True,
                                                            last_action_type=last_action_type)
            processed_obs_list.append(env_player_obs)
            processed_share_obs_list.append(share_output_obs)
        share_obs = self.default_collate_with_dim(processed_share_obs_list, device=self.device)
        return np.stack(processed_obs_list), share_obs, processed_share_obs_list
    
    def default_collate_with_dim(self, batch,device='cpu',dim=0, k=None,cat=False):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        # print("********type is:  ",elem_type)
        #if k is not None:
        #    print(k)

        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            # try:
            if cat == True:
                return torch.cat(batch, dim=dim, out=out).to(device=device)
            else:
                return torch.stack(batch, dim=dim, out=out).to(device=device)
            # except:
            #     print(batch)
            #     if k is not None:
            #         print(k)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate_with_dim([torch.from_numpy(b,device=device) for b in batch],device=device,dim=dim,cat=cat)
            elif elem.shape == ():  # scalars
                # try:
                return torch.as_tensor(batch,device=device)
                # except:
                #     print(batch)
                #     if k is not None:
                #         print(k)
        elif isinstance(elem, float):
            # try:
            return torch.tensor(batch,device=device)
            # except:
            #     print(batch)
            #     if k is not None:
            #         print(k)
        elif isinstance(elem, int_classes):
            # try:
            return torch.tensor(batch,device=device)
            # except:
            #     print(batch)
            #     if k is not None:
            #         print(k)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.default_collate_with_dim([d[key] for d in batch if key in d.keys()],device=device,dim=dim, k=key, cat=cat) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    def get_player2team(self, ):
        player2team = {}
        for player_id in range(self._player_num_per_team * self._team_num):
            player2team[player_id] = player_id // self._player_num_per_team
        return player2team

    def transform_obs(self, obs, game_player_id=1, padding=True, last_action_type=None, ):
        global_state, player_observations = obs
        player2team = self.get_player2team()
        own_player_id = game_player_id
        leaderboard = global_state['leaderboard']
        team2rank = {key: rank for rank, key in enumerate(sorted(leaderboard, key=leaderboard.get, reverse=True), )}

        own_player_obs = player_observations[own_player_id]
        own_team_id = player2team[own_player_id]

        # ===========
        # scalar info
        # ===========
        scene_size = global_state['border'][0]
        own_left_top_x, own_left_top_y, own_right_bottom_x, own_right_bottom_y = own_player_obs['rectangle']
        own_view_center = [(own_left_top_x + own_right_bottom_x - scene_size) / 2,
                           (own_left_top_y + own_right_bottom_y - scene_size) / 2]
        own_view_width = float(own_right_bottom_x - own_left_top_x)
        # own_view_height = float(own_right_bottom_y - own_left_top_y)

        own_score = own_player_obs['score'] / 100
        own_team_score = global_state['leaderboard'][own_team_id] / 100
        own_rank = team2rank[own_team_id]

        scalar_info = {
            'view_x': torch.tensor(own_view_center[0]).round().long().unsqueeze(0),
            'view_y': torch.tensor(own_view_center[1]).round().long().unsqueeze(0),
            'view_width': torch.tensor(own_view_width).round().long().unsqueeze(0),
            'score': torch.log(torch.tensor(own_score) / 10).round().long().clamp_(max=9).unsqueeze(0),
            'team_score': torch.log(torch.tensor(own_team_score / 10)).round().long().clamp_(max=9).unsqueeze(0),
            'time': torch.tensor(global_state['last_time']//20, dtype=torch.long).unsqueeze(0),
            'rank': torch.tensor(own_rank, dtype=torch.long).unsqueeze(0),
            'last_action_type': torch.tensor(last_action_type, dtype=torch.long).unsqueeze(0)
        }
        share_scalar_info = {
            'view_x': torch.tensor(own_view_center[0]).round().long(),
            'view_y': torch.tensor(own_view_center[1]).round().long(),
            'view_width': torch.tensor(own_view_width).round().long(),
            'score': torch.log(torch.tensor(own_score) / 10).round().long().clamp_(max=9),
            'team_score': torch.log(torch.tensor(own_team_score / 10)).round().long().clamp_(max=9),
            'time': torch.tensor(global_state['last_time']//20, dtype=torch.long),
            'rank': torch.tensor(own_rank, dtype=torch.long),
            'last_action_type': torch.tensor(last_action_type, dtype=torch.long)
        }
        # ===========
        # team_info
        # ===========

        all_players = []
        scene_size = global_state['border'][0]

        for game_player_id in player_observations.keys():
            game_team_id = player2team[game_player_id]
            game_player_left_top_x, game_player_left_top_y, game_player_right_bottom_x, game_player_right_bottom_y = \
                player_observations[game_player_id]['rectangle']
            if game_player_id == own_player_id:
                alliance = 0
            elif game_team_id == own_team_id:
                alliance = 1
            else:
                alliance = 2
            if alliance != 2:
                game_player_view_x = (game_player_right_bottom_x + game_player_left_top_x - scene_size) / 2
                game_player_view_y = (game_player_right_bottom_y + game_player_left_top_y - scene_size) / 2

                # game_player_view_width = game_player_right_bottom_x - game_player_left_top_x
                # game_player_view_height = game_player_right_bottom_y -  game_player_left_top_y
                #
                # game_player_score = math.log((player_observations[game_player_id]['score'] + 1) / 1000)
                # game_player_team_score = math.log((global_state['leaderboard'][game_team_id] + 1) / 1000)
                # game_player_rank = team2rank[game_team_id]

                all_players.append([alliance,
                                    game_player_view_x,
                                    game_player_view_y,
                                    # game_player_view_width,
                                    # game_player_view_height,
                                    # game_player_score,
                                    # game_player_team_score,
                                    # game_player_rank,
                                    ])
        all_players = torch.as_tensor(all_players)
        player_padding_num = self.max_player_num - len(all_players)
        player_num = len(all_players)
        all_players = torch.nn.functional.pad(all_players, (0, 0, 0, player_padding_num), 'constant', 0)
        team_info = {
            'alliance': all_players[:, 0].long(),
            'view_x': all_players[:, 1].round().long(),
            'view_y': all_players[:, 2].round().long(),
            # 'view_width': all_players[:,3].round().long(),
            # 'view_height': all_players[:,4].round().long(),
            # 'score': all_players[:,5].round().long().clamp_(max=10,min=0),
            # 'team_score': all_players[:,6].round().long().clamp_(max=10,min=0),
            # 'team_rank': all_players[:,7].long(),
            'player_num': torch.tensor(player_num, dtype=torch.long).unsqueeze(0),
        }
        share_team_info = {
            'alliance': all_players[:, 0].long(),
            'view_x': all_players[:, 1].round().long(),
            'view_y': all_players[:, 2].round().long(),
            # 'view_width': all_players[:,3].round().long(),
            # 'view_height': all_players[:,4].round().long(),
            # 'score': all_players[:,5].round().long().clamp_(max=10,min=0),
            # 'team_score': all_players[:,6].round().long().clamp_(max=10,min=0),
            # 'team_rank': all_players[:,7].long(),
            'player_num': torch.tensor(player_num, dtype=torch.long),
        }

        # ===========
        # ball info
        # ===========
        ball_type_map = {'clone': 1, 'food': 2, 'thorns': 3, 'spore': 4}
        clone = own_player_obs['overlap']['clone']
        thorns = own_player_obs['overlap']['thorns']
        food = own_player_obs['overlap']['food']
        spore = own_player_obs['overlap']['spore']

        neutral_team_id = self._team_num
        neutral_player_id = self._team_num * self._player_num_per_team
        neutral_team_rank = self._team_num

        clone = [[ball_type_map['clone'], bl[3], bl[-2], bl[-1], team2rank[bl[-1]], bl[0], bl[1],
                  *self.next_position(bl[0], bl[1], bl[4], bl[5])] for bl in clone]
        thorns = [[ball_type_map['thorns'], bl[3], neutral_player_id, neutral_team_id, neutral_team_rank, bl[0], bl[1],
                   *self.next_position(bl[0], bl[1], bl[4], bl[5])] for bl in thorns]
        food = [
            [ball_type_map['food'], bl[3], neutral_player_id, neutral_team_id, neutral_team_rank, bl[0], bl[1], bl[0],
             bl[1]] for bl in food]

        spore = [
            [ball_type_map['spore'], bl[3], bl[-1], player2team[bl[-1]], team2rank[player2team[bl[-1]]], bl[0],
             bl[1],
             *self.next_position(bl[0], bl[1], bl[4], bl[5])] for bl in spore]

        all_balls = clone + thorns + food + spore

        for b in all_balls:
            if b[2] == own_player_id and b[0] == 1:
                if b[5] < own_left_top_x or b[5] > own_right_bottom_x or \
                        b[6] < own_left_top_y or b[6] > own_right_bottom_y:
                    b[5] = int((own_left_top_x + own_right_bottom_x) / 2)
                    b[6] = int((own_left_top_y + own_right_bottom_y) / 2)
                    b[7], b[8] = b[5], b[6]
        all_balls = torch.as_tensor(all_balls, dtype=torch.float)

        origin_x = own_left_top_x
        origin_y = own_left_top_y

        all_balls[:, -4] = ((all_balls[:, -4] - origin_x) / own_view_width * self.spatial_x)
        all_balls[:, -3] = ((all_balls[:, -3] - origin_y) / own_view_width * self.spatial_y)
        all_balls[:, -2] = ((all_balls[:, -2] - origin_x) / own_view_width * self.spatial_x)
        all_balls[:, -1] = ((all_balls[:, -1] - origin_y) / own_view_width * self.spatial_y)

        # ball
        ball_indices = torch.logical_and(all_balls[:, 0] != 2,
                                         all_balls[:, 0] != 4)  # include player balls and thorn balls
        balls = all_balls[ball_indices]

        balls_num = len(balls)

        # consider position of thorns ball
        if balls_num > self.max_ball_num:  # filter small balls
            own_indices = balls[:, 3] == own_player_id
            teammate_indices = (balls[:, 4] == own_team_id) & ~own_indices
            enemy_indices = balls[:, 4] != own_team_id

            own_balls = balls[own_indices]
            teammate_balls = balls[teammate_indices]
            enemy_balls = balls[enemy_indices]

            if own_balls.shape[0] + teammate_balls.shape[0] >= self.max_ball_num:
                remain_ball_num = self.max_ball_num - own_balls.shape[0]
                teammate_ball_score = teammate_balls[:, 1]
                teammate_high_score_indices = teammate_ball_score.sort(descending=True)[1][:remain_ball_num]
                teammate_remain_balls = teammate_balls[teammate_high_score_indices]
                balls = torch.cat([own_balls, teammate_remain_balls])
            else:
                remain_ball_num = self.max_ball_num - own_balls.shape[0] - teammate_balls.shape[0]
                enemy_ball_score = enemy_balls[:, 1]
                enemy_high_score_ball_indices = enemy_ball_score.sort(descending=True)[1][:remain_ball_num]
                remain_enemy_balls = enemy_balls[enemy_high_score_ball_indices]

                balls = torch.cat([own_balls, teammate_balls, remain_enemy_balls])

        ball_padding_num = self.max_ball_num - len(balls)
        if padding or ball_padding_num < 0:
            balls = torch.nn.functional.pad(balls, (0, 0, 0, ball_padding_num), 'constant', 0)
            alliance = torch.zeros(self.max_ball_num)
            balls_num = min(self.max_ball_num, balls_num)
        else:
            alliance = torch.zeros(balls_num)
        alliance[balls[:, 3] == own_team_id] = 2
        alliance[balls[:, 2] == own_player_id] = 1
        alliance[balls[:, 3] != own_team_id] = 3
        alliance[balls[:, 0] == 3] = 0

        ## score&radius
        scale_score = balls[:, 1] / 100
        radius = (torch.sqrt(scale_score * 0.042 + 0.15) / own_view_width).clamp_(max=1)
        score = ((torch.sqrt(scale_score * 0.042 + 0.15) / own_view_width).clamp_(max=1) * 50).round().long().clamp_(
            max=49)

        ## rank:
        ball_rank = balls[:, 4]

        ## coordinate
        x = balls[:, -4] - self.spatial_x // 2
        y = balls[:, -3] - self.spatial_y // 2
        next_x = balls[:, -2] - self.spatial_x // 2
        next_y = balls[:, -1] - self.spatial_y // 2

        ball_info = {
            'alliance': alliance.long(),
            'score': score.long(),
            'radius': radius,
            'rank': ball_rank.long(),
            'x': x.round().long(),
            'y': y.round().long(),
            'next_x': next_x.round().long(),
            'next_y': next_y.round().long(),
            'ball_num': torch.tensor(balls_num, dtype=torch.long).unsqueeze(0)
        }
        share_ball_info = {
            'alliance': alliance.long(),
            'score': score.long(),
            'radius': radius,
            'rank': ball_rank.long(),
            'x': x.round().long(),
            'y': y.round().long(),
            'next_x': next_x.round().long(),
            'next_y': next_y.round().long(),
            'ball_num': torch.tensor(balls_num, dtype=torch.long)
        }

        # ============
        # spatial info
        # ============
        # ball coordinate for scatter connection
        ball_x = balls[:, -4]
        ball_y = balls[:, -3]

        food_indices = all_balls[:, 0] == 2
        food_x = all_balls[food_indices, -4]
        food_y = all_balls[food_indices, -3]
        food_num = len(food_x)
        food_padding_num = self.max_food_num - len(food_x)
        if padding or food_padding_num < 0:
            food_x = torch.nn.functional.pad(food_x, (0, food_padding_num), 'constant', 0)
            food_y = torch.nn.functional.pad(food_y, (0, food_padding_num), 'constant', 0)
        food_num = min(food_num, self.max_food_num)

        spore_indices = all_balls[:, 0] == 4
        spore_x = all_balls[spore_indices, -4]
        spore_y = all_balls[spore_indices, -3]
        spore_num = len(spore_x)
        spore_padding_num = self.max_spore_num - len(spore_x)
        if padding or spore_padding_num < 0:
            spore_x = torch.nn.functional.pad(spore_x, (0, spore_padding_num), 'constant', 0)
            spore_y = torch.nn.functional.pad(spore_y, (0, spore_padding_num), 'constant', 0)
        spore_num = min(spore_num, self.max_spore_num)

        spatial_info = {
            'food_x': food_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'food_y': food_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'spore_x': spore_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'spore_y': spore_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'ball_x': ball_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'ball_y': ball_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'food_num': torch.tensor(food_num, dtype=torch.long).unsqueeze(0),
            'spore_num': torch.tensor(spore_num, dtype=torch.long).unsqueeze(0)
        }
        share_spatial_info = {
            'food_x': food_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'food_y': food_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'spore_x': spore_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'spore_y': spore_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'ball_x': ball_x.round().clamp_(min=0, max=self.spatial_x - 1).long(),
            'ball_y': ball_y.round().clamp_(min=0, max=self.spatial_y - 1).long(),
            'food_num': torch.tensor(food_num, dtype=torch.long),
            'spore_num': torch.tensor(spore_num, dtype=torch.long)
        }

        output_obs = {
            'scalar_info': scalar_info,
            'team_info': team_info,
            'ball_info': ball_info,
            'spatial_info': spatial_info,
        }
        share_output_obs = {
            'scalar_info': share_scalar_info,
            'team_info': share_team_info,
            'ball_info': share_ball_info,
            'spatial_info': share_spatial_info,
        }
        return output_obs, share_output_obs
    
    def next_position(self, x, y, vel_x, vel_y):
        next_x = x + self.second_per_frame * vel_x * self.step_mul
        next_y = y + self.second_per_frame * vel_y * self.step_mul
        return next_x, next_y
