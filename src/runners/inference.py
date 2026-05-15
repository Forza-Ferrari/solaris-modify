import abc
import functools
import os
import time
from collections import defaultdict

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch.multiprocessing as mp
from absl import logging
from einops import rearrange
from flax import nnx
from tqdm import tqdm

import src.data.utils as data_utils
import src.utils.sharding as sharding_utils
import src.utils.wandb as wandb_utils
from src.data.dataset import VideoReadError
from src.models.model_loaders import get_jax_clip_model, get_vae_model
from src.runners.base_mp_runner import BaseMPRunner
from src.runners.base_runner import BaseRunner
from src.utils.config import get_obj_from_str, instantiate_from_config
from src.utils.preprocessing_mp import wan_image_condition_preprocess


class Inference(BaseMPRunner):

    def __init__(
        self,
        model_weights_path,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.model_weights_path = model_weights_path

        def get_model(rngs):
            nnx_rngs = nnx.Rngs(params=rngs)
            model = instantiate_from_config(self.network_config, rngs=nnx_rngs)
            return nnx.split(model)

        _, state_shape = jax.eval_shape(functools.partial(get_model, rngs=self.rngs))
        self.model_sharding = sharding_utils.apply_sharding(state_shape, self.mesh)
        self.model_graph, model_state = jax.jit(
            get_model, out_shardings=(self.repl_sharding, self.model_sharding)
        )(self.rngs)
        self.model_state = self._restore_model_state(model_weights_path, model_state)
        self.multiplayer_method = self.network_config.params.multiplayer_method

    def _create_player_embed_state(self, num_players, model_dim):
        dummy_embed = nnx.Embed(
            num_embeddings=num_players,
            features=model_dim,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(12)),
        )
        _, embed_state = nnx.split(dummy_embed)

        def to_global(x):
            if isinstance(x, jax.Array):
                host = np.asarray(jax.device_get(x), dtype=x.dtype)
                return jax.device_put(host, self.repl_sharding)
            return x

        return jax.tree_util.tree_map(to_global, embed_state)

    def _resize_player_embed_state(self, restored_embed_state, target_embed_state):
        restored_embed = restored_embed_state["embedding"].value
        target_embed = target_embed_state["embedding"].value

        restored_host = np.asarray(jax.device_get(restored_embed), dtype=target_embed.dtype)
        target_shape = tuple(target_embed.shape)
        resized = np.zeros(target_shape, dtype=target_embed.dtype)

        copy_rows = min(restored_host.shape[0], target_shape[0])
        if copy_rows > 0:
            resized[:copy_rows] = restored_host[:copy_rows]
        if copy_rows < target_shape[0]:
            fill_row = restored_host[copy_rows - 1] if copy_rows > 0 else 0
            for row_idx in range(copy_rows, target_shape[0]):
                resized[row_idx] = fill_row

        target_embed_state["embedding"].value = jax.device_put(
            resized, target_embed.sharding
        )
        return target_embed_state

    def _restore_model_state(self, model_weights_path, model_state):
        model_dim = int(self.network_config.params.dim)
        target_player_count = int(self.network_config.params.num_players)

        if target_player_count == 2:
            return self.pretrained_checkpointer.restore(model_weights_path, model_state)

        restore_template = model_state.copy()
        restore_template["player_embed"] = self._create_player_embed_state(
            num_players=2,
            model_dim=model_dim,
        )
        restored_state = self.pretrained_checkpointer.restore(
            model_weights_path, restore_template
        )
        restored_state["player_embed"] = self._resize_player_embed_state(
            restored_state["player_embed"],
            model_state["player_embed"],
        )
        logging.info(
            "Expanded player_embed from checkpoint capacity 2 to runtime capacity %d",
            target_player_count,
        )
        return restored_state

    def _evaluate(
        self,
        model_state,
        model_graph,
        vae_state,
        vae_graph,
        clip_state,
        clip_graph,
        video, 
        mouse_actions,
        keyboard_actions,
        real_lengths,
        eval_dir,
        mesh,
        left_action_padding,
        num_denoising_steps=None,
    ):
        return self.evaluate_mp(
            bidirectional=False,
            model_state=model_state,
            model_graph=model_graph,
            vae_state=vae_state,
            vae_graph=vae_graph,
            clip_state=clip_state,
            clip_graph=clip_graph,
            video=video,
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            real_lengths=real_lengths,
            eval_dir=eval_dir,
            mesh=mesh,
            left_action_padding=left_action_padding,
            num_denoising_steps=num_denoising_steps,
        )

    def run(self):

        with self.mesh:
            self.run_evals()

    def run_evals(self):

        for eval_dataset_name, eval_dataloader_info in self.eval_dataloaders.items():
            model = nnx.merge(self.model_graph, self.model_state)
            logging.info(f"Running eval on {eval_dataset_name}")
            self.run_eval(
                model=model,
                num_denoising_steps=None,
                eval_dataloader_info=eval_dataloader_info,
                eval_dir_name=eval_dataset_name,
            )
