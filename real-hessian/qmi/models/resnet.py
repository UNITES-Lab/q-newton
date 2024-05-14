# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/2
"""
Simple ResNet-18 model for MNIST classification
"""

from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import ResNetConfig
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
)


class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


class FlaxResNetConvLayer(nn.Module):
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    activation: Optional[str] = "relu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.kernel_size // 2,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="normal", dtype=self.dtype),
        )
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        hidden_state = self.activation_func(hidden_state)
        return hidden_state


class FlaxResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedder = FlaxResNetConvLayer(
            self.config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=self.config.hidden_act,
            dtype=self.dtype,
        )

        self.max_pool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        if num_channels != self.config.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values, deterministic=deterministic)
        embedding = self.max_pool(embedding)
        return embedding


class FlaxResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    out_channels: int
    stride: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convolution = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=self.stride,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal"),
            dtype=self.dtype,
        )
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        return hidden_state


class FlaxResNetBasicLayerCollection(nn.Module):
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer = [
            FlaxResNetConvLayer(self.out_channels, stride=self.stride, dtype=self.dtype),
            FlaxResNetConvLayer(self.out_channels, activation=None, dtype=self.dtype),
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


class FlaxResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    in_channels: int
    out_channels: int
    stride: int = 1
    activation: Optional[str] = "relu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)
            if should_apply_shortcut
            else None
        )
        self.layer = FlaxResNetBasicLayerCollection(
            out_channels=self.out_channels,
            stride=self.stride,
            dtype=self.dtype,
        )
        self.activation_func = ACT2FN[self.activation]

    def __call__(self, hidden_state, deterministic: bool = True):
        residual = hidden_state
        hidden_state = self.layer(hidden_state, deterministic=deterministic)

        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state += residual

        hidden_state = self.activation_func(hidden_state)
        return hidden_state


class FlaxResNetBottleNeckLayerCollection(nn.Module):
    out_channels: int
    stride: int = 1
    activation: Optional[str] = "relu"
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        reduces_channels = self.out_channels // self.reduction

        self.layer = [
            FlaxResNetConvLayer(reduces_channels, kernel_size=1, dtype=self.dtype, name="0"),
            FlaxResNetConvLayer(reduces_channels, stride=self.stride, dtype=self.dtype, name="1"),
            FlaxResNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name="2"),
        ]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


class FlaxResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions. The first `1x1` convolution reduces the
    input by a factor of `reduction` in order to make the second `3x3` convolution faster. The last `1x1` convolution
    remaps the reduced features to `out_channels`.
    """

    in_channels: int
    out_channels: int
    stride: int = 1
    activation: Optional[str] = "relu"
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = (
            FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype)
            if should_apply_shortcut
            else None
        )

        self.layer = FlaxResNetBottleNeckLayerCollection(
            self.out_channels,
            stride=self.stride,
            activation=self.activation,
            reduction=self.reduction,
            dtype=self.dtype,
        )

        self.activation_func = ACT2FN[self.activation]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        residual = hidden_state

        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state = self.layer(hidden_state, deterministic)
        hidden_state += residual
        hidden_state = self.activation_func(hidden_state)
        return hidden_state


class FlaxResNetStageLayersCollection(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        layer = FlaxResNetBottleNeckLayer if self.config.layer_type == "bottleneck" else FlaxResNetBasicLayer

        layers = [
            # downsampling is done in the first layer with stride of 2
            layer(
                self.in_channels,
                self.out_channels,
                stride=self.stride,
                activation=self.config.hidden_act,
                dtype=self.dtype,
                name="0",
            ),
        ]

        for i in range(self.depth - 1):
            layers.append(
                layer(
                    self.out_channels,
                    self.out_channels,
                    activation=self.config.hidden_act,
                    dtype=self.dtype,
                    name=str(i + 1),
                )
            )

        self.layers = layers

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_state = x
        for layer in self.layers:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state


class FlaxResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    config: ResNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxResNetStageLayersCollection(
            self.config,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            depth=self.depth,
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.layers(x, deterministic=deterministic)


class FlaxResNetStageCollection(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        stages = [
            FlaxResNetStage(
                self.config,
                self.config.embedding_size,
                self.config.hidden_sizes[0],
                stride=2 if self.config.downsample_in_first_stage else 1,
                depth=self.config.depths[0],
                dtype=self.dtype,
                name="0",
            )
        ]

        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(
                FlaxResNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1))
            )

        self.stages = stages

    def __call__(
            self,
            hidden_state: jnp.ndarray,
            output_hidden_states: bool = False,
            deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

            hidden_state = stage_module(hidden_state, deterministic=deterministic)

        return hidden_state, hidden_states


class FlaxResNetEncoder(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.stages = FlaxResNetStageCollection(self.config, dtype=self.dtype)

    def __call__(
            self,
            hidden_state: jnp.ndarray,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        hidden_state, hidden_states = self.stages(
            hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class FlaxResNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
            self,
            config: ResNetConfig,
            input_shape=(1, 224, 224, 3),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        rngs = {"params": rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
            self,
            pixel_values,
            params: dict = None,
            train: bool = False,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}

        return self.module.apply(
            {
                "params": params["params"] if params is not None else self.params["params"],
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=["batch_stats"] if train else False,  # Returing tuple with batch_stats only when train is True
        )


class FlaxResNetModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embedder = FlaxResNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxResNetEncoder(self.config, dtype=self.dtype)

        # Adaptive average pooling used in resnet
        self.pooler = partial(
            nn.avg_pool,
            padding=((0, 0), (0, 0)),
        )

    def __call__(
            self,
            pixel_values,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, deterministic=deterministic)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
            strides=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        ).transpose(0, 3, 1, 2)

        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class FlaxResNetModel(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetModule


class FlaxResNetClassifierCollection(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name="1")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.classifier(x)


class FlaxResNetForImageClassificationModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.resnet = FlaxResNetModule(config=self.config, dtype=self.dtype)

        if self.config.num_labels > 0:
            self.classifier = FlaxResNetClassifierCollection(self.config, dtype=self.dtype)
        else:
            self.classifier = Identity()

    def __call__(
            self,
            pixel_values=None,
            deterministic: bool = True,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output[:, :, 0, 0])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)


class FlaxResNetForImageClassification(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetForImageClassificationModule


resnet_toy_for_mnit_config = ResNetConfig(
    hidden_sizes=[8, 8, 8],
    num_labels=10,
    num_channels=1,
    embedding_size=16,
    depths=[2, 2, 2],
)

resnet_tiny_for_test = ResNetConfig(
    hidden_sizes=[8, 8, 8],
    num_labels=10,
    num_channels=1,
    embedding_size=8,
    depths=[2, 2, 2],
)

resnet_toy_for_mnist = partial(FlaxResNetForImageClassification,
                               resnet_toy_for_mnit_config)

resnet_tiny_for_test = partial(FlaxResNetForImageClassification,
                               resnet_tiny_for_test)
