# -*- coding: utf-8 -*-
# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to interface with the `vis/` (the JS viewer).
"""
from __future__ import annotations

import asyncio
import enum
import os
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from cryptography.utils import CryptographyDeprecationWarning

from arcnerf.render.ray_helper import get_rays
from common.utils.cfgs_utils import get_value_from_cfgs_field
from .arcnerf_to_ns_viewer import ns_view_to_arcnerf_cam
from .ns_utils import (
    load_from_json, write_to_json, get_dict_to_torch, apply_colormap, apply_depth_colormap, apply_boolean_colormap
)
from .subprocess import run_viewer_bridge_server_as_subprocess
from .utils import force_codec, get_intrinsics_matrix_and_camera_to_world_h
from .video_stream import SingleFrameStreamTrack
from .visualizer import Viewer

warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)


def get_viewer_version() -> str:
    """Get the version of the viewer."""
    json_filename = os.path.join(os.path.dirname(__file__), '../app/package.json')
    version = load_from_json(Path(json_filename))['version']
    return version


class OutputTypes(str, enum.Enum):
    """Noncomprehsnive list of output render types"""

    INIT = 'init'
    RGB = 'rgb'
    RGB_FINE = 'rgb_fine'
    ACCUMULATION = 'accumulation'
    ACCUMULATION_FINE = 'accumulation_fine'


class ColormapTypes(str, enum.Enum):
    """Noncomprehsnive list of colormap render types"""

    INIT = 'init'
    DEFAULT = 'default'
    TURBO = 'turbo'
    DEPTH = 'depth'
    SEMANTIC = 'semantic'
    BOOLEAN = 'boolean'


class IOChangeException(Exception):
    """Basic camera exception to interrupt viewer"""


class SetTrace:
    """Basic trace function"""

    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)


class RenderThread(threading.Thread):
    """Thread that does all the rendering calls while listening for interrupts

    Args:
        state: current viewer state object
        model: current checkpoint of model
    """

    def __init__(self, state: 'ViewerState', model, inputs):
        threading.Thread.__init__(self)
        self.state = state
        self.model = model
        self.inputs = inputs
        self.exc = None
        self.vis_outputs = None

    def run(self):
        """run function that renders out images given the current graph and ray bundles.
        Interlaced with a trace function that checks to see if any I/O changes were registered.
        Exits and continues program if IOChangeException thrown.
        """
        outputs = None
        try:
            with SetTrace(self.state.check_interrupt):
                with torch.no_grad():
                    _outputs = self.model(self.inputs, inference_only=True)  # (1, HW, 3/1)
                # change to ns_viewer format
                H, W = self.inputs['H'], self.inputs['W']
                outputs = {
                    'rgb': _outputs['rgb'].view(H, W, -1),
                    'accumulation': _outputs['mask'].view(H, W, -1),
                    'depth': _outputs['depth'].view(H, W, -1),
                }

        except Exception as e:  # pylint: disable=broad-except
            self.exc = e

        if outputs:
            outputs = get_dict_to_torch(outputs)
            self.vis_outputs = outputs

        self.state.check_done_render = True
        self.state.check_interrupt_vis = False

    def join(self, timeout=None):
        threading.Thread.join(self)
        if self.exc:
            raise self.exc


class CheckThread(threading.Thread):
    """Thread the constantly checks for io changes and sets a flag indicating interrupt

    Args:
        state: current viewer state object
    """

    def __init__(self, state):
        threading.Thread.__init__(self)
        self.state = state

    def run(self):
        """Run function that checks to see if any of the existing state has changed
        (e.g. camera pose/output type/resolutions).
        Sets the viewer state flag to true to signal
        to render thread that an interrupt was registered.
        """
        self.state.check_done_render = False
        while not self.state.check_done_render:
            # check camera
            data = self.state.vis['renderingState/camera'].read()
            if data is not None:
                camera_object = data['object']
                if self.state.prev_camera_matrix is None or (
                    not np.allclose(camera_object['matrix'], self.state.prev_camera_matrix)
                    and not self.state.prev_moving
                ):
                    self.state.check_interrupt_vis = True
                    self.state.prev_moving = True
                    return
                self.state.prev_moving = False

            # check output type
            output_type = self.state.vis['renderingState/output_choice'].read()
            if output_type is None:
                output_type = OutputTypes.INIT
            if self.state.prev_output_type != output_type:
                self.state.check_interrupt_vis = True
                return

            # check colormap type
            colormap_type = self.state.vis['renderingState/colormap_choice'].read()
            if colormap_type is None:
                colormap_type = ColormapTypes.INIT
            if self.state.prev_colormap_type != colormap_type:
                self.state.check_interrupt_vis = True
                return

            # check max render
            max_resolution = self.state.vis['renderingState/maxResolution'].read()
            if max_resolution is not None:
                if self.state.max_resolution != max_resolution:
                    self.state.check_interrupt_vis = True
                    return


class ViewerState:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
    """

    def __init__(self, config, logger, log_filename: Path):
        self.config = config
        self.vis = None
        self.viewer_url = None
        self.logger = logger
        self.log_filename = log_filename
        if self.config.launch_bridge_server:
            # start the viewer bridge server
            assert self.config.websocket_port is not None
            self.log_filename.parent.mkdir(exist_ok=True)
            zmq_port = run_viewer_bridge_server_as_subprocess(
                self.config.websocket_port,
                zmq_port=self.config.zmq_port,
                ip_address=self.config.ip_address,
                log_filename=str(self.log_filename),
            )
            # TODO(ethan): log the output of the viewer bridge server in a file where the training logs go
            version = get_viewer_version()
            websocket_url = f'ws://localhost:{self.config.websocket_port}'
            self.viewer_url = f'https://viewer.nerf.studio/versions/{version}/?websocket_url={websocket_url}'
            self.logger.add_log(f'[Public] Open the viewer at {self.viewer_url}\n')
            self.vis = Viewer(zmq_port=zmq_port, ip_address=self.config.ip_address)
            self.vis_webrtc_thread = Viewer(zmq_port=zmq_port, ip_address=self.config.ip_address)
        else:
            assert self.config.zmq_port is not None
            self.vis = Viewer(zmq_port=self.config.zmq_port, ip_address=self.config.ip_address)
            self.vis_webrtc_thread = Viewer(zmq_port=self.config.zmq_port, ip_address=self.config.ip_address)

        # viewer specific variables
        self.prev_camera_matrix = None
        self.prev_output_type = OutputTypes.INIT
        self.prev_colormap_type = ColormapTypes.INIT
        self.prev_moving = False
        self.output_type_changed = True
        self.max_resolution = get_value_from_cfgs_field(self.config, 'max_resolution', 1024)
        self.rays_per_sec = get_value_from_cfgs_field(self.config, 'rays_per_sec', 80000)
        self.check_interrupt_vis = False
        self.check_done_render = True
        self.step = 0
        self.static_fps = 1
        self.moving_fps = 24
        self.camera_moving = False
        self.prev_camera_timestamp = 0

        self.output_list = None

        # webrtc
        self.pcs = set()
        self.video_tracks = set()
        self.webrtc_thread = None
        self.kill_webrtc_signal = False

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(self, dataset, start_train=True) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            start_train: whether to start train when viewer init;
                if False, only displays dataset until resume train is toggled
        """
        # set the config base dir
        self.vis['renderingState/config_base_dir'].write(str(self.log_filename.parents[0]))

        # clear the current scene
        self.vis['sceneState/sceneBox'].delete()
        self.vis['sceneState/cameras'].delete()

        # draw the training cameras and images
        image_indices = self._pick_drawn_image_idxs(len(dataset))
        for idx in image_indices:
            image = dataset[idx]['image']
            bgr = image[..., [2, 1, 0]]
            camera_json = dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.vis[f'sceneState/cameras/{idx:06d}'].write(camera_json)

        # draw the scene box (i.e., the bounding box)
        json_ = dataset.scene_box.to_json()
        self.vis['sceneState/sceneBox'].write(json_)

        # set the initial state whether to train or not
        self.vis['renderingState/isTraining'].write(start_train)

        # self.vis["renderingState/render_time"].write(str(0))

        # set the properties of the camera
        # self.vis["renderingState/camera"].write(json_)

        # set the main camera intrinsics to one from the dataset
        # K = camera.get_intrinsics_matrix()
        # set_persp_intrinsics_matrix(self.vis, K.double().numpy())

    def _check_camera_path_payload(self, trainer, step: int):
        """Check to see if the camera path export button was pressed."""
        # check if we should interrupt from a button press?
        camera_path_payload = self.vis['camera_path_payload'].read()
        if camera_path_payload:
            # save a model checkpoint
            trainer.save_checkpoint(step)
            # write to json file
            camera_path_filename = camera_path_payload['camera_path_filename']
            camera_path = camera_path_payload['camera_path']
            write_to_json(Path(camera_path_filename), camera_path)
            self.vis['camera_path_payload'].delete()

    def _check_webrtc_offer(self):
        """Check if there is a webrtc offer to respond to."""
        data = self.vis['webrtc/offer'].read()

        if data:
            if self.webrtc_thread and self.webrtc_thread.is_alive():
                # kill the previous thread if the webpage refreshes
                self.kill_webrtc_signal = True
                return

            def loop_in_thread(loop):
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_webrtc_answer(data))

            loop = asyncio.get_event_loop()
            self.webrtc_thread = threading.Thread(target=loop_in_thread, args=(loop, ))
            self.webrtc_thread.daemon = True
            self.webrtc_thread.start()
            # remove the offer from the state tree
            self.vis['webrtc/offer'].delete()

    def update_scene(self, trainer, step, graph, num_rays_per_batch=1024, sec_per_batch=0.0):
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            graph: the model
        """
        is_training = self.vis['renderingState/isTraining'].read()
        self.step = step

        self._check_camera_path_payload(trainer, step)
        self._check_webrtc_offer()

        camera_object = self._get_camera_object()
        if camera_object is None:
            return

        if sec_per_batch > 0.0:
            rays_per_sec = num_rays_per_batch / sec_per_batch
        else:
            rays_per_sec = self.rays_per_sec

        if is_training is None or is_training:
            # in training mode

            if self.camera_moving:
                # if the camera is moving, then we pause training and update camera continuously
                while self.camera_moving:
                    self._render_image_in_viewer(camera_object, graph, is_training, rays_per_sec)
                    camera_object = self._get_camera_object()
            else:
                # if the camera is not moving, then we approximate how many training steps need to be taken
                # to render at a FPS defined by self.static_fps.
                if sec_per_batch > 0.0:
                    step_per_sec = 1.0 / sec_per_batch
                    num_steps = max(int(1 / self.static_fps * step_per_sec), 1)

                else:
                    num_steps = 1

                if step % num_steps == 0:
                    self._render_image_in_viewer(camera_object, graph, is_training, rays_per_sec)

        else:
            # in pause training mode, enter render loop with set graph
            local_step = step
            run_loop = not is_training
            while run_loop:
                # if self._is_render_step(local_step) and step > 0:
                if step > 0:
                    self._render_image_in_viewer(camera_object, graph, is_training, rays_per_sec)
                    camera_object = self._get_camera_object()
                is_training = self.vis['renderingState/isTraining'].read()
                self._check_camera_path_payload(trainer, step)
                self._check_webrtc_offer()
                run_loop = not is_training
                local_step += 1

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == 'line':
            if self.check_interrupt_vis and not self.camera_moving:
                raise IOChangeException
        return self.check_interrupt

    def _get_camera_object(self):
        """Gets the camera object from the viewer and updates the movement state if it has changed."""

        data = self.vis['renderingState/camera'].read()
        if data is None:
            return None

        camera_object = data['object']

        if self.prev_camera_matrix is not None and np.allclose(camera_object['matrix'], self.prev_camera_matrix):
            self.camera_moving = False
        else:
            self.prev_camera_matrix = camera_object['matrix']
            self.camera_moving = True
        return camera_object

    def _apply_colormap(self, outputs: Dict[str, Any], colors: torch.Tensor = None, eps=1e-6):
        """Determines which colormap to use based on set colormap type

        Args:
            outputs: the output tensors for which to apply colormaps on
            colors: is only set if colormap is for semantics. Defaults to None.
            eps: epsilon to handle floating point comparisons
        """
        if self.output_list:
            reformatted_output = self._process_invalid_output(self.prev_output_type)

        # default for rgb images
        if self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].shape[-1] == 3:
            return outputs[reformatted_output]

        # rendering depth outputs
        if self.prev_colormap_type == ColormapTypes.DEPTH or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].dtype == torch.float and
            (torch.max(outputs[reformatted_output]) - 1.0) > eps  # handle floating point arithmetic
        ):
            accumulation_str = (
                OutputTypes.ACCUMULATION
                if OutputTypes.ACCUMULATION in self.output_list else OutputTypes.ACCUMULATION_FINE
            )
            return apply_depth_colormap(outputs[reformatted_output], accumulation=outputs[accumulation_str])

        # rendering accumulation outputs
        if self.prev_colormap_type == ColormapTypes.TURBO or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].dtype == torch.float
        ):
            return apply_colormap(outputs[reformatted_output])

        # rendering semantic outputs
        if self.prev_colormap_type == ColormapTypes.SEMANTIC or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].dtype == torch.int
        ):
            logits = outputs[reformatted_output]
            labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)  # type: ignore
            assert colors is not None
            return colors[labels]

        # rendering boolean outputs
        if self.prev_colormap_type == ColormapTypes.BOOLEAN or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].dtype == torch.bool
        ):
            return apply_boolean_colormap(outputs[reformatted_output])

        raise NotImplementedError

    async def send_webrtc_answer(self, data):
        """Setup the webrtc connection."""
        # returns the description to for WebRTC to the specific websocket connection
        offer = RTCSessionDescription(data['sdp'], data['type'])

        if self.config.skip_openrelay:
            ice_servers = [
                RTCIceServer(urls='stun:stun.l.google.com:19302'),
            ]
        else:
            ice_servers = [
                RTCIceServer(urls='stun:stun.l.google.com:19302'),
                RTCIceServer(urls='stun:openrelay.metered.ca:80'),
                RTCIceServer(
                    urls='turn:openrelay.metered.ca:80', username='openrelayproject', credential='openrelayproject'
                ),
                RTCIceServer(
                    urls='turn:openrelay.metered.ca:443', username='openrelayproject', credential='openrelayproject'
                ),
                RTCIceServer(
                    urls='turn:openrelay.metered.ca:443?transport=tcp',
                    username='openrelayproject',
                    credential='openrelayproject',
                ),
            ]

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self.pcs.add(pc)

        video = SingleFrameStreamTrack()
        self.video_tracks.add(video)
        video_sender = pc.addTrack(video)
        force_codec(pc, video_sender, 'video/VP8')

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self.vis_webrtc_thread['webrtc/answer'].write({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
        self.vis_webrtc_thread['webrtc/answer'].delete()

        # continually exchange media
        while True:
            await asyncio.sleep(1)
            if self.kill_webrtc_signal:
                self.kill_webrtc_signal = False
                return

    def set_image(self, image):
        """Write the image over webrtc."""
        for video_track in self.video_tracks:
            video_track.put_frame(image)

    def _send_output_to_viewer(self, outputs: Dict[str, Any], colors: torch.Tensor = None, eps=1e-6):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the graph
            colors: is only set if colormap is for semantics. Defaults to None.
            eps: epsilon to handle floating point comparisons
        """
        if self.output_list is None:
            self.output_list = list(outputs.keys())
            viewer_output_list = list(np.copy(self.output_list))
            # remapping rgb_fine -> rgb for all cases just so that we dont have 2 of them in the options
            if OutputTypes.RGB_FINE in self.output_list:
                viewer_output_list.remove(OutputTypes.RGB_FINE)
            viewer_output_list.insert(0, OutputTypes.RGB)
            self.vis['renderingState/output_options'].write(viewer_output_list)

        reformatted_output = self._process_invalid_output(self.prev_output_type)
        # re-register colormaps and send to viewer
        if self.output_type_changed or self.prev_colormap_type == ColormapTypes.INIT:
            self.prev_colormap_type = ColormapTypes.DEFAULT
            colormap_options = [ColormapTypes.DEFAULT]
            if (
                outputs[reformatted_output].shape[-1] != 3 and outputs[reformatted_output].dtype == torch.float
                and (torch.max(outputs[reformatted_output]) - 1.0) <= eps  # handle floating point arithmetic
            ):
                # accumulation can also include depth
                colormap_options.extend(['depth'])
            self.output_type_changed = False
            self.vis['renderingState/colormap_choice'].write(self.prev_colormap_type)
            self.vis['renderingState/colormap_options'].write(colormap_options)
        selected_output = (self._apply_colormap(outputs, colors) * 255).type(torch.uint8)
        image = selected_output.cpu().numpy()
        self.set_image(image)

    def _update_viewer_stats(self, num_rays: int, image_height: int, image_width: int) -> None:
        """Function that calculates and populates all the rendering statistics accordingly

        Args:
            num_rays: number of rays rendered
            image_height: resolution of the current view
            image_width: resolution of the current view
        """
        self.vis['renderingState/eval_res'].write(f'{image_height}x{image_width}px')
        # TODO: Just update
        self.vis['renderingState/train_eta'].write('Paused')
        self.vis['renderingState/vis_train_ratio'].write('100% spent on viewer')

    def _calculate_image_res(self, camera_object, is_training: bool, rays_per_sec: float) -> Optional[Tuple[int, int]]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            camera_object: the camera object to use for rendering
            is_training: whether or not we are training
            rays_per_sec: estimated rays per second
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_resolution = self.vis['renderingState/maxResolution'].read()
        if max_resolution:
            self.max_resolution = max_resolution

        if self.camera_moving or not is_training:
            target_train_util = 0
        else:
            target_train_util = self.vis['renderingState/targetTrainUtil'].read()
            if target_train_util is None:
                target_train_util = 0.9

        current_fps = self.moving_fps if self.camera_moving else self.static_fps

        # calculate number of rays that can be rendered given the target fps
        num_vis_rays = rays_per_sec / current_fps * (1 - target_train_util)

        aspect_ratio = camera_object['aspect']

        if not self.camera_moving and not is_training:
            image_height = self.max_resolution
        else:
            image_height = (num_vis_rays / aspect_ratio)**0.5
            image_height = int(round(image_height, -1))
            image_height = min(self.max_resolution, image_height)
        image_width = int(image_height * aspect_ratio)
        if image_width > self.max_resolution:
            image_width = self.max_resolution
            image_height = int(image_width / aspect_ratio)

        return image_height, image_width

    def _process_invalid_output(self, output_type: str) -> str:
        """Check to see whether we are in the corner case of RGB; if still invalid, throw error
        Returns correct string mapping given improperly formatted output_type.

        Args:
            output_type: reformatted output type
        """
        if output_type == OutputTypes.INIT:
            output_type = OutputTypes.RGB

        # check if rgb or rgb_fine should be the case TODO: add other checks here
        attempted_output_type = output_type
        if output_type not in self.output_list and output_type == OutputTypes.RGB:
            output_type = OutputTypes.RGB_FINE

        # check if output_type is not in list
        if output_type not in self.output_list:
            assert (
                NotImplementedError
            ), f'Output {attempted_output_type} not in list. Tried to reformat as {output_type} but still not found.'
        return output_type

    def _render_image_in_viewer(self, camera_object, graph, is_training: bool, rays_per_sec: float) -> None:
        # pylint: disable=too-many-statements
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent of a TCP connection and then uses WebRTC to send it to the viewer.

        Args:
            graph: current checkpoint of model
        """
        # Check that timestamp is newer than the last one
        if int(camera_object['timestamp']) < self.prev_camera_timestamp:
            return

        self.prev_camera_timestamp = int(camera_object['timestamp'])

        # check and perform output type updates
        output_type = self.vis['renderingState/output_choice'].read()
        output_type = OutputTypes.INIT if output_type is None else output_type
        self.output_type_changed = self.prev_output_type != output_type
        self.prev_output_type = output_type

        # check and perform colormap type updates
        colormap_type = self.vis['renderingState/colormap_choice'].read()
        colormap_type = ColormapTypes.INIT if colormap_type is None else colormap_type
        self.prev_colormap_type = colormap_type

        # Calculate camera pose and intrinsics
        try:
            image_height, image_width = self._calculate_image_res(camera_object, is_training, rays_per_sec)
        except ZeroDivisionError as e:
            self.vis['renderingState/log_errors'].write('Error: Screen too small; no rays intersecting scene.')
            time.sleep(0.03)  # sleep to allow buffer to reset
            print(f'Error: {e}')
            return

        if image_height is None:
            return

        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            camera_object, image_height=image_height
        )

        camera_to_world = camera_to_world_h[:3, :]
        camera_to_world = torch.stack(
            [
                camera_to_world[0, :],
                camera_to_world[2, :],
                camera_to_world[1, :],
            ],
            dim=0,
        )

        # This is to get the inputs for arcnerf
        c2w = torch.cat([camera_to_world, torch.Tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        c2w = ns_view_to_arcnerf_cam(c2w)
        rays = get_rays(
            image_width,
            image_height,
            wh_order=False,
            c2w=c2w.cuda(),
            intrinsic=intrinsics_matrix.cuda(),
            center_pixel=True
        )
        inputs = {
            'rays_o': rays[0][None],
            'rays_d': rays[1][None],
            'rays_r': rays[3][None],
            'H': image_height,
            'W': image_width
        }
        n_rays = rays[0].shape[0]

        graph.eval()

        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, model=graph, inputs=inputs)

        check_thread.daemon = True
        render_thread.daemon = True

        check_thread.start()
        render_thread.start()
        try:
            render_thread.join()
            check_thread.join()
        except IOChangeException:
            del inputs
            torch.cuda.empty_cache()
        except RuntimeError as e:
            self.vis['renderingState/log_errors'].write(
                'Error: GPU out of memory. Reduce resolution to prevent viewer from crashing.'
            )
            print(f'Error: {e}')
            # del inputs
            torch.cuda.empty_cache()
            time.sleep(0.5)  # sleep to allow buffer to reset

        graph.train()
        outputs = render_thread.vis_outputs
        if outputs is not None:
            colors = graph.colors if hasattr(graph, 'colors') else None
            self._send_output_to_viewer(outputs, colors=colors)
            self._update_viewer_stats(num_rays=n_rays, image_height=image_height, image_width=image_width)
