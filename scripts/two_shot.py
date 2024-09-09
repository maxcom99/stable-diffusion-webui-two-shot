import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from scripts.sketch_helper import get_high_freq_colors, color_quantization, create_binary_matrix_base64, create_binary_mask
import numpy as np
import cv2
from modules import devices, script_callbacks
import modules.scripts as scripts
import gradio as gr
from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoised
from modules.processing import StableDiffusionProcessing

MAX_COLORS = 12
switch_values_symbol = '\U000021C5' # ⇅

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

# abstract base class for filters
from abc import ABC, abstractmethod

class Filter(ABC):
    @abstractmethod
    def create_tensor(self):
        pass

@dataclass
class Division:
    y: float
    x: float

@dataclass
class Position:
    y: float
    x: float
    ey: float
    ex: float

class RectFilter(Filter):
    def __init__(self, division: Division, position: Position, weight: float):
        self.division = division
        self.position = position
        self.weight = weight

    def create_tensor(self, num_channels: int, height_b: int, width_b: int) -> torch.Tensor:
        x = torch.zeros(num_channels, height_b, width_b).to(devices.device)
        division_height = height_b / self.division.y
        division_width = width_b / self.division.x
        y1 = int(division_height * self.position.y)
        y2 = int(division_height * self.position.ey)
        x1 = int(division_width * self.position.x)
        x2 = int(division_width * self.position.ex)
        x[:, y1:y2, x1:x2] = self.weight
        return x

class MaskFilter:
    def __init__(self, binary_mask: np.array = None, weight: float = None, float_mask: np.array = None):
        if float_mask is None:
            self.mask = binary_mask.astype(np.float32) * weight
        elif binary_mask is None and weight is None:
            self.mask = float_mask
        else:
            raise ValueError('Either float_mask or binary_mask and weight must be provided')
        self.tensor_mask = torch.tensor(self.mask).to(devices.device)

    def create_tensor(self, num_channels: int, height_b: int, width_b: int) -> torch.Tensor:
        mask = torch.nn.functional.interpolate(self.tensor_mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='nearest-exact').squeeze(0).squeeze(0)
        mask = mask.unsqueeze(0).repeat(num_channels, 1, 1)
        return mask

class PastePromptTextboxTracker:
    def __init__(self):
        self.scripts = []
        return

    def set_script(self, script):
        self.scripts.append(script)

    def on_after_component_callback(self, component, **_kwargs):
        if not self.scripts:
            return
        if type(component) is gr.State:
            return
        script = None
        if type(component) is gr.Textbox and component.elem_id == 'txt2img_prompt':
            # select corresponding script
            script = next(x for x in self.scripts if x.is_txt2img)
            self.scripts.remove(script)
        if type(component) is gr.Textbox and component.elem_id == 'img2img_prompt':
            # select corresponding script
            script = next(x for x in self.scripts if x.is_img2img)
            self.scripts.remove(script)
        if script is None:
            return
        script.target_paste_prompt = component

prompt_textbox_tracker = PastePromptTextboxTracker()

class Script(scripts.Script):
    def __init__(self):
        self.ui_root = None
        self.num_batches: int = 0
        self.end_at_step: int = 20
        self.filters: List[Filter] = []
        self.debug: bool = False
        self.selected_twoshot_tab = 0
        self.ndmasks = []
        self.area_colors = []
        self.mask_denoise = False
        prompt_textbox_tracker.set_script(self)
        self.target_paste_prompt = None

    def title(self):
        return "Latent Couple extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def create_rect_filters_from_ui_params(self, raw_divisions: str, raw_positions: str, raw_weights: str):
        divisions = []
        for division in raw_divisions.split(','):
            y, x = division.split(':')
            divisions.append(Division(float(y), float(x)))

        def start_and_end_position(raw: str):
            nums = [float(num) for num in raw.split('-')]
            if len(nums) == 1:
                return nums[0], nums[0] + 1.0
            else:
                return nums[0], nums[1]

        positions = []
        for position in raw_positions.split(','):
            y, x = position.split(':')
            y1, y2 = start_and_end_position(y)
            x1, x2 = start_and_end_position(x)
            positions.append(Position(y1, x1, y2, x2))

        weights = []
        for w in raw_weights.split(','):
            weights.append(float(w))

        # todo: assert len
        return [RectFilter(division, position, weight) for division, position, weight in zip(divisions, positions, weights)]

    def create_mask_filters_from_ui_params(self, raw_divisions: str, raw_positions: str, raw_weights: str):
        divisions = []
        for division in raw_divisions.split(','):
            y, x = division.split(':')
            divisions.append(Division(float(y), float(x)))

        def start_and_end_position(raw: str):
            nums = [float(num) for num in raw.split('-')]
            if len(nums) == 1:
                return nums[0], nums[0] + 1.0
            else:
                return nums[0], nums[1]

        positions = []
        for position in raw_positions.split(','):
            y, x = position.split(':')
            y1, y2 = start_and_end_position(y)
            x1, x2 = start_and_end_position(x)
            positions.append(Position(y1, x1, y2, x2))

        weights = []
        for w in raw_weights.split(','):
            weights.append(float(w))

        # todo: assert len
        return [Filter(division, position, weight) for division, position, weight in zip(divisions, positions, weights)]

    def do_visualize(self, raw_divisions: str, raw_positions: str, raw_weights: str):
        self.filters = self.create_rect_filters_from_ui_params(raw_divisions, raw_positions, raw_weights)
        return [f.create_tensor(1, 128, 128).squeeze(dim=0).cpu().numpy() for f in self.filters]

    def do_apply(self, extra_generation_params: str):
        # parse "Latent Couple" extra_generation_params
        raw_params = {}
        for assignment in extra_generation_params.split(' '):
            pair = assignment.split('=', 1)
            if len(pair) != 2:
                continue
            raw_params[pair[0]] = pair[1]
        return raw_params.get('divisions', '1:1,1:2,1:2'), raw_params.get('positions', '0:0,0:0,0:1'), raw_params.get('weights', '0.2,0.8,0.8'), int(raw_params.get('step', '20'))

    def ui(self, is_img2img):
        process_script_params = []
        id_part = "img2img" if is_img2img else "txt2img"
        canvas_html = ""

        def create_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

        def process_sketch(img_arr, input_binary_matrixes):
            input_binary_matrixes.clear()
            im2arr = img_arr
            sketch_colors, color_counts = np.unique(im2arr.reshape(-1, im2arr.shape[2]), axis=0, return_counts=True)
            colors_fixed = []
            edge_color_correction_arr = []
            for sketch_color_idx, color in enumerate(sketch_colors[:-1]):  # exclude white
                if color_counts[sketch_color_idx] < im2arr.shape[0] * im2arr.shape[1] * 0.002:
                    edge_color_correction_arr.append(sketch_color_idx)
            edge_fix_dict = {}
            area_colors = np.delete(sketch_colors, edge_color_correction_arr, axis=0)
            if self.mask_denoise:
                for edge_color_idx in edge_color_correction_arr:
                    edge_color = sketch_colors[edge_color_idx]
                    color_distances = np.linalg.norm(area_colors - edge_color, axis=1)
                    nearest_index = np.argmin(color_distances)
                    nearest_color = area_colors[nearest_index]
                    edge_fix_dict[edge_color_idx] = nearest_color
                    cur_color_mask = np.all(im2arr == edge_color, axis=2)
                    im2arr[cur_color_mask] = nearest_color
                sketch_colors, color_counts = np.unique(im2arr.reshape(-1, im2arr.shape[2]), axis=0, return_counts=True)
                area_colors = sketch_colors
            area_color_maps = []
            self.ndmasks = []
            self.area_colors = area_colors
            for color in area_colors:
                r, g, b = color
                mask, binary_matrix = create_binary_matrix_base64(im2arr, color)
                self.ndmasks.append(mask)
                input_binary_matrixes.append(binary_matrix)
                colors_fixed.append(gr.update(value=f'rgb({r},{g},{b})'))
            return colors_fixed + [gr.update(visible=True)] * (len(area_colors)) + [gr.update(visible=False)] * (MAX_COLORS - len(area_colors))

        with gr.Group() as group_two_shot_root:
            binary_matrixes = gr.State([])
            with gr.Accordion("Latent Couple", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Tabs(elem_id="script_twoshot_tabs") as twoshot_tabs:
                    with gr.TabItem("Mask", elem_id="tab_twoshot_mask") as twoshot_tab_mask:
                        canvas_data = gr.JSON(value={}, visible=False)
                        mask_denoise_checkbox = gr.Checkbox(value=False, label="Denoise Mask")
                        def update_mask_denoise_flag(flag):
                            self.mask_denoise = flag
                        mask_denoise_checkbox.change(fn=update_mask_denoise_flag, inputs=[mask_denoise_checkbox], outputs=None)
                        canvas_image = gr.Image(source='upload', mirror_webcam=False, type='numpy', tool='color-sketch',
                                                elem_id='twoshot_canvas_sketch', interactive=True, height=480)
                        button_run = gr.Button("I've finished my sketch", elem_id="main_button", interactive=True)
                        prompts = []
                        colors = []
                        color_row = [None] * MAX_COLORS
                        with gr.Column(visible=False) as post_sketch:
                            with gr.Row(visible=False) as alpha_mask_row:
                                with gr.Box(elem_id="alpha_mask"):
                                    alpha_color = gr.HTML(
                                        '<div class="alpha-mask-item" style="background-color: black"></div>')
                            general_prompt = gr.Textbox(label="General Prompt")
                            alpha_blend = gr.Slider(label="Alpha Blend", minimum=0.0, maximum=1.0, value=0.2, step=0.01, interactive=True)
                            with gr.Row():
                                canvas_swap_res = gr.Button(value=switch_values_symbol, variant="tool")
                                canvas_clear = gr.Button(value='\U0001F5D1', variant="tool")  # 🗑
                            with gr.Row():
                                for i in range(MAX_COLORS):
                                    with gr.Column():
                                        color_row[i] = gr.ColorPicker(label=f"Color {i + 1}", interactive=False, visible=False)
                                        prompts.append(gr.Textbox(label=f"Prompt {i + 1}", visible=False, elem_id=f"two_shot_prompt_{i}"))
                                        colors.append(gr.Slider(label=f"Weight {i + 1}", minimum=0.0, maximum=1.0, value=1.0, step=0.01, visible=False))

                        def swap_resolution():
                            return gr.Image.update(height=480 if canvas_image.height == 640 else 640, width=480 if canvas_image.width == 640 else 640)

                        canvas_swap_res.click(fn=swap_resolution, inputs=None, outputs=[canvas_image])

                        def clear_canvas():
                            return gr.Image.update(value=None)

                        canvas_clear.click(fn=clear_canvas, inputs=None, outputs=[canvas_image])

                        button_run.click(
                            fn=process_sketch,
                            inputs=[
                                canvas_image,
                                binary_matrixes,
                            ],
                            outputs=color_row + prompts + colors + [post_sketch]
                        )

                    with gr.TabItem("Manual", elem_id="tab_twoshot_manual") as twoshot_tab_manual:
                        divisions = gr.Textbox(label="Divisions", value="1:1,1:2,1:2")
                        positions = gr.Textbox(label="Positions", value="0:0,0:0,0:1")
                        weights = gr.Textbox(label="Weights", value="0.2,0.8,0.8")
                        end_at_step = gr.Slider(minimum=0, maximum=100, step=1, label="End at step", value=20)
                        visualize = gr.Button(value="Visualize")
                        gallery = gr.Gallery(label="Visualization")
                        visualize.click(
                            fn=self.do_visualize,
                            inputs=[
                                divisions,
                                positions,
                                weights,
                            ],
                            outputs=[gallery]
                        )

                    def update_selected_twoshot_tab(selected_tab):
                        self.selected_twoshot_tab = selected_tab

                    twoshot_tabs.select(fn=update_selected_twoshot_tab, inputs=[twoshot_tabs])

                    def on_ui_settings():
                        section = ('latent-couple', "Latent Couple")
                        shared.opts.add_option("latent_couple_debug", shared.OptionInfo(False, "Debug mode", section=section))

                    script_callbacks.on_ui_settings(on_ui_settings)

                    def on_cfg_denoised(params: CFGDenoisedParams):
                        if not enabled.value:
                            return
                        if self.selected_twoshot_tab == 0:
                            binary_matrixes = binary_matrixes.value
                            if len(binary_matrixes) == 0:
                                return
                            self.filters = []
                            for idx, binary_matrix in enumerate(binary_matrixes):
                                mask = create_binary_mask(binary_matrix)
                                self.filters.append(MaskFilter(float_mask=mask))
                        else:
                            self.filters = self.create_rect_filters_from_ui_params(divisions.value, positions.value, weights.value)
                        self.end_at_step = int(end_at_step.value)
                        x = params.x
                        if params.sampling_step >= self.end_at_step:
                            return
                        for f in self.filters:
                            x += f.create_tensor(x.shape[1], x.shape[2], x.shape[3])
                        params.x = x

                    script_callbacks.on_cfg_denoised(on_cfg_denoised)

                    def process(p: StableDiffusionProcessing, *args):
                        if not enabled.value:
                            return
                        if self.selected_twoshot_tab == 0:
                            binary_matrixes = binary_matrixes.value
                            if len(binary_matrixes) == 0:
                                return
                            self.filters = []
                            for idx, binary_matrix in enumerate(binary_matrixes):
                                mask = create_binary_mask(binary_matrix)
                                self.filters.append(MaskFilter(float_mask=mask))
                        else:
                            self.filters = self.create_rect_filters_from_ui_params(divisions.value, positions.value, weights.value)
                        self.end_at_step = int(end_at_step.value)
                        extra_generation_params = {
                            "Latent Couple": f"divisions={divisions.value} positions={positions.value} weights={weights.value} step={self.end_at_step}"
                        }
                        p.extra_generation_params.update(extra_generation_params)

                    return [enabled, divisions, positions, weights, end_at_step]
