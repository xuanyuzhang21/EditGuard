import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from copy import deepcopy
import cv2
from test_gradio import load_image, image_editing

import options.options as option
from utils.JPEG import DiffJPEG
from scipy.io.wavfile import read as wav_read
from scipy.io import wavfile

import os
import math
import argparse
import random
import logging

import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

from utils import util
from data.util import read_img 
from data import create_dataloader, create_dataset
from models import create_model as create_model_editguard
from diffusers import StableDiffusionInpaintPipeline

import base64
import gradio as gr

from diffusers import StableDiffusionInpaintPipeline
from scipy.ndimage import zoom

import matplotlib.pyplot as plt

def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = img_to_base64("../logo.png")

html_content = f"""
<div style='display: flex; align-items: center; justify-content: center; padding: 20px;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 50px; margin-right: 20px;'>
    <strong><font size='8'>EditGuard<font></strong>
</div>
"""

# Examples
examples = [
    ["../dataset/examples/0011.png"],
    ["../dataset/examples/0012.png"],
    ["../dataset/examples/0003.png"],
    ["../dataset/examples/0004.png"],
    ["../dataset/examples/0005.png"],
    ["../dataset/examples/0006.png"],
    ["../dataset/examples/0007.png"],
    ["../dataset/examples/0008.png"],
    ["../dataset/examples/0009.png"],
    ["../dataset/examples/0010.png"],
    ["../dataset/examples/0002.png"],
]

default_example = examples[0]

def hiding(image_input, bit_input, model):

    message = np.array([int(bit_input[i:i+1]) for i in range(0, len(bit_input), 1)])
    message = message - 0.5
    val_data = load_image(image_input, message)
    model.feed_data(val_data)
    container = model.image_hiding()

    from PIL import Image
    image = Image.fromarray(container)
    return container, container

def rand(num_bits=64):
    random_str = ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])
    return random_str

def ImageEdit(img, prompt, model_index):
    image, mask = img["image"], np.float32(img["mask"])

    received_image = image_editing(image, mask, prompt)
    return received_image, received_image, received_image


def imgae_model_select(ckp_index=0):
    # options
    opt = option.parse("options/test_editguard.yml", is_train=True)
    # distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                    map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True
    # create model

    model = create_model_editguard(opt)

    if ckp_index == 0:
        model_pth = '../checkpoints/clean.pth'
    print(model_pth)
    model.load_test(model_pth)
    return model



def Gaussian_image_degradation(image, NL):
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    image = image.unsqueeze(0)
    NL = NL / 255.0
    noise = np.random.normal(0, NL, image.shape)
    torchnoise = torch.from_numpy(noise).float()
    y_forw = image + torchnoise
    y_forw = torch.clamp(y_forw, 0, 1)
    y_forw = y_forw.permute(0, 2, 3, 1)
    y_forw = y_forw.cpu().detach().numpy().squeeze()

    y_forw = (y_forw * 255.0).astype(np.uint8)
    return y_forw, y_forw



def JPEG_image_degradation(image, NL):
    image = image.astype(np.float32)
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    image = image.unsqueeze(0)
    JPEG = DiffJPEG(differentiable=True, quality=int(NL))
    y_forw = JPEG(image)
    y_forw = y_forw.permute(0, 2, 3, 1)
    y_forw = y_forw.cpu().detach().numpy().squeeze()
    y_forw = (y_forw * 255.0).astype(np.uint8)

    return y_forw, y_forw


def revealing(image_edited, input_bit, model_list, model):

    if model_list==0:
        number = 0.2
    else:
        number = 0.2

    container_data = load_image(image_edited) ## load tampered images
    model.feed_data(container_data)
    mask, remesg = model.image_recovery(number)
    mask = Image.fromarray(mask.astype(np.uint8))
    remesg = remesg.cpu().numpy()[0]
    remesg = ''.join([str(int(x)) for x in remesg])
    bit_acc = calculate_similarity_percentage(input_bit, remesg)
    return mask, remesg, bit_acc



def calculate_similarity_percentage(str1, str2):

    if len(str1) == 0:
        return "原始版权水印未知"
    elif len(str1) != len(str2):
        return "输入输出水印长度不同"
    total_length = len(str1)
    same_count = sum(1 for x, y in zip(str1, str2) if x == y)
    similarity_percentage = (same_count / total_length) * 100
    return f"{similarity_percentage}%"



# Description
title = "<center><strong><font size='8'>EditGuard<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title="EditGuard") as demo:
    gr.HTML(html_content)
    model = gr.State(value = None)
    save_h = gr.State(value = None)
    save_w = gr.State(value = None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem('多功能取证水印'):

            DESCRIPTION = """
            ## 使用方法：
            - 上传图像和版权水印（64位比特序列），点击"嵌入水印"按钮，生成带水印的图像。
            - 涂抹要编辑的区域，并使用Inpainting算法编辑图像。
            - 点击"提取"按钮检测篡改区域并输出版权水印。"""
            
            gr.Markdown(DESCRIPTION)
            save_inpainted_image = gr.State(value=None)
            with gr.Column():
                with gr.Row():
                    model_list = gr.Dropdown(label="选择模型", choices=["模型1"], type = 'index')
                    clear_button = gr.Button("清除全部")
                with gr.Box():
                    gr.Markdown("# 1. 嵌入水印")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(source='upload', label="原始图片", interactive=True, type="numpy", value=default_example[0])
                            with gr.Row():
                                bit_input = gr.Textbox(label="输入版权水印（64位比特序列）", placeholder="在这里输入...")
                                rand_bit = gr.Button("🎲 随机生成版权水印")
                            hiding_button = gr.Button("嵌入水印")
                        with gr.Column():
                            image_watermark = gr.Image(source="upload", label="带有水印的图片", interactive=True, type="numpy")


                with gr.Box():
                    gr.Markdown("# 2. 篡改图片")
                    with gr.Row():
                        with gr.Column():
                            image_edit = gr.Image(source='upload',tool="sketch", label="选取篡改区域", interactive=True, type="numpy")
                            inpainting_model_list = gr.Dropdown(label="选择篡改模型", choices=["模型1：SD_inpainting"], type = 'index')
                            text_prompt = gr.Textbox(label="篡改提示词")
                            inpainting_button = gr.Button("篡改图片")
                        with gr.Column():
                            image_edited = gr.Image(source="upload", label="篡改结果", interactive=True, type="numpy")
                

                with gr.Box():
                    gr.Markdown("# 3. 提取水印&篡改区域")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1 = gr.Image(source="upload", label="待提取图片", interactive=True, type="numpy")
                            
                            revealing_button = gr.Button("提取")
                        with gr.Column():
                            edit_mask = gr.Image(source='upload', label="编辑区域蒙版预测", interactive=True, type="numpy")
                            bit_output = gr.Textbox(label="版权水印预测")
                            acc_output = gr.Textbox(label="水印预测准确率")
                
                gr.Examples(
                            examples=examples,
                            inputs=[image_input],
                        )


                model_list.change(
                    imgae_model_select, inputs = [model_list], outputs=[model]
                    )
                hiding_button.click(
                    hiding, inputs=[image_input, bit_input, model], outputs=[image_watermark, image_edit]
                    )
                rand_bit.click(
                    rand, inputs=[], outputs=[bit_input]
                    )


                inpainting_button.click(
                    ImageEdit, inputs = [image_edit, text_prompt, inpainting_model_list], outputs=[image_edited, image_edited_1, save_inpainted_image]
                    )

                revealing_button.click(
                    revealing, inputs=[image_edited_1, bit_input, model_list, model], outputs=[edit_mask, bit_output, acc_output]
                    )

demo.launch(server_name="0.0.0.0", server_port=2002, share=True, favicon_path='../logo.png')