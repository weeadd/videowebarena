from typing import List
import base64 
import json 
from io import BytesIO 
import requests 
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

def remote_caption_image_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    """
    该函数用于生成一个 caption_images 函数，该函数在调用时会将输入图像（及可选的 prompt）
    发送给远程 GPU 服务器，由远程服务器调用 BLIP-2 模型生成描述。

    参数:
      - device, dtype: 为了接口兼容性保留，远程调用时并不使用这两个参数。
      - model_name: 模型名称，目前仅支持 BLIP-2 模型。

    返回:
      - caption_images: 一个函数，该函数接收一个图像列表和可选的 prompt 列表，返回生成的 caption 列表。
    """
    # 目前仅支持 BLIP-2 模型
    if "blip2" not in model_name.lower():
        raise NotImplementedError("目前仅支持 BLIP-2 模型的远程调用")

    # 设定远程 GPU 服务器的 URL，请根据实际部署情况修改此 URL
    remote_url = "http://http://10.245.95.242:5000/api/caption "

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        # 将每张图像转换为 base64 编码的字符串
        encoded_images = []
        for img in images:
            buffered = BytesIO()
            # 这里采用 JPEG 格式，可根据需要修改为其他格式
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(img_str)

        # 构造请求的 payload
        payload = {
            "model_name": model_name,
            "images": encoded_images,
            "max_new_tokens": max_new_tokens,
            "device": str(device), 
            "dtype": str(dtype), 
            # 如果未提供 prompt，则认为是 VQA 模式
            "mode": "vqa" if prompt is None else "captioning"
        }
        if prompt is not None:
            if len(images) != len(prompt):
                raise ValueError("图像和 prompt 数量不匹配，分别为 {} 和 {}".format(len(images), len(prompt)))
            payload["prompt"] = prompt

        # 发送 POST 请求到远程 GPU 服务器
        print("using remote captioning...")
        remote_url = "http://http://10.245.95.242:5000/api/caption "
        response = requests.post(remote_url, json=payload)
        if response.status_code != 200:
            raise Exception("远程调用失败，状态码：{}，响应：{}".format(response.status_code, response.text))

        # 假设远程服务器返回的数据格式为 {"captions": [...]} 
        result = response.json()
        if "captions" not in result:
            raise Exception("远程调用返回的数据格式错误，缺少 'captions' 字段")
        return result["captions"]

    return caption_images



def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )
    captioning_model.to(device)

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        if prompt is None:
            # Perform VQA
            inputs = captioning_processor(
                images=images, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            # Regular captioning. Prompt is a list of strings, one for each image
            assert len(images) == len(
                prompt
            ), "Number of images and prompts must match, got {} and {}".format(
                len(images), len(prompt)
            )
            inputs = captioning_processor(
                images=images, text=prompt, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return captions

    return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
