import torch
import matplotlib.pyplot as plt
import click
from loguru import logger
from pathlib import Path
import os

from model import *
from model import BigramLanguageModel

# python sample.py --model_path --num

def get_image_path(model_path:Path)->Path:
    head, file_name = os.path.split(model_path)
    time_part = os.path.split(head)[1]
    model_part = os.path.split(os.path.split(head)[0])[1]
    tail,_ = os.path.splitext(file_name)
    result = Path(os.path.join(model_part,time_part, tail).replace("/", "_"))
    parent_path=Path("/data/mnist_gpt/images/")
    images_dir=parent_path/result
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir

def load_model(model_path:Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Loading weights")
    state_dict=torch.load(model_path)
    model=BigramLanguageModel()
    model.load_state_dict(state_dict['model_state'])
    model=model.to(device)
    model.eval()
    return model


def get_sample(num:int):
    BOT_tensor = torch.full((num, 1), 256)

    BOT_tensor=BOT_tensor.to(device)
    
    return BOT_tensor

def save_images(result_tensor,num:int,model_path:Path):
    result_reshape=result_tensor[:,1:].view(-1,28,28)

    # 创建一个网格以显示多个图像
    fig, axes = plt.subplots(nrows=int(num/5), ncols=5, figsize=(10, 5))

    # 将张量数据转化为numpy后显示
    for i, ax in enumerate(axes.flatten()):
        image = result_reshape[i].cpu().numpy()
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')
    

    parent_path=get_image_path(model_path)


    plt.savefig(parent_path/f"{num:02d}_{starttime[11:16]}.png")


@click.command()
@click.option("--model_path",type=Path)
@click.option("--num",type=int,default=10)
def main(model_path:Path,num:int):
    model=load_model(model_path)

    BOT_sample=get_sample(num)
    # BOT_sample(num,1)

    logger.info("generating中")
    result_tensor=model.generate(BOT_sample,max_new_tokens=784)
    print(result_tensor.shape)
    save_images(result_tensor,num,model_path)


if __name__ == "__main__":
    main()