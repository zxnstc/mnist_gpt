import torch
import matplotlib.pyplot as plt
import click
from loguru import logger
from pathlib import Path

from model import *
from model import BigramLanguageModel

# python sample.py --model_path --num


def load_model(model_path:Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Loading weights")
    model = torch.load(model_path)
    model=model.to(device)
    model.eval()
    return model


def get_sample(num:int):
    all_sample = torch.tensor(images, dtype=torch.long)
    sample=all_sample[-num:]
    # 切片之后的sample
    half_sample = sample[:, :392]
    half_sample=half_sample.to(device)

    return half_sample

def save_images(result_tensor,num:int):
    print("begin")
    result_reshape=result_tensor.view(-1,28,28)

    # 创建一个网格以显示多个图像
    fig, axes = plt.subplots(nrows=int(num/5), ncols=5, figsize=(10, 5))

    # 将张量数据转化为numpy后显示
    for i, ax in enumerate(axes.flatten()):
        image = result_reshape[i].cpu().numpy()
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')

    plt.savefig('image.png')


@click.command()
@click.option("--model_path",type=Path,default='model.pth')
@click.option("--num",type=int,default=10)
def main(model_path:Path,num:int):
    model=load_model(model_path)

    half_sample=get_sample(num)

    logger.info("generating中")
    result_tensor=model.generate(half_sample,max_new_tokens=392)
    save_images(result_tensor,num)


if __name__ == "__main__":
    main()