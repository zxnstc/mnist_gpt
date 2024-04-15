import torch
import matplotlib.pyplot as plt
import click
from pathlib import Path

from model import *
from model import BigramLanguageModel

@click.command()
@click.option("--model_path",type=Path,default='model.pth')
def load_model(model_path:Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load(model_path)
    model=model.to(device)
    model.eval()


@click.command()
@click.option("--num",type=int,default='10')
def get_sample(num:int):
    all_sample = torch.tensor(images, dtype=torch.long)
    sample=all_sample[-num:]
    # 切片之后的sample
    half_sample = sample[:, :392]
    half_sample=half_sample.to(device)

    return half_sample

def show_images(result_tensor,num:int):
    result_reshape=result_tensor.view(-1,28,28)

    # 创建一个网格以显示多个图像
    fig, axes = plt.subplots(nrows=num/5, ncols=5, figsize=(10, 5))

    # 将张量数据转化为numpy后显示
    for i, ax in enumerate(axes.flatten()):
        image = result_reshape[i].cpu().numpy()
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')

    plt.savefig('image.png')



def main():
    load_model()
    half_sample=get_sample()
    result_tensor=model.generate(half_sample,max_new_tokens=392)
    show_images(result_tensor)


if __name__ == "__main__":
    main()