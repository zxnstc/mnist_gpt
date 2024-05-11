import torch
import matplotlib.pyplot as plt
import click
from loguru import logger
from pathlib import Path
import os
import pickle

from model import *
from model import BigramLanguageModel

# python sample.py --model_path --num

def load_data():
    def load_dict_from_file(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    data_dict = load_dict_from_file('/data/mnist_gpt/data/test_data_sorted.pkl')
    sample_list=[]
    for label,image_list in data_dict.items():
        for i in range(10):
            image=image_list[i]
            image[0]=256
            sample_list.append(image)

    result_tensor=torch.stack(sample_list)
    print(result_tensor.shape)
    return result_tensor
    
            

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
    # all_sample = torch.tensor(images, dtype=torch.long)
    sample=load_data() #torch.Size([100, 785])
    # 切片之后的sample
    half_sample = sample[:, :588]
    # BOT_sample=torch.full((half_sample.size(0), 1), 256)
    # half_sample=torch.cat((BOT_sample,half_sample),dim=1)
    print("half_s",half_sample.shape,half_sample[0][0])
    half_sample=half_sample.to(device)

    return half_sample

def save_images(result_tensor,num:int,model_path:Path):
    result_reshape=result_tensor[:,1:].view(-1,28,28)

    # 创建一个网格以显示多个图像
    fig, axes = plt.subplots(nrows=int(num/10), ncols=10, figsize=(10, 10))

    # 将张量数据转化为numpy后显示
    for i, ax in enumerate(axes.flatten()):
        image = result_reshape[i].cpu().numpy()
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')
    

    parent_path=get_image_path(model_path)


    plt.savefig(parent_path/f"{num:02d}_{starttime[11:16]}.png")


@click.command()
@click.option("--model_path",type=Path)
@click.option("--num",type=int,default=100)
def main(model_path:Path,num:int):
    model=load_model(model_path)

    half_sample=get_sample(num)

    logger.info("generating中")
    result_tensor=model.generate(half_sample,max_new_tokens=197)
    save_images(result_tensor,num,model_path)


if __name__ == "__main__":
    main()