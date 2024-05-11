import torch
import matplotlib.pyplot as plt
import click
from loguru import logger
from pathlib import Path
import time

from model_condition import *
from model_condition import BigramLanguageModel
from utils import get_exp_id

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
condition_images_dict = {i: [] for i in range(10)}

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
    state_dict = torch.load(model_path)

    # torch.save(state_dict,'new.pkl')
    model=BigramLanguageModel()
    model.load_state_dict(state_dict['model_state'])
    model=model.to(device)
    
    model.eval()
    return model

def save_images(max_images_per_key:int,model_path:Path):
    # 创建一个网格以显示多个图像
    fig = plt.figure(figsize=(max_images_per_key, 10))

    for condition,images_list in condition_images_dict.items():
        # 对于每一个图片.indices:10
        for j, img in enumerate(images_list[:max_images_per_key]):
            # 创建一个subplot在适当的位置
            ax = fig.add_subplot(10, 10, condition*max_images_per_key + j +1 )
            ax.axis('off')
            # 显示图片
            
            img=img.narrow(0,1,784)
            print("img",img.shape)

            img=img.view(28,28).cpu().numpy()
            ax.imshow(img)
            # 移除坐标轴
            ax.axis('off')

        # 在键对应的第一个图像上添加键作为标题
        # ax = fig.add_subplot(10, max_images_per_key, condition*max_images_per_key + 1)
        ax.set_title(condition)

    parent_path=get_image_path(model_path)
    plt.savefig(parent_path/f"{max_images_per_key:02d}_{starttime[11:16]}.png")
        
@click.command()
@click.option("--model_path",type=Path)
@click.option("--num",type=int,default=10)
@click.option("--tem",type=float,default=1.0)
@click.option("--topk",type=int,default=None)
def main(model_path:Path,num:int,tem:float,topk:int):
    model=load_model(model_path)
    for condition in range(10):
        condition_tensor=torch.tensor([condition+256])
        # block_size_tensor=torch.zeros(block_size)
        # single_tensor = torch.cat([condition_tensor, block_size_tensor], dim=0)
        tensor_list=[condition_tensor for _ in range(num)]
        input_tensor=torch.stack(tensor_list,dim=0)
        input_tensor = input_tensor.long().to(device)


        print("input",input_tensor.shape)

        result_tensor=model.generate(input_tensor,max_new_tokens=784,temperature=tem,top_k=topk)
        print("result",result_tensor.shape)
        condition_images_dict[condition]=result_tensor
    
    logger.info("Generated Done")
    logger.info("Saving") 

    save_images(num,model_path)

if __name__ == "__main__":
    main()
    


# main(load_model('/data/mnist_gpt/checkpoints/condition_model.pth'))
# @click.command()
# @click.option("--model_path",type=Path)
# @click.option("--num",type=int,default=10)
# def main(model_path:Path,condition:int,num:int):
#     model=load_model(model_path)
