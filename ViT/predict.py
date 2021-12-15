import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    # 加载测试图片
    img_path = "./test.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 扩展维度
    img = torch.unsqueeze(img, dim=0)

    # 读取分类类别字典
    json_path = './class_indices.json'
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 定义模型实例
    model = create_model(num_classes=5, has_logits=False).to(device)
    # 加载模型权重
    # model_weight_path = "./weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        # 预测类别
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    end_time = time.time()
    print_res = "class:{} prob:{:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{} prob:{:.3}".format(class_indict[str(i)],
                                           predict[i].numpy()))
    plt.show()
    print('total time:{}'.format(end_time-start_time))


if __name__ == '__main__':
    main()
