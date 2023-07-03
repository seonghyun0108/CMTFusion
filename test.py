import os
from model import CMTFusion
import torch
from torchvision.utils import save_image
import utils
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./fusion_outputs/', help='path of fused image')
    parser.add_argument('--test_images', type=str, default='./test_image_roadscene/', help='path of source image')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    parser.add_argument('--weights', type=str, default='0', help='dataset name')
    args = parser.parse_args()

    if os.path.exists(args.out_path) is False:
        os.mkdir(args.out_path)

    # device setting for gpu users
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    fusion_model = torch.nn.DataParallel(CMTFusion(), device_ids=[0])
    fusion_model.load_state_dict(
        torch.load("saved_models/%s/model_fusion%s.pth" % (args.dataset_name, args.weights)))
    print("===>Testing using weights: ", args.weights)
    fusion_model.cuda()
    fusion_model.eval()

    with torch.no_grad():
        for i in range(os.listdir(args.test_images)):
            index = i + 1
            infrared_path = args.test_images + 'IR' + str(index) + '.png'
            visible_path = args.test_images + 'VIS' + str(index) + '.png'
            print(index)
            if os.path.isfile(infrared_path):
                real_ir_imgs = utils.get_test_images(infrared_path, height=None, width=None)
                real_rgb_imgs = utils.get_test_images(visible_path, height=None, width=None)

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                fused, fused2, fused3 = fusion_model(real_rgb_imgs.cuda(), real_ir_imgs.cuda())
                # # save images
                save_image(fused, "fusion_outputs/%d.png" % index, normalize=True)

    print('Done......')


if __name__ == '__main__':
    run()