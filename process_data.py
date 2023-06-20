import os
import cv2
import numpy as np
from config import Config
from natsort import natsorted
from glob import glob
import pandas as pd
import math



# CRAFT
# ---------------------------------------------------- #
import time
import torch
import torch.backends.cudnn as cudnn
from craft_model.craft import CRAFT
from torch.autograd import Variable
import craft_model.craft_utils
import craft_model.imgproc
import craft_model.file_utils
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    # resize
    img_resized, target_ratio, size_heatmap = craft_model.imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = craft_model.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_model.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_model.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_model.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = craft_model.imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


# Global definition 
# Load Craft Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = CRAFT()     # initialize

print('Loading weights from checkpoint (' + 'craft_model/craft_mlt_25k.pth' + ')')
net.load_state_dict(copyStateDict(torch.load('craft_model/craft_mlt_25k.pth', map_location=device)))

if torch.cuda.is_available():
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()
# LinkRefiner
refine_net = None

main_result = 'result/'
os.makedirs(main_result, exist_ok=True)


def store(image_folder, result_folder):
    print('check')
    image_list = os.listdir(image_folder)
    for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = craft_model.imgproc.loadImage(os.path.join(image_folder, image_path))

        bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, torch.cuda.is_available(), False, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        craft_model.file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    print()

# ----------------------------------------------------- #

def obtain_transform(im, points):
    x1,y1, x2,y2, x3,y3, x4,y4 = points
    length = round(abs(x2 - x1))
    width = round(abs(y3 - y2)) 

    source_vertices = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
    target_vertices = np.array([[0, 0], [0, length], [width, length], [width, 0]], dtype='float32')
    M = cv2.getPerspectiveTransform(source_vertices, target_vertices)
    
    output_size = (width, length) 
    blank_image = np.zeros((width, length), dtype=np.uint8)
    transformed_polygon = cv2.warpPerspective(im, M, output_size).T

    ret, thresh = cv2.threshold(transformed_polygon, 0, 255, cv2.THRESH_OTSU)
    thresh = 255-thresh

    return thresh

def get_patches(img1, img2):
    path1 = img1
    path2 = img2
    # img1 = cv2.imread(path1, 0)
    # img2 = cv2.imread(path2, 0)
    # print(path1, img1.shape)

    df1 = pd.read_csv(path1.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])
    P = []
    for i in range(len(df1)):
        # df2 = pd.read_csv(path2.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])

        # TODO: Instead of random, pick one with high length/area etc
        p1_points = df1.iloc[i, :].values
        # p2_points = df2.iloc[np.random.randint(0,len(df2)), :].values

        path1 = path1.replace('result', 'dataset/train').replace('res_', '')
        # path2 = path2.replace('result', 'dataset/train').replace('res_', '')
        img1 = cv2.imread(path1, 0)
        # img2 = cv2.imread(path2, 0)

        p1 = obtain_transform(img1, p1_points)
        # p2 = obtain_transform(img2, p2_points)
        P.append(p1)
    
    return P# p1, p2

def prepare_data_pairs(base_dir, patch_size):
    """
    Prepare data pairs for training
    Args:
        base_dir(str): Base directory where train data lies
        patch_size(int): Patch size of pairs used for training
    
        Contains lot of outliers
    """
    # Generate 30 pair per writer for both same and not same
    path = os.path.join(base_dir, 'train')
    writers = natsorted(os.listdir(path))
    if '.DS_Store' in writers:
        writers.remove('.DS_Store')

    counter_1 = 1
    counter_0 = 1
    
    for idx, writer in enumerate(writers):
        # os.system('rm -r result_1/')
        
        result_folder =  main_result + writer + '/'
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
            store(os.path.join(path, writer), result_folder)
        
        same_writer_imgs = glob(os.path.join(result_folder, '*[0-9].jpg'))
    


        writer_patches = []
        for img in same_writer_imgs:
            df = pd.read_csv(img.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])
            P = get_patches(img, img)
            writer_patches.extend(P)

        # Same writer pairs
        c = 0
        for i, p1 in enumerate(writer_patches):
            if c > 100:
                break
            for j, p2 in enumerate(writer_patches):
                if c > 100:
                    break
                if i != j:
                    print(f"Writer: {idx}, Similar :",counter_1)
                    file_name = 'img_' + str(counter_1) + '_1.png' # Label 1 for same writer
                    cv2.imwrite(base_dir + 'data_1/' + file_name, p1)
                    cv2.imwrite(base_dir + 'data_2/' + file_name, p2)
                    counter_1 += 1
                    c += 1
                
        
        # Dissimilar writer pairs
        c = 0 
        while c < 140:
            diff_writer = np.random.choice(writers)
            while diff_writer == writer:
                diff_writer = np.random.choice(writers)
            result_folder =  main_result + diff_writer + '/'
            diff_writer_imgs = glob(os.path.join(result_folder, '*[0-9].jpg'))

            img = np.random.choice(diff_writer_imgs)
            df = pd.read_csv(img.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])
            P = get_patches(img, img)
            
            r = np.random.randint(0, len(writer_patches))
            p1 = writer_patches[r]
            c2 = 0
            for p2 in P:
                if c2 > 2:
                    break
                print(f"Writer: {idx}, Dissimilar :",counter_0)
                file_name = 'img_' + str(counter_0) + '_0.png' # Label 0 for same writer
                cv2.imwrite(base_dir + 'data_1/' + file_name, p1)
                cv2.imwrite(base_dir + 'data_2/' + file_name, p2)
                counter_0 += 1
                c2 += 1
                c += 1

if __name__ == '__main__':
    cfg = Config().parse()
    base_dir = cfg.data_path
    patch_size = cfg.patch_size

    os.makedirs(base_dir + 'data_1/', exist_ok=True)
    os.makedirs(base_dir + 'data_2/', exist_ok=True)

    os.system('rm ' + base_dir + 'data_1')
    os.system('rm ' + base_dir + 'data_2')


    # Prepare patches
    prepare_data_pairs(base_dir, patch_size)
    

