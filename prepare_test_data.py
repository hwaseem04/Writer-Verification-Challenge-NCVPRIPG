import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from config import Config
from torch.utils.data import DataLoader
from load_data import Read_patch, custom_collate



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


def store(image_path, result_folder):

    image = craft_model.imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, torch.cuda.is_available(), False, refine_net)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder  + filename + '_mask.jpg'
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

def get_save_patches(img1, img2, label):
    global counter
    path1 = img1
    path2 = img2

    df1 = pd.read_csv(path1.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])
    df2 = pd.read_csv(path2.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])

    P1 = []
    P2 = []
    for i in range(len(df1)):
        p1_points = df1.iloc[np.random.randint(0,len(df1)), :].values
        path1 = path1.replace('all_result_test', f'{cfg.data_path}val').replace('res_', '')
        img1 = cv2.imread(path1, 0)       
        p1 = obtain_transform(img1, p1_points)
        P1.append(p1)

    for i in range(len(df2)):
        p2_points = df2.iloc[np.random.randint(0,len(df2)), :].values
        path2 = path2.replace('all_result_test', f'{cfg.data_path}val').replace('res_', '')
        img2 = cv2.imread(path2, 0)
        p2 = obtain_transform(img2, p2_points)
        P2.append(p2)
    
    if len(P1) == 0 or len(P2) == 0:
        return 1
    
    for p1 in P1:
        for p2 in P2:
            cv2.imwrite(f'result_test/data_1/img_{counter}_{label}.png', p1)
            cv2.imwrite(f'result_test/data_2/img_{counter}_{label}.png', p2)
            counter += 1
    # print(len(P1), len(P2), counter-1)
    return 0

def predict(model, test_loader):
    proba = []
    for mask, x, y in test_loader:
        pred = torch.sigmoid(model(x)[:,0])
        proba.extend(pred.cpu().tolist())
    proba = torch.tensor(proba)

    return torch.mean(proba).detach().numpy()


def create_loader():
    cfg = Config().parse()
    dataset = Read_patch('result_test', cfg.patch_size)
    test_loader = DataLoader(dataset, collate_fn=custom_collate, batch_size=cfg.batch_size, shuffle=True)
    return test_loader

counter = 1
if __name__ == '__main__':
    cfg = Config().parse()
    csv_path = cfg.csv_path
    test_path = cfg.test_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    model = torch.load('model.pth', map_location=device)

    df = pd.read_csv(csv_path)
    prob = []
    pred_labels = []

    os.makedirs('all_result_test', exist_ok=True)
    
    print(df)
    for i in range(len(df)):
        print(i+1)
        path1 = os.path.join(test_path, df.iloc[i,0])
        path2 = os.path.join(test_path, df.iloc[i,1])

        store(path1, 'all_result_test/')
        store(path2, 'all_result_test/')
