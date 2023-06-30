import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from config import Config
import matplotlib.pyplot as plt
from scipy import stats as st
from torch.utils.data import DataLoader
from model.writerModel import BaseModel
from load_data import Read_patch, custom_collate
from sklearn.metrics import f1_score, roc_auc_score



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
device = torch.device('mps')

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
    # print()

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
    path1 = img1
    path2 = img2
    df1 = pd.read_csv(path1.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])
    df2 = pd.read_csv(path2.split('.')[0] + '.txt', names=['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4'])

    P1 = []
    P2 = []
    for i in range(len(df1)):
        p1_points = df1.iloc[np.random.randint(0,len(df1)), :].values
        path1 = path1.replace('all_result_test', f'{cfg.test_path}').replace('res_', '')
        img1 = cv2.imread(path1, 0)       
        p1 = obtain_transform(img1, p1_points)
        P1.append(p1)

    for i in range(len(df2)):
        p2_points = df2.iloc[np.random.randint(0,len(df2)), :].values
        path2 = path2.replace('all_result_test', f'{cfg.test_path}').replace('res_', '')
        img2 = cv2.imread(path2, 0)
        p2 = obtain_transform(img2, p2_points)
        P2.append(p2)
    
    if len(P1) == 0 or len(P2) == 0:
        return 1

    counter = 1
    for p1 in P1:
        for p2 in P2:
            cv2.imwrite(f'result_test/data_1/img_{counter}_{label}.png', p1)
            cv2.imwrite(f'result_test/data_2/img_{counter}_{label}.png', p2)
            counter += 1
    # print(len(P1), len(P2), counter-1)
    return 0
    
def custom_sigmoid(z, threshold):
    denominator = 1 + np.exp( 10 * (threshold - z))
    return 1 / denominator

def predict(model, test_loader):
    threshold = 2.0
    pred_label = []
    for mask, x, y, indices in test_loader:
        x = x/255
        mask, x, y = mask.to(device), x.to(device), y.to(device)
        pred = model(x, mask, indices)
        D = (pred[:, 0, :] - pred[:, 1, :]).pow(2).sum(1).sqrt()

        pred_label.append(D.cpu().detach().numpy()[0])   

    mode = np.mean(np.array(pred_label))
    proba = custom_sigmoid(mode, threshold)
    
    if mode > threshold:
        return 1, proba
    else:
        return 0, proba



def create_loader():
    cfg = Config().parse()
    dataset = Read_patch('result_test', cfg.patch_size)
    test_loader = DataLoader(dataset, collate_fn=custom_collate, batch_size=cfg.batch_size)
    return test_loader

if __name__ == '__main__':
    inspect = False
    cfg = Config().parse()
    csv_path = cfg.csv_path
    test_path = cfg.test_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps')
    model = torch.load('model.pth', map_location=device)
    model.bs = cfg.batch_size
    df = pd.read_csv(csv_path)
    prob = []
    pred_labels = []
    true_label = []
    for i in range(len(df)):
        label = 0 #int(df.iloc[i, 2])
        path1 = os.path.join(test_path, df.iloc[i,0])
        path2 = os.path.join(test_path, df.iloc[i,1])

        os.system('rm -r result_test')
        os.makedirs('result_test', exist_ok=True)
        os.makedirs('result_test/data_1', exist_ok=True)
        os.makedirs('result_test/data_2', exist_ok=True)
        
        path1 = os.path.join('all_result_test', f'res_{df.iloc[i,0]}')
        path2 = os.path.join('all_result_test', f'res_{df.iloc[i,1]}')

        flag = get_save_patches(path1, path2, label)

        if flag:
            proba = np.random.rand()
            if proba >= 0.5:
                pred_label=1
            else:
                pred_label=0
        else:
            test_loader = create_loader()
            pred_label, proba = predict(model, test_loader)
        
        prob.append(proba)
        pred_labels.append(pred_label)
        # true_label.append(label)
        # f1 = f1_score(y_pred=np.array(pred_labels), y_true= np.array(true_label))
        # if i > 20:
        #     auc = roc_auc_score(y_score=np.array(prob), y_true=np.array(true_label))
        # else:
        #     auc = None

        print(f'Iteration: {i+1}, Prediction: {pred_label}')#, Actual: {label}, F1 Score: {f1}, AUC Score: {auc}')
        # if (label != pred_label) and inspect:
        #     plt.figure(figsize=(15,5))
        #     img1 = cv2.imread(path1, 0)
        #     img2 = cv2.imread(path2, 0)
        #     # print(files[i][0].shape)
        #     plt.subplot(1,2,1)
        #     plt.imshow(img1, cmap='gray')
        #     plt.axis('off')
        #     plt.subplot(1,2,2)
        #     plt.imshow(img2, cmap='gray')
        #     plt.axis('off')
        #     plt.show()
            
        print()
        print()
        # break
    prob = np.array(prob)
    pred_labels = np.array(pred_labels)
    # pred_labels =  np.where(prob >= 0.5, 1, 0)

    df['proba'] = prob 
    df['label'] = pred_labels 

    new_df = pd.DataFrame()
    new_df['id'] = df['img1_name'] + '_' + df['img2_name']
    new_df['proba'] = df['proba']
    new_df.to_csv('Single Vision_02.csv', index=False)
        
