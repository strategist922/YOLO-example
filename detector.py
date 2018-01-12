# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import cv2 as cv
import argparse

g_tp = 0
g_tn = 0
g_fp = 0
g_fn = 0
precision = lambda tp, fp: 1.0*tp / (tp + fp)
recall = lambda tp, fn: 1.0*tp / (tp + fn)
f1 = lambda p, r: 2.0 * (p * r) / (p + r)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect_cv(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = dn.make_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, 
                 boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res
    
def read_data(fname_meta):
    values = {}
    with open(fname_meta, 'r') as f:
        for line in f:
            tokens = line.strip().split('=')
            if tokens[0] != '':
                values[tokens[0].strip()] = tokens[1].strip()
    return values
    
def points_from_center(xc, yc, w, h):
    p1 = (int(xc - w/2), int(yc - h/2)) 
    p2 = (int(xc + w/2), int(yc + h/2))
    return p1, p2
    
def read_ground_truth(img_name, img_shape, gts):
    gt_name = img_name.split('.')[0] + '.txt'
    #print gt_name
    rts = []
    w, h, _ = img_shape
    with open(gt_name, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            r = map(float, tokens)
            #print r
            rxc = r[1] * w
            ryc = r[2] * h
            rw = r[3] * w
            rh = r[4] * h
            p1, p2 = points_from_center(rxc, ryc, rw, rh)
            rts.append((r[0], (p1, p2)))
    #print rts
    gts[img_name] = rts
    return gts
           
def draw_bbox(img, ret, color):
    '''    rxc = ret[0]
    ryc = ret[1]
    rw = ret[2]
    rh = ret[3]
    p1, p2 = points_from_center(rxc, ryc, rw, rh)
    #print p1, p2
    '''    
    cv.rectangle(img, ret[0], ret[1], color, thickness=2)
    return img  

def draw_bbox_groud_truth(img, r):
    #boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h
    for roi in r:
        #print roi[0]
        #print roi[1]
        color = (255, 0, 0)
        #ret = map(int, roi[1])
        img = draw_bbox(img, roi[1], color)
    return img
    
def draw_bbox_detections(img, r):
    #boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h
    for roi in r:
        #print roi[0]
        #print roi[1]
        if roi[0].lower() == 'bottle':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        #ret = map(int, roi[2])
        img = draw_bbox(img, roi[2], color)
    return img

def convert_detections(r):
    #boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h
    nr = []
    for roi in r:
        p1, p2 = points_from_center(roi[2][0], roi[2][1], roi[2][2], roi[2][3])
        nr.append((roi[0], roi[1], (p1, p2)))
    return nr

def draw_bbox_inter(img, r, color):
    #color = (0, 0, 255)
    print r
    ret = ((r[0][0], r[0][1]), (r[1][0], r[1][1]))
    img = draw_bbox(img, ret, color)
    cv.imshow('image', arr)
    cv.waitKey(0)
    return img
   
def bb_intersection_over_union(boxA1, boxB1):
    boxA = (boxA1[0][0], boxA1[0][1], boxA1[1][0], boxA1[1][1])
    boxB = (boxB1[0][0], boxB1[0][1], boxB1[1][0], boxB1[1][1])
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    if yB - yA <= 0.0 or xB - xA <= 0.0: 
        #print yB - yA, xB - xA
        return 0.0, ((xA, yA), (xB, yB))
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou, ((xA, yA), (xB, yB))
    
def compare_gt_detections(r_gt, r_pred):
    global g_tp, g_fp, g_tn, g_fn
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    found_preds = []
    found_gts = []
    n_gts = len(r_gt)
    for roi_pred in r_pred:
        ret_pred = roi_pred[2]
        max_roi_gt_i = None
        max_inter = None
        max_iou = 0.0
        for i in range(len(r_gt)):
            roi_gt = r_gt[i]
            ret_gt = roi_gt[1]
            iou, inter = bb_intersection_over_union(ret_gt, ret_pred)
            if iou >= max_iou:
                max_iou = iou
                max_roi_gt_i = i
                max_inter = inter
        if max_iou >= .25:
            tp += 1
            found_preds.append(roi_pred)
            found_gts.append(r_gt.pop(max_roi_gt_i))
    fp = len(r_pred) - tp
    fn = n_gts - tp
    print 'TP=%d, FP=%d, FN=%d' % (tp, fp, fn)
    print 'Precision=%.2lf' % (precision(tp, fp))
    print 'Recall=%.2lf' % (recall(tp, fn)) 
    
    #update globals
    g_tp += tp
    g_fp += fp
    g_fn += fn
    return found_gts, found_preds
               
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Darketnet Python Detector CV', 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-w', '--weights', help='path to .weights', 
     default='yolo.weights')
    parser.add_argument('-d', '--data', help='path to .data', dest='fname_data',
     default='cfg/coco_luxoft.data')
    parser.add_argument('-m', '--model', help='path to .cfg', 
     default='cfg/yolo.cfg')
    parser.add_argument('-o', '--outputdir', default='.',
     help='output directory to save detections')
    parser.add_argument('-n', '--nogpu', action='store_true', dest='cpu',
     help='run in cpu mode')
    args = parser.parse_args()
    print args

    data = read_data(args.fname_data)
    if args.cpu:
        dn.set_gpu(0)
    
    net = dn.load_net(args.model, args.weights, 0)
    meta = dn.load_meta(args.fname_data)
    
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    gts = {}
    with open(data['valid'], 'r') as f:
        for img_name in f:
            img_name = img_name.strip()
            print img_name
            #r = dn.detect(net, meta, img_name.strip(), .3)
            arr = cv.imread(img_name)
            im = array_to_image(arr)
            dn.rgbgr_image(im)
            r = detect_cv(net, meta, im, .3)
            r = convert_detections(r)
            #arr = draw_bbox_detections(arr, r)
            #print r
            
            gts = read_ground_truth(img_name, arr.shape, gts)
            #arr = draw_bbox_groud_truth(arr, gts[img_name])
            found_gts, foud_preds = compare_gt_detections(gts[img_name], r)
            arr = draw_bbox_detections(arr, foud_preds)
            arr = draw_bbox_groud_truth(arr, found_gts)
            #cv.imshow('image', arr)
            #cv.waitKey(0)
            cv.imwrite(img_name.split('.')[0] + '_result.jpg', arr)
            #break
        print 'TP=%d, FP=%d, FN=%d' % (g_tp, g_fp, g_fn)
        print 'Precision=%.2lf' % (precision(g_tp, g_fp))
        print 'Recall=%.2lf' % (recall(g_tp, g_fn))
