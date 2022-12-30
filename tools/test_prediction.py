import os
import glob
import argparse
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("Test Prediction Error.")
    parser.add_argument("--txt_path", type=str, default="", help="path to tracking result path in MOTChallenge format")
    parser.add_argument("--pred_path", type=str, default="", help="path to tracking result path in Prediction format")
    parser.add_argument("--pred_frame_num", type=int, default=10, help="predicted future after n frame")
    parser.add_argument("--unit_test", action="store_true", default=False, help="test with single txt file")
    
    return parser

def IoU_error(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0]-boxA[2]*0.5, boxB[0]-boxB[2]*0.5)
    yA = max(boxA[1]-boxA[3]*0.5, boxB[1]-boxB[3]*0.5)
    xB = min(boxA[0]+boxA[2]*0.5, boxB[0]+boxB[2]*0.5)
    yB = min(boxA[1]+boxA[3]*0.5, boxB[1]+boxB[3]*0.5)
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def prediction_error(txt_path, pred_path):
    if args.unit_test:
        seq_txts = [txt_path]
        seq_preds = [pred_path]
    else:
        seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
        seq_preds = sorted(glob.glob(os.path.join(pred_path, '*.txt')))
    
    # if len(seq_txts) != len(seq_preds):
    #     print("Number of txts", len(seq_txts), "is Not Equal to Number of prediction txts", len(seq_preds))
    #     return
    
    Final_loss = 0.0
    Final_IoU = 0.0
    Total_num = 0
    print("File, Total, Success, Failure, mse, mIoU")
    for i in range(len(seq_preds)):
        seq_name = seq_txts[i].split('/')[-1]
        # print(seq_name)
        txt_data = np.loadtxt(seq_txts[i], dtype=np.float64, delimiter=',')
        pred_data = np.loadtxt(seq_preds[i], dtype=np.float64, delimiter=',')
        
        min_id = max(int(np.min(txt_data[:, 1])), int(np.min(pred_data[:, 1])))
        max_id = min(int(np.max(txt_data[:, 1])), int(np.max(pred_data[:, 1])))
        
        txt_mse_loss = 0.0
        txt_iou_loss = 0.0
        total_txt = 0
        for track_id in range(min_id, max_id+1):
            txt_data_index = (txt_data[:, 1] == track_id)
            pred_data_index = (pred_data[:, 1] == track_id)
            
            txt_tracklet = txt_data[txt_data_index]
            pred_tracklet = pred_data[pred_data_index]
            
            if txt_tracklet.shape[0] == 0 or pred_tracklet.shape[0] == 0:
                continue
            
            n_frames = txt_tracklet.shape[0]
            if n_frames > max(args.pred_frame_num, 25):
                id_loss = 0.0
                id_iou = 0.0
                id_num = 0
                for k in range(args.pred_frame_num, n_frames):
                    pred_frame_idx = np.where(pred_tracklet[:,0] == k)
                    txt_frame_idx = np.where(txt_tracklet[:,0] == k)
                    if np.shape(pred_frame_idx)[1] != 0 and np.shape(txt_frame_idx)[1] != 0:
                        # print(k, pred_frame_idx, txt_frame_idx)
                        # print(txt_tracklet[txt_frame_idx[0][0], 0:2], pred_tracklet[pred_frame_idx[0][0], 0:2])
                        x, y, w, h = txt_tracklet[txt_frame_idx[0][0], 2:6]
                        xp,yp,wp,hp = pred_tracklet[pred_frame_idx[0][0], 2:6]
                        
                        # IoU
                        iou = IoU_error(txt_tracklet[txt_frame_idx[0][0], 2:6], pred_tracklet[pred_frame_idx[0][0], 2:6])
                        id_iou += iou
                        
                        # distance of predicted center
                        d = np.sqrt((x-xp)*(x-xp) + (y-yp)*(y-yp))                    
                        # print(d)
                        id_loss += d
                        
                        id_num += 1
                
                if id_num > 0:
                    id_loss = id_loss / float(id_num)
                    id_iou = id_iou / float(id_num)
                    txt_mse_loss += id_loss
                    txt_iou_loss += id_iou
                    total_txt += 1
                    # print("Id", track_id, "Loss", id_loss, "IoU", id_iou, sep=",")
                # else:
                #     print("Id", track_id, "Prediction Fail.")
                
        # print("Prediction Total/Success/Failure:", max_id-min_id, total_txt, max_id-min_id-total_txt, sep=",")
        # print("Average Loss:", txt_mse_loss / total_txt, "IoU:", txt_iou_loss / total_txt, sep=",")
        Final_loss += txt_mse_loss
        Final_IoU += txt_iou_loss
        Total_num += total_txt
        print(seq_name, max_id-min_id, total_txt, max_id-min_id-total_txt, txt_mse_loss / total_txt, txt_iou_loss / total_txt, sep=',')
        
    print(",,,Total Average Loss,", Final_loss / Total_num, ",", Final_IoU / Total_num)
    # print(",,,Total Average IoU,", )

if __name__ == '__main__':
    args = make_parser().parse_args()
    
    if not os.path.exists(args.txt_path):
        print("txt_path Not Found.")
        exit(1)
    if not os.path.exists(args.pred_path):
        print("pred_path Not Found")
        exit(1)
    
    prediction_error(args.txt_path, args.pred_path)
    