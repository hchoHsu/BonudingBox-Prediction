import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch
from natsort import natsorted

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

from tracker.bot_sort import BoTSORT, STrack

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def plot_tracking(image, tlwhs, laynum=0):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    line_thickness = -1
    radius = max(5, int(im_w/140.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
       
        color = [256*laynum, 256*laynum, 256]
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
    return im

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    # parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    # parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    # parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    # parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc |none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument('--predict_frame_num', type=int, default=30, help='prediction future after n frame')

    return parser

def cvt_online_targets(online_targets):
    # change online_targets into predictor outputs
    new_detections = []
    for t in online_targets:
        tlbr = STrack.tlwh_to_tlbr(t.tlwh)
        tid = t.track_id
        score = t.score
        det = [None] * 7
        det[:4] = tlbr
        det[4] = score
        det[5] = 1
        det[6] = tid
        new_detections.append(det) 
    
    return np.array(new_detections)

def cvtwh_cor(ID):
    x, y, w, h = ID
    return np.array([[x-0.5*w, y-0.5*h], [x-0.5*w, y+0.5*h],
            [x+0.5*w, y-0.5*h], [x+0.5*w, y+0.5*h]], dtype=np.float32)


def cvtcor_wh(ID):
    x = (ID[0][0][0] + ID[0][2][0]) * 0.5
    y = (ID[0][0][1] + ID[0][1][1]) * 0.5
    w = ID[0][2][0] - ID[0][0][0]
    h = ID[0][3][1] - ID[0][0][1]
    return np.array([x, y, w, h], dtype=np.float32)


def rtPerspectiveTransform(cur, pred):
    src = cvtwh_cor(cur)
    dst = cvtwh_cor(pred)
    M = cv2.getPerspectiveTransform(src, dst)
    return M

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            if fp16:
                x = x.half()
            self.model(x)
            self.model = model_trt
        
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        
        return outputs, img_info

def inference(predictor, tracker, args, exp, img):
    # Detect objects
    outputs, img_info = predictor.inference(img)
    # online_im = np.zeros(shape=img_info["raw_img"].shape)
    online_im = np.zeros(shape=img_info["raw_img"].shape)
    scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

    detections = []
    if outputs[0] is not None:
        outputs = outputs[0].cpu().numpy()
        detections = outputs[:, :7]
        detections[:, :4] /= scale

        online_targets = tracker.update(detections, img_info["raw_img"], isNotPred=True)

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        
    
        # ======================== Future Prediction ======================== #
        # Saving state for future prediction
        state = tracker.save_state()
        
        # translate
        new_detections = cvt_online_targets(online_targets)
        if len(new_detections) <= 0:
            return online_im
        
        online_targets = tracker.update(np.array(new_detections), img_info['raw_img'], isNotPred=False)
        
        online_tlwhs_pred = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical and tid in online_ids:
                cur_idx = online_ids.index(tid)
                cur_tlwh = online_tlwhs[cur_idx]
                
                # Perspective Transform
                PM = rtPerspectiveTransform(cur_tlwh, tlwh)
                for pred_time in range(0, args.predict_frame_num-1):
                    tlwh = cv2.perspectiveTransform(cvtwh_cor(tlwh)[None, :, :], PM)
                    tlwh = cvtcor_wh(tlwh)
                    
                    online_tlwhs_pred.append(tlwh)

            online_im = plot_tracking(online_im, online_tlwhs_pred, laynum=1)
        online_im = plot_tracking(online_im, online_tlwhs, laynum=0)
        tracker.reload_state(state)
        # ======================== End Prediction ======================== #
    else:
        online_im = online_im

    return online_im

def test_inference(predictor, tracker, args, exp):
    if osp.isdir("../../NTHU_23/20220917_111613534391/image_raw/"):
        files = get_image_list("../../NTHU_23/20220917_111613534391/image_raw/")
    else:
        files = ["../../NTHU_23/20220917_111613534391/image_raw/"]

    files = natsorted(files)
    frame_num = len(files)
    
    import time
    start_time = time.time()
    for frame_id, img_path in enumerate(files, 1):
        img = inference(predictor, tracker, args, exp, img_path)
        cv2.imwrite(os.path.join("/home/elsalab/Desktop/23/husky_ws/src/bbox_model/src/zed_video/NTHU_23/", os.path.basename(img_path)), img)
        print(img_path, img.shape)
    end_time = time.time()
    
    print(f"frame num:{frame_num}, fps:{frame_num/(end_time-start_time)}")
    

if __name__ == "__main__":
    args = make_parser().parse_args()
    args.name = None
    args.fp16 = True
    args.device = "cuda"
    args.ablation = False
    args.mot20 = False
    args.fps = 30
    args.batch_size = 1
    args.trt = True

    args.exp_file = r'./yolox/exps/example/mot/yolox_s_mix_det.py'

    exp = get_exp(args.exp_file, None)

    args.track_high_thresh = 0.6
    args.track_low_thresh = 0.1
    args.track_buffer = 30
    args.new_track_thresh = args.track_high_thresh + 0.1

    exp.test_conf = max(0.001, args.track_low_thresh - 0.01)
    
    args.experiment_name = exp.exp_name

    model = exp.get_model().to(args.device)
    model.eval()
    
    if not args.trt:
        ckpt_file = r'./pretrained/bytetrack_s_mot17.pth.tar'
        ckpt = torch.load(ckpt_file, map_location="cpu")

        # load the model state dict
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join("./YOLOX_outputs/yolox_s_mix_det/model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        # logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    tracker = BoTSORT(args, frame_rate=args.fps)
    
    test_inference(predictor, tracker, args, exp)
