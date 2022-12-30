import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from tracker.bot_sort import BoTSORT, STrack
from tracker.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    parser.add_argument('--predict_frame_num', type=int, default=10, help='prediction future after n frame')
    parser.add_argument('--result_name', type=str, default=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), help="result txt file name")
    parser.add_argument('--kalman', default=False, action="store_true", help="Using kalman prediction.")
    
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def plot_tracking(image, tlwhs, laynum=0):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    line_thickness = 3
    radius = max(5, int(im_w/140.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
       
        color = [256*laynum, 256*laynum, 256]
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
    return im

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
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

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
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info
    

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]

    files.sort()
    timer = Timer()
    results = []
    tracker = BoTSORT(args, frame_rate=args.fps)

    for frame_id, img_path in enumerate(files, 1):
        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        detections = []
        if outputs[0] is not None:
            # ====================== Current Frame Detection ====================== #
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            online_targets = tracker.update(detections, img_info['raw_img'], isNotPred=True)

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
            
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, laynum=1)

# ========================================== Future Prediction ========================================== #
            state = tracker.save_state()
            
            if args.kalman is False:
                # 預測幾個frame就跑幾次
                for predict_idx in range(0, args.predict_frame_num):
                    # translate
                    new_detections = cvt_online_targets(online_targets)
                    if len(new_detections) <= 0:
                        break

                    online_targets = tracker.update(np.array(new_detections), img_info['raw_img'], isNotPred=False)

                    if predict_idx == args.predict_frame_num - 1:
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
                                
                                results.append(
                                    f"{frame_id+args.predict_frame_num},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
                        online_im = plot_tracking(online_im, online_tlwhs, laynum=1)
            else:
                # translate
                new_detections = cvt_online_targets(online_targets)
                if len(new_detections) <= 0:
                    break

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
                            
                            if pred_time % 2 == 1:
                                online_tlwhs_pred.append(tlwh)

                        # 確保會過
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            results.append(
                                f"{frame_id+args.predict_frame_num-1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                online_im = plot_tracking(online_im, online_tlwhs_pred, laynum=1)

            tracker.reload_state(state)
# ========================================== End Prediction ========================================== #
            
            timer.toc()
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        # if args.save_result:
        save_folder = osp.join(vis_folder, args.result_name)
        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

    if args.save_result:
        res_file = osp.join(vis_folder, f"{args.result_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image" or args.demo == "images":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    else:
        raise ValueError("Error: Unknown source: " + args.demo)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
