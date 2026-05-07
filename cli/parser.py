import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def str2bool_or_str(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip()
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    return value


def add_bool_argument(parser, name, default, help_text):
    parser.add_argument(name, type=str2bool, nargs="?", const=True, default=default, help=help_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    # default.yaml / parameter to load the model
    add_bool_argument(
        parser,
        "--config",
        True,
        "method to load the model, i.e. true or false to load the model by this way or by default.yaml",
    )
    # Task and Mode
    parser.add_argument('--task', type=str, default='detect',
                        help='YOLO task, i.e. detect, segment, classify, pose, obb')
    parser.add_argument('--mode', type=str, default='train',
                        help='YOLO mode, i.e. train, val, predict, export, track, benchmark')

    # Train settings
    parser.add_argument(
        '--model',
        type=str,
        default="",
        help='path to model file, or leave empty/use "auto"/"best" for scale-aware FAN auto selection',
    )
    parser.add_argument('--data_path', type=str, default='', help='optional root path for dataset yaml lookup')
    parser.add_argument(
        '--data',
        type=str,
        default='cfg/datasets/VisDrone.yaml',
        help='path to dataset yaml, e.g. cfg/datasets/VisDrone.yaml',
    )
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train for')
    parser.add_argument('--time', type=float, default=None,
                        help='number of hours to train for, overrides epochs if supplied')
    parser.add_argument('--patience', type=int, default=100,
                        help='epochs to wait for no observable improvement for early stopping of training')
    parser.add_argument('--batch', type=int, default=32, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[896],
                        help='input images size as int for train and val modes, or list[h,w] for predict and export modes')
    add_bool_argument(parser, '--save', True, 'save train checkpoints and predict results')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', default=False, action='store_true',
                        help='True/ram, disk or False. Use cache for data loading')
    parser.add_argument('--device', type=str, default=[0],
                        help='device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of worker threads for data loading (per RANK if DDP)')
    parser.add_argument('--project', type=str, default="./run/detect", help='project name')
    parser.add_argument('--name', type=str, default=None,
                        help='experiment name, results saved to project/name directory')
    parser.add_argument('--exist_ok', default=False, action='store_true',
                        help='whether to overwrite existing experiment')
    parser.add_argument(
        '--pretrained',
        type=str2bool_or_str,
        nargs='?',
        const=True,
        default=False,
        help='whether to use a pretrained model (bool) or a model to load weights from (str)',
    )
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]')
    add_bool_argument(parser, '--verbose', True, 'whether to print verbose output')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    add_bool_argument(parser, '--deterministic', False, 'whether to enable deterministic mode')
    parser.add_argument('--single_cls', default=False, action='store_true',
                        help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', default=False,
                        help='rectangular training if mode=train or rectangular validation if mode=val')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10,
                        help='disable mosaic augmentation for final epochs (0 to disable)')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from last checkpoint')
    add_bool_argument(
        parser,
        '--amp',
        True,
        'Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check',
    )
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile ONNX and TensorRT speeds during training for loggers')
    parser.add_argument('--freeze', type=str, default=None,
                        help='freeze first n layers, or freeze list of layer indices during training')
    parser.add_argument('--multi_scale', action='store_true', default=False,
                        help='Whether to use multiscale during training')

    # Segmentation specific
    add_bool_argument(
        parser,
        '--overlap_mask',
        True,
        'merge object masks into a single image mask during training (segment train only)',
    )
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')

    # Classification specific
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

    # Val/Test settings
    add_bool_argument(parser, '--val', True, 'validate/test during training')
    parser.add_argument('--split', type=str, default='val',
                        help='dataset split to use for validation, i.e. val, test or train')
    add_bool_argument(parser, '--save_json', True, 'save results to JSON file')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, default=None,
                        help='object confidence threshold for detection (default 0.25 predict, 0.001 val)')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    add_bool_argument(parser, '--half', True, 'use half precision (FP16)')
    parser.add_argument('--dnn', action='store_true', default=False, help='use OpenCV DNN for ONNX inference')
    add_bool_argument(parser, '--plots', True, 'save plots and images during train/val')

    # Predict settings
    parser.add_argument('--source', type=str, default=None, help='source directory for images or videos')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', action='store_true', default=False,
                        help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize model features')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='filter results by class, i.e. classes=0, or classes=[0,2,3]')
    parser.add_argument('--retina_masks', action='store_true', default=False,
                        help='use high-resolution segmentation masks')
    parser.add_argument('--embed', type=int, nargs='+', default=None,
                        help='return feature vectors/embeddings from given layers')

    # Visualize settings
    parser.add_argument('--show', action='store_true', default=False,
                        help='show predicted images and videos if environment allows')
    parser.add_argument('--save_frames', action='store_true', default=False,
                        help='save predicted individual video frames')
    parser.add_argument('--save_txt', action='store_true', default=False, help='save results as .txt file')
    parser.add_argument('--save_conf', action='store_true', default=False, help='save results with confidence scores')
    parser.add_argument('--save_crop', action='store_true', default=False, help='save cropped images with results')
    add_bool_argument(parser, '--show_labels', True, 'show prediction labels, i.e. person')
    add_bool_argument(parser, '--show_conf', True, 'show prediction confidence, i.e. 0.99')
    add_bool_argument(parser, '--show_boxes', True, 'show prediction boxes')
    parser.add_argument('--line_width', type=int, default=None,
                        help='line width of the bounding boxes. Scaled to image size if None.')

    # Export settings
    parser.add_argument('--format', type=str, default='torchscript',
                        help='format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats')
    parser.add_argument('--keras', action='store_true', default=False, help='use Keras')
    parser.add_argument('--optimize', action='store_true', default=False, help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', default=False, help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', default=False, help='ONNX/TF/TensorRT: dynamic axes')
    add_bool_argument(parser, '--simplify', True, 'ONNX: simplify model using onnxslim')
    parser.add_argument('--opset', type=int, default=None, help='ONNX: opset version')
    parser.add_argument('--workspace', type=float, default=None,
                        help='TensorRT: workspace size (GiB), None will let TensorRT auto-allocate memory')
    parser.add_argument('--nms', action='store_true', default=False, help='CoreML: add NMS')

    # Hyperparameters
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate (i.e. SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay 5e-4')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='warmup epochs (fractions ok)')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='warmup initial bias lr')
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain (scale with pixels)')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--fan_freq', type=float, default=0.1, help='FAN frequency-consistency loss gain')
    parser.add_argument('--fan_task', type=float, default=0.15, help='FAN prior/gate alignment loss gain')
    parser.add_argument('--fan_sem', type=float, default=0.0,
                        help='FAN box-guided semantic discrimination loss gain')
    parser.add_argument('--fan_hf', type=float, default=1.0, help='FAN foreground high-frequency enhancement weight')
    parser.add_argument('--fan_lf', type=float, default=1.0, help='FAN background low-frequency suppression weight')
    parser.add_argument('--fan_phase', type=float, default=1.0, help='FAN boundary phase preservation weight')
    parser.add_argument('--fan_prior', type=float, default=1.0, help='FAN prior alignment weight')
    parser.add_argument('--fan_gate', type=float, default=1.0, help='FAN scan gate alignment weight')
    parser.add_argument('--fan_hf_margin', type=float, default=0.05, help='FAN foreground high-frequency margin')
    parser.add_argument('--fan_lf_margin', type=float, default=0.02,
                        help='FAN background low-frequency reduction margin')
    parser.add_argument('--fan_obj_bg_margin', type=float, default=0.05, help='FAN object-background contrast margin')
    parser.add_argument('--fan_sem_margin', type=float, default=0.05,
                        help='FAN object-background semantic separation margin')
    parser.add_argument('--fan_sem_logit', type=float, default=4.0,
                        help='FAN semantic contrast logit scale')
    parser.add_argument('--fan_pred_mix', type=float, default=0.08,
                        help='max predicted-prior mixing ratio during FAN guide warmup')
    parser.add_argument('--fan_guide_warmup', type=int, default=8000,
                        help='iterations used to ramp FAN guide from GT prior toward predicted prior')
    parser.add_argument('--fan_loss_size', type=int, default=48,
                        help='max spatial size used by FAN auxiliary losses, 0 disables compression')
    parser.add_argument('--fan_cls_balance', type=float, default=0.25,
                        help='boost positive cls supervision for rare classes within a batch')
    parser.add_argument('--fan_cls_balance_pow', type=float, default=0.5,
                        help='inverse-frequency exponent used by FAN class balancing')
    parser.add_argument('--fan_cls_balance_cap', type=float, default=4.0,
                        help='max per-class boost used by FAN class balancing')
    parser.add_argument('--fan_small_box', type=float, default=0.2,
                        help='extra loss weight applied to tiny positive boxes')
    parser.add_argument('--fan_small_area_ref', type=float, default=0.005,
                        help='normalized area threshold below which FAN boosts tiny boxes')
    parser.add_argument('--fan_small_box_max', type=float, default=2.0,
                        help='max scale factor used by FAN tiny-box weighting')
    parser.add_argument('--fan_bg_suppress', type=float, default=0.0,
                        help='extra box-guided hard-negative suppression weight for clear background anchors')
    parser.add_argument('--fan_bg_topk', type=float, default=0.005,
                        help='fraction of clear-background anchors mined as hard negatives per image')
    parser.add_argument('--fan_bg_gamma', type=float, default=1.5,
                        help='confidence exponent used for box-guided hard-negative suppression')
    parser.add_argument('--fan_bg_warmup', type=int, default=12000,
                        help='iterations used to ramp box-guided hard-negative suppression')
    parser.add_argument('--fan_bg_rare_floor', type=float, default=0.25,
                        help='minimum suppression scale kept for rare/absent classes during box-guided background suppression')
    parser.add_argument('--fan_bg_rare_pow', type=float, default=0.5,
                        help='frequency exponent that biases box-guided background suppression toward common classes')
    parser.add_argument('--fan_det', type=float, default=0.0,
                        help='direct GT-box calibration weight applied on dense detection score maps')
    parser.add_argument('--fan_det_margin', type=float, default=0.05,
                        help='margin used by FAN direct score calibration between object and background responses')
    parser.add_argument('--fan_det_warmup', type=int, default=4000,
                        help='iterations used to ramp FAN direct score calibration')
    parser.add_argument('--fan_rare_sampling', type=float, default=0.2,
                        help='image-level rare-class resampling strength for FAN training')
    parser.add_argument('--fan_rare_pow', type=float, default=0.5,
                        help='inverse-frequency exponent used by FAN rare-class resampling')
    parser.add_argument('--fan_rare_cap', type=float, default=2.5,
                        help='max class boost used by FAN rare-class resampling')
    parser.add_argument('--fan_rare_decay_start', type=int, default=-1,
                        help='epoch index at which FAN rare-class resampling starts to decay toward 0; -1 keeps legacy schedule')
    parser.add_argument('--pose', type=float, default=12.0, help='pose loss gain')
    parser.add_argument('--kobj', type=float, default=1.0, help='keypoint obj loss gain')
    parser.add_argument('--nbs', type=int, default=64, help='nominal batch size')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='image perspective (+/- fraction), range 0-0.001')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--bgr', type=float, default=0.0, help='image channel BGR (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='image cutmix (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='segment copy-paste (probability)')
    parser.add_argument('--copy_paste_mode', type=str, default='flip',
                        help='the method to do copy_paste augmentation (flip, mixup)')
    parser.add_argument('--auto_augment', type=str, default='randaugment',
                        help='auto augmentation policy for classification (randaugment, autoaugment, augmix)')
    parser.add_argument('--erasing', type=float, default=0.4,
                        help='probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0.')
    parser.add_argument('--crop_fraction', type=float, default=1.0,
                        help='image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0.')

    # Custom config.yaml
    parser.add_argument('--cfg', type=str, default=None, help='for overriding defaults.yaml')

    # Tracker settings
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                        help='tracker type, choices=[botsort.yaml, bytetrack.yaml')

    return parser.parse_args()
