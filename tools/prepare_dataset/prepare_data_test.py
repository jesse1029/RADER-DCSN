import os
import sys
import shutil
import numpy as np
import json
import pickle
import argparse

from cruw import CRUW
from cruw.annotation.init_json import init_meta_json
from cruw.mapping import ra2idx

from rodnet.core.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from rodnet.utils.load_configs import load_configs_from_file
from rodnet.utils.visualization import visualize_confmap

SPLITS_LIST = ['train', 'valid', 'test', 'demo']


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RODNet data.')
    parser.add_argument('--config', type=str, dest='config', help='configuration file path')
    parser.add_argument('--data_root', type=str, help='directory to the prepared data')
    parser.add_argument('--split', type=str, dest='split', help='choose from train, valid, test, supertest')
    parser.add_argument('--out_data_dir', type=str, default='./data',
                        help='data directory to save the prepared data')
    parser.add_argument('--overwrite', action="store_true", help="overwrite prepared data if exist")
    args = parser.parse_args()
    return args


def load_anno_txt(txt_path, n_frame, dataset):
    folder_name_dict = dict(
        #cam_0='IMAGES_0',
        rad_h='RADAR_RA_H'
    )
    anno_dict = init_meta_json(n_frame, folder_name_dict)
    with open(txt_path, 'r') as f:
        data = f.readlines()
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        rid, aid = ra2idx(r, a, dataset.range_grid, dataset.angle_grid)
        anno_dict[frame_id]['rad_h']['n_objects'] += 1
        anno_dict[frame_id]['rad_h']['obj_info']['categories'].append(class_name)
        anno_dict[frame_id]['rad_h']['obj_info']['centers'].append([r, a])
        anno_dict[frame_id]['rad_h']['obj_info']['center_ids'].append([rid, aid])
        anno_dict[frame_id]['rad_h']['obj_info']['scores'].append(1.0)

    return anno_dict


def generate_confmaps(metadata_dict, n_class, viz):
    confmaps = []
    for metadata_frame in metadata_dict:
        n_obj = metadata_frame['rad_h']['n_objects']
        obj_info = metadata_frame['rad_h']['obj_info']
        if n_obj == 0:
            confmap_gt = np.zeros(
                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                dtype=float)
            confmap_gt[-1, :, :] = 1.0  # initialize noise channal
        else:
            confmap_gt = generate_confmap(n_obj, obj_info, dataset, config_dict)
            confmap_gt = normalize_confmap(confmap_gt)
            confmap_gt = add_noise_channel(confmap_gt, dataset, config_dict)
        assert confmap_gt.shape == (
            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
        if viz:
            visualize_confmap(confmap_gt)
        confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    return confmaps


def prepare_data(dataset, config_dict, data_dir, split, save_dir, viz=False, overwrite=False):
    """
    Prepare pickle data for RODNet training and testing
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param data_dir: output directory of the processed data
    :param split: train, valid, test, demo, etc.
    :param save_dir: output directory of the prepared data
    :param viz: whether visualize the prepared data
    :param overwrite: whether overwrite the existing prepared data
    :return:
    """
    camera_configs = dataset.sensor_cfg.camera_cfg
    radar_configs = dataset.sensor_cfg.radar_cfg
    n_chirp = radar_configs['n_chirps']
    n_class = dataset.object_cfg.n_class

    data_root = config_dict['dataset_cfg']['data_root']
    anno_root = config_dict['dataset_cfg']['anno_root']
    set_cfg = config_dict['dataset_cfg'][split]
    if 'seqs' not in set_cfg:
        sets_seqs = sorted(os.listdir(os.path.join(data_root, set_cfg['subdir'])))
    else:
        sets_seqs = set_cfg['seqs']

    if overwrite:
        if os.path.exists(os.path.join(data_dir, split)):
            shutil.rmtree(os.path.join(data_dir, split))
        os.makedirs(os.path.join(data_dir, split))

    for seq in sets_seqs:
        seq_path = os.path.join(data_root, set_cfg['subdir'], seq)
        seq_anno_path = os.path.join(anno_root, set_cfg['subdir'], seq + config_dict['dataset_cfg']['anno_ext'])
        save_path = os.path.join(save_dir, seq + '.pkl')
        print("Sequence %s saving to %s" % (seq_path, save_path))
        # ~ print('\n\n\n')
        # ~ print('config_dict = ',config_dict)
        # ~ print('save_path = ',save_path)
        # ~ print("Sequence %s saving to %s" % (seq_path, save_path))
        # ~ #RADAR_RA_H
        # ~ print('seq_path = ',seq_path)
        # ~ print('camera_configs = ',camera_configs)
        # ~ print('radar_configs = ',radar_configs)
        # ~ print('radar_configs[data_type] = ',radar_configs['data_type'])
        # ~ print('\n\n\n')

        try:
            if not overwrite and os.path.exists(save_path):
                print("%s already exists, skip" % save_path)
                continue
            # ~ image_dir = os.path.join(seq_path, camera_configs['image_folder'])
            image_dir = os.path.join(seq_path, 'RADAR_RA_H')
            # ~ print('image_dir = ',image_dir)
            image_paths = sorted([os.path.join(image_dir, name) for name in os.listdir(image_dir) if
                                  name.endswith(radar_configs['ext'])])
                                  #camera_configs  ---> radar_configs
            # ~ print('image_paths = ',image_paths)
            n_frame = len(image_paths)
            print('n_frame = ',n_frame)

            # ~ n_frame = 0
            radar_dir = os.path.join(seq_path, dataset.sensor_cfg.radar_cfg['chirp_folder'])
            # ~ print('radar_dir = ',radar_dir)
            # ~ if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                # ~ radar_paths = sorted([os.path.join(radar_dir, name) for name in os.listdir(radar_dir) if
                                      # ~ name.endswith(dataset.sensor_cfg.radar_cfg['ext'])])
                # ~ n_radar_frame = len(radar_paths)
                # ~ assert n_frame == n_radar_frame
            # ~ elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                # ~ radar_paths_chirp = []
                # ~ for chirp_id in range(n_chirp):
                    # ~ chirp_dir = os.path.join(radar_dir, '%04d' % chirp_id)
                    # ~ paths = sorted([os.path.join(chirp_dir, name) for name in os.listdir(chirp_dir) if
                                    # ~ name.endswith(config_dict['dataset_cfg']['radar_cfg']['ext'])])
                    # ~ n_radar_frame = len(paths)
                    # ~ assert n_frame == n_radar_frame
                    # ~ radar_paths_chirp.append(paths)
                # ~ radar_paths = []
                # ~ for frame_id in range(n_frame):
                    # ~ frame_paths = []
                    # ~ for chirp_id in range(n_chirp):
                        # ~ frame_paths.append(radar_paths_chirp[chirp_id][frame_id])
                    # ~ radar_paths.append(frame_paths)
            if radar_configs['data_type'] == 'ROD2021':
            # ~ elif radar_configs['data_type'] == 'ROD2021':
                # ~ assert len(os.listdir(radar_dir)) == n_frame * len(radar_configs['chirp_ids'])
                # ~ print('radar if = ',len(os.listdir(radar_dir)))
                radar_paths = []
                for frame_id in range(n_frame):
                    chirp_paths = []
                    for chirp_id in radar_configs['chirp_ids']:
                        path = os.path.join(radar_dir, '%06d_%04d.' % (frame_id, chirp_id) +
                                            dataset.sensor_cfg.radar_cfg['ext'])
                        chirp_paths.append(path)
                    radar_paths.append(chirp_paths)
            else:
                raise ValueError

            data_dict = dict(
                data_root=data_root,
                data_path=seq_path,
                seq_name=seq,
                n_frame=n_frame,
                image_paths=image_paths,
                radar_paths=radar_paths,
                anno=None,
            )

            if split == 'demo':
                # no labels need to be saved
                pickle.dump(data_dict, open(save_path, 'wb'))
                continue
            else:
                anno_obj = {}
                # ~ if config_dict['dataset_cfg']['anno_ext'] == '.txt':
                    # ~ anno_obj['metadata'] = load_anno_txt(seq_anno_path, n_frame, dataset)

                # ~ elif config_dict['dataset_cfg']['anno_ext'] == '.json':
                    # ~ with open(os.path.join(seq_anno_path), 'r') as f:
                        # ~ anno = json.load(f)
                    # ~ anno_obj['metadata'] = anno['metadata']
                # ~ else:
                    # ~ raise

                # ~ anno_obj['confmaps'] = generate_confmaps(anno_obj['metadata'], n_class, viz)
                data_dict['anno'] = anno_obj
                # save pkl files
                pickle.dump(data_dict, open(save_path, 'wb'))
            # end frames loop

        except Exception as e:
            print("Error while preparing %s: %s" % (seq_path, e))


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    splits = args.split.split(',')
    out_data_dir = args.out_data_dir
    overwrite = args.overwrite

    dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    config_dict = load_configs_from_file(args.config)
    radar_configs = dataset.sensor_cfg.radar_cfg

    for split in splits:
        if split not in SPLITS_LIST:
            raise TypeError("split %s cannot be recognized" % split)

    for split in splits:
        save_dir = os.path.join(out_data_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Preparing %s sets ...' % split)
        prepare_data(dataset, config_dict, out_data_dir, split, save_dir, viz=False, overwrite=overwrite)






# ~ warning: loading calibration data failed
# ~ Preparing test sets ...
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_CM1S013 saving to data/data/test/2019_05_28_CM1S013.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_CM1S013: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_CM1S013/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_MLMS005 saving to data/data/test/2019_05_28_MLMS005.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_MLMS005: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_MLMS005/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PBMS006 saving to data/data/test/2019_05_28_PBMS006.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PBMS006: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PBMS006/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PCMS004 saving to data/data/test/2019_05_28_PCMS004.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PCMS004: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PCMS004/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S012 saving to data/data/test/2019_05_28_PM2S012.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S012: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S012/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S014 saving to data/data/test/2019_05_28_PM2S014.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S014: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_PM2S014/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD004 saving to data/data/test/2019_09_18_ONRD004.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD004: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD004/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD009 saving to data/data/test/2019_09_18_ONRD009.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD009: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_18_ONRD009/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_29_ONRD012 saving to data/data/test/2019_09_29_ONRD012.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_29_ONRD012: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_09_29_ONRD012/IMAGES_0'
# ~ Sequence /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_10_13_ONRD048 saving to data/data/test/2019_10_13_ONRD048.pkl
# ~ Error while preparing /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_10_13_ONRD048: [Errno 2] No such file or directory: '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_10_13_ONRD048/IMAGES_0'

# ~ seq_path =  /media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/test/2019_05_28_CM1S013
# ~ camera_configs =  {'image_width': 1440, 'image_height': 864, 'frame_rate': 30, 'image_folder': 'IMAGES_0', 'ext': 'jpg'}
# ~ radar_configs =  {'ramap_rsize': 128, 'ramap_asize': 128, 'frame_rate': 30, 'crop_num': 3, 'n_chirps': 255, 'chirp_ids': [0, 64, 128, 192], 'sample_freq': 4000000.0, 'sweep_slope': 21001700000000.0, 'data_type': 'ROD2021', 'chirp_folder': 'RADAR_RA_H', 'ext': 'npy', 'ramap_rsize_label': 122, 'ramap_asize_label': 121, 'ra_min_label': -60, 'ra_max_label': 60, 'rr_min': 1.0, 'rr_max': 25.0, 'ra_min': -90, 'ra_max': 90}
# ~ config_dict =  {'dataset_cfg': {'dataset_name': 'ROD2021', 'base_root': '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet', 'data_root': '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet', 'anno_root': '/media/jess/ec60f916-4bd2-4bf2-b75d-2c2fe8508c47/ROD2021/RODNet/TRAIN_RAD_H_ANNO', 'anno_ext': '.txt', 'train': {'subdir': 'train'}, 'valid': {'subdir': 'valid', 'seqs': []}, 'test': {'subdir': 'test'}, 'demo': {'subdir': 'demo', 'seqs': []}}, 'model_cfg': {'type': 'HG', 'name': 'rodnet-hg1-win16-wobg', 'max_dets': 20, 'peak_thres': 0.4, 'ols_thres': 0.3, 'stacked_num': 1}, 'confmap_cfg': {'confmap_sigmas': {'pedestrian': 15, 'cyclist': 20, 'car': 30}, 'confmap_sigmas_interval': {'pedestrian': [5, 15], 'cyclist': [8, 20], 'car': [10, 30]}, 'confmap_length': {'pedestrian': 1, 'cyclist': 2, 'car': 3}}, 'train_cfg': {'n_epoch': 50, 'batch_size': 16, 'lr': 1e-05, 'lr_step': 5, 'win_size': 16, 'train_step': 1, 'train_stride': 4, 'log_step': 100, 'save_step': 10000}, 'test_cfg': {'test_step': 1, 'test_stride': 8, 'rr_min': 1.0, 'rr_max': 20.0, 'ra_min': -60.0, 'ra_max': 60.0}}


