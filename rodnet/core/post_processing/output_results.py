from rodnet.core.object_class import get_class_name
from cruw.mapping import ra2idx, idx2ra
import math


def write_dets_results(res, data_id, save_path, dataset):
    batch_size, win_size, max_dets, _ = res.shape
    classes = dataset.object_cfg.classes
    with open(save_path, 'a+') as f:
        for b in range(batch_size):
            for w in range(win_size):
                for d in range(max_dets):
                    cla_id = int(res[b, w, d, 0])
                    if cla_id == -1:
                        continue
                    row_id = res[b, w, d, 1]
                    col_id = res[b, w, d, 2]
                    conf = res[b, w, d, 3]
                    f.write("%d %s %d %d %s\n" % (data_id + w, get_class_name(cla_id, classes), row_id, col_id, conf))


def write_dets_results_single_frame(res, data_id, save_path, dataset):
    max_dets, _  = res.shape
    classes = dataset.object_cfg.classes
    with open(save_path, 'a+') as f:
        for d in range(max_dets):
            cla_id = int(res[d, 0])
            if cla_id == -1:
                continue
            row_id = res[d, 1]
            col_id = res[d, 2]
            conf = res[d, 3]
            rid = int(row_id)
            aid =  int(col_id)
            rng, azm = idx2ra(rid, aid, dataset.range_grid, dataset.angle_grid)
            f.write("%d %f %f %s %s\n" % (data_id , rng, azm, get_class_name(cla_id, classes), conf))
