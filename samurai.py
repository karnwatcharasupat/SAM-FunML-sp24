import glob
import os

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import largestinteriorrectangle as lir
from matplotlib.patches import Rectangle

from scipy.spatial import ConvexHull, QhullError, convex_hull_plot_2d
from matplotlib.path import Path

from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

SAM_CHECKPOINT = 'weights/sam_vit_h_4b8939.pth'
MODEL_TYPE = "vit_h"

PROMPT_PATH = "/home/kwatchar3/projects/SAM-FunML-sp24/data/full_data/prompts"
GT_PATH = "/home/kwatchar3/projects/SAM-FunML-sp24/data/full_data/gt"

MAX_FILES = 400

def get_outer_box(points):
    x, y = points.T
    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()
    return x1, y1, x2, y2

def is_in_box(points, box):
    x, y = points.T
    x1, y1, x2, y2 = box

    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def compute_iou(mask, gt):

    intersection = np.logical_and(mask, gt).sum()
    union = np.logical_or(mask, gt).sum()

    if union == 0:
        return 0

    iou = intersection / union

    return iou


def compute_iou_from_convex_hull(gch: ConvexHull, gt: np.ndarray):

    mask = np.zeros((*gt.shape, 1), dtype=np.uint8)

    cv2.fillPoly(mask, [gch.points[gch.vertices].astype(np.int32)[None, ...]], 1)

    mask = mask[..., 0]

    return compute_iou(mask, gt)


def load_student_data(predictor, samples, labels, student_path, run_sam=False):

    data = []

    student_id = os.path.basename(student_path)

    df = pd.read_excel(os.path.join(student_path, f"{student_id}.xlsx"))

    df = df[["slice", "# green dots of best", "# red dots of best ", "best score"]].dropna()
    df = df.set_index("slice")
    df.columns = ["ng", "nr", "score"]
    df.index = df.index.astype(np.int32)

    points_path = os.path.join(student_path, "points")
    er_path = os.path.join(student_path, "eachround")
    score_path = os.path.join(student_path, "scores")
    sort_path = os.path.join(student_path, "sorts")

    for file in tqdm(df.index):

        image = samples[file]
        gt = labels[file]
        if len(image.shape) == 2:
            image = cv2.cvtColor(
                (np.array(((image + 1) / 2) * 255, dtype='uint8')),
                cv2.COLOR_GRAY2RGB
                )

        if run_sam:
            predictor.set_image(image)

        g = os.path.join(points_path, f"{file}_green.npy")
        r = os.path.join(points_path, f"{file}_red.npy")

        g = np.load(g, allow_pickle=True)
        r = np.load(r, allow_pickle=True)

        ng = df.loc[file, "ng"].astype(np.int32)
        nr = df.loc[file, "nr"].astype(np.int32)

        for gg, rr in zip(g, r):
            if gg.shape[0] >= ng:
                break

        g = gg.astype(np.int32)[:ng]
        r = rr.astype(np.int32)[:nr]

        if g.shape[0] <= 2:
            continue

        try:
            gch = ConvexHull(g)
        except QhullError as qe:
            # print(qe)
            continue

        ghpnts = gch.points[gch.vertices].astype(np.int32)

        inner_bb  = lir.lir(ghpnts[None, ...])
        inner_bb = (*lir.pt1(inner_bb), *lir.pt2(inner_bb))

        outer_bb = get_outer_box(ghpnts)

        is_in_convex_hull = Path(ghpnts).contains_points(g)
        is_in_inner_bb = is_in_box(g, inner_bb)
        is_in_outer_bb = is_in_box(g, outer_bb)

        n_in_convex_hull = is_in_convex_hull.sum()
        n_in_inner_bb = is_in_inner_bb.sum()
        n_in_outer_bb = is_in_outer_bb.sum()

        is_red_in_convex_hull = Path(ghpnts).contains_points(r)
        is_red_in_inner_bb = is_in_box(r, inner_bb)
        is_red_in_outer_bb = is_in_box(r, outer_bb)

        n_red_in_convex_hull = is_red_in_convex_hull.sum()
        n_red_in_inner_bb = is_red_in_inner_bb.sum()
        n_red_in_outer_bb = is_red_in_outer_bb.sum()

        if run_sam:

            obb_masks, _, _ = predictor.predict(
                point_coords=r,
                point_labels=np.zeros(r.shape[0]),
                box=np.array(outer_bb),
                multimask_output=True,
            )

            obb_iou = compute_iou(obb_masks[0], gt)


            g_outside_ibb = g[~is_in_inner_bb]

            point_coords = np.concatenate([g_outside_ibb, r])
            point_labels = np.concatenate([np.zeros(g_outside_ibb.shape[0]), np.ones(r.shape[0])])

            ibb_masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array(inner_bb),
                multimask_output=True,
            )

            ibb_iou = compute_iou(ibb_masks[0], gt)

            recon_coords = np.concatenate([g[:2], r[:1]])
            recon_labels = np.concatenate([np.ones(g[:2].shape[0]), np.zeros(r[:1].shape[0])])

            recon_masks, _, _ = predictor.predict(
                point_coords=recon_coords,
                point_labels=recon_labels,
                box=None,
                multimask_output=True,
            )

            recon_iou = compute_iou(recon_masks[0], gt)
        else:
            obb_iou = None
            ibb_iou = None
            recon_iou = None

        convex_hull_iou = compute_iou_from_convex_hull(gch, gt)
        print(convex_hull_iou)

        data.append(
            {
                "student_id": student_id,
                "file": file,
                "ng": ng,
                "nr": nr,
                "ng_in_convex_hull": n_in_convex_hull,
                "ng_in_inner_bb": n_in_inner_bb,
                "ng_in_outer_bb": n_in_outer_bb,
                "nr_in_convex_hull": n_red_in_convex_hull,
                "nr_in_inner_bb": n_red_in_inner_bb,
                "nr_in_outer_bb": n_red_in_outer_bb,
                "obb_iou": obb_iou,
                "ibb_iou": ibb_iou,
                "original_iou": df.loc[file, "score"],
                "recon_iou": recon_iou,
                "convex_hull_iou": convex_hull_iou,
            }
        )

        if file % 25 == 0 and run_sam:
            print(f"File: {file}, OBB IOU: {obb_iou}, "
                  f"IBB IOU: {ibb_iou}, "
                  f"Recon IOU: {recon_iou}, "
                  f"Original IOU: {df.loc[file, 'score']}")

    data = pd.DataFrame(data)
    if run_sam:
        path = os.path.join(student_path, f"{student_id}_processed.csv")
    else:
        path = os.path.join(student_path, f"{student_id}_processed_no_sam.csv")
    data.to_csv(path, index=False)

    return data


def test_sam(class_name="Bus", run_sam=True):


    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).cuda()
    predictor = SamPredictor(sam)

    samples = np.load(os.path.join(GT_PATH, class_name, "samples.npy"), allow_pickle=True)
    labels = np.load(os.path.join(GT_PATH, class_name, "labels.npy"), allow_pickle=True)

    students = glob.glob(os.path.join(PROMPT_PATH, class_name, "st*"))

    dfs = []

    for student in students:
        df = load_student_data(predictor, samples, labels, student)

        dfs.append(df)

    df = pd.concat(dfs)

    if run_sam:
        df.to_csv(os.path.join(PROMPT_PATH, class_name, "all_processed.csv"), index=False)
    else:
        df.to_csv(os.path.join(PROMPT_PATH, class_name, "all_processed_no_sam.csv"), index=False)

def test_all_classes():
    classes = glob.glob(os.path.join(GT_PATH, "*"))

    print(classes)

    for class_path in classes:
        try:
            class_name = os.path.basename(class_path)
            print(f"Testing {class_name}")
            test_sam(class_name)
        except Exception as e:
            print(e)



import seaborn as sns
def analyze_results(class_name, with_sam=True):
    try:
        if with_sam:
            df = pd.read_csv(os.path.join(PROMPT_PATH, class_name, "all_processed.csv"))
        else:
            df = pd.read_csv(os.path.join(PROMPT_PATH, class_name, "all_processed_no_sam.csv"))
    except:
        print(f"Could not read {class_name}")
        return None

    if not with_sam:
        return df

    print(df[['obb_iou', 'ibb_iou', 'recon_iou', 'original_iou']].describe())

    f, ax = plt.subplots(1, 1, figsize=(15, 5))

    edges = np.linspace(0, 1, 100)

    sns.histplot(df['obb_iou'], ax=ax, color='r', label='Outer BB', stat='density', bins=edges)
    # sns.histplot(df['ibb_iou'], ax=ax, color='g', label='Inner BB', stat='density', bins=edges)
    sns.histplot(df['recon_iou'], ax=ax, color='y', label='2G1R', stat='density', bins=edges)
    sns.histplot(df['original_iou'], ax=ax, color='b', label='Original', stat='density', bins=edges)

    ax.legend()
    plt.savefig(f"./results/{class_name}_iou.png")
    plt.show(block=False)

    return df

def analyze_all_classes(with_sam=True):

    classes = glob.glob(os.path.join(GT_PATH, "*"))
    class_names = [os.path.basename(class_path) for class_path in classes]

    df = []

    for class_name in class_names:
        print(f"Analyzing {class_name}")
        dfc = analyze_results(class_name, with_sam=with_sam)

        if dfc is None:
            continue

        dfc['class'] = class_name
        df.append(dfc)

    df = pd.concat(df)

    if with_sam:
        df.to_csv("./results/all_classes.csv", index=False)
    else:
        df.to_csv("./results/all_classes_no_sam.csv", index=False)

def read_results(thresh=5):
    # df = pd.read_csv("./results/all_classes.csv")
    df = pd.read_csv("./results/all_classes.csv")


    # print(df.groupby('class')[['obb_iou', 'ibb_iou', 'recon_iou', 'original_iou']].describe())

    n_classes = df['class'].nunique()
    print(f"Number of classes: {n_classes}")
    f, axs = plt.subplots(3, 4, figsize=(30, 20))

    df = df[df.ng > thresh]

    print(df.columns)

    for i, class_name in enumerate(df['class'].unique()):
        if i >= 12:
            break
        ax = axs[i // 4, i % 4]
        dfc = df[df['class'] == class_name]
        edges = np.linspace(0, 1, 100)

        sns.kdeplot(dfc['obb_iou'], ax=ax, color='r', label='Outer BB')
        sns.kdeplot(dfc['recon_iou'], ax=ax, color='y', label='2G1R')
        sns.kdeplot(dfc['original_iou'], ax=ax, color='b', label='Original')
        sns.kdeplot(dfc['convex_hull_iou'], ax=ax, color='g', label='Convex Hull')

        ax.set_title(class_name)
        ax.legend()

    plt.suptitle(f"IOU Distribution Samples with > {thresh} Green Dots")

    plt.savefig(f"./results/all_iou-{thresh}.png")
    plt.show(block=True)


if __name__ == '__main__':
    import fire

    fire.Fire()