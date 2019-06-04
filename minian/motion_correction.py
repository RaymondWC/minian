import numpy as np
import xarray as xr
import cv2
import sys
import itertools as itt
import pyfftw.interfaces.numpy_fft as npfft
import numba as nb
import dask.array as darr
from scipy.stats import zscore
from scipy.ndimage import center_of_mass
from collections import OrderedDict
from skimage import transform as tf
from scipy.stats import zscore
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from IPython.core.debugger import set_trace


def detect_and_correct_old(mov):
    surf = cv2.xfeatures2d.SURF_create(200)
    matcher = cv2.BFMatcher(crossCheck=True)
    detect_list = [surf.detectAndCompute(f, None) for f in mov]
    kp_list = [d[0] for d in detect_list]
    des_list = [d[1] for d in detect_list]
    match_list = []
    for des0, des1 in zip(des_list[:-1], des_list[1:]):
        match_list.append(matcher.match(des0, des1))
    matching_points = []
    for iframe, matches in enumerate(match_list):
        points0 = []
        points1 = []
        matches.sort(key=lambda ma: ma.distance, reverse=True)
        for ma in matches[:3]:
            points0.append(kp_list[iframe][ma.queryIdx].pt)
            points1.append(kp_list[iframe + 1][ma.trainIdx].pt)
        points0 = np.float32(np.array(points0))
        points1 = np.float32(np.array(points1))
        matching_points.append((points0, points1))
    trans_list = [
        cv2.getAffineTransform(pt[0], pt[1]) for pt in matching_points
    ]
    mov_correct = mov.copy()
    for iframe, trans in enumerate(trans_list):
        mov_correct[iframe + 1] = cv2.warpAffine(mov_correct[iframe], trans,
                                                 mov[0].shape[::-1])
    return mov_correct


def detect_and_correct(varray,
                       d_th=None,
                       r_th=None,
                       z_th=None,
                       q_th=None,
                       h_th=400,
                       std_thres=5,
                       opt_restr=5,
                       opt_std_thres=15,
                       opt_h_prop=0.1,
                       opt_err_thres=40,
                       method='translation',
                       upsample=None,
                       weight=False,
                       invert=False,
                       enhance=True):
    surf = cv2.xfeatures2d.SURF_create(h_th, extended=True)
    matcher = cv2.BFMatcher_create(crossCheck=True)
    # clache = cv2.createCLAHE(clipLimit=2, tileGridSize=(50, 50))
    varray = varray.transpose('frame', 'height', 'width')
    varr_mc = varray.astype(np.uint8)
    varr_ref = varray.astype(np.uint8)
    lk_params = dict(
        winSize=(200, 300),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,
                  0.0001),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    frm_idx = varr_mc.coords['frame']
    if invert:
        varr_ref = 255 - varr_ref
    if upsample:
        w = varray.coords['width']
        h = varray.coords['height']
        w_up = np.linspace(w[0], w[-1], len(w) * upsample)
        h_up = np.linspace(h[0], h[-1], len(h) * upsample)
        varr_ref = varr_ref.reindex(method='nearest', width=w_up, height=h_up)
    if enhance:
        for fid in frm_idx.values:
            fm = varr_ref.sel(frame=fid)
            fm.values = cv2.bilateralFilter(
                cv2.equalizeHist(fm.values), 9, 250, 250)
            varr_ref.loc[dict(frame=fid)] = fm
    match_dict = OrderedDict()
    shifts, shifts_idx, drop_idx = ([], [], [])
    last_registered = frm_idx[0].values
    for i, fid in enumerate(frm_idx[1:].values):
        im_src = varr_ref.sel(frame=last_registered).values
        im_dst = varr_ref.sel(frame=fid).values
        detect_src = surf.detectAndCompute(im_src, None)
        detect_dst = surf.detectAndCompute(im_dst, None)
        if not detect_src[0]:
            sys.stdout.write("\033[K")
            print("insufficient features for frame {}".format(last_registered))
            drop_idx.append(fid)
            continue
        if not detect_dst[0]:
            sys.stdout.write("\033[K")
            print("insufficient features for frame {}".format(fid))
            drop_idx.append(fid)
            continue
        match = matcher.match(detect_src[1], detect_dst[1])
        p_src, p_dst, eu_d, eu_x, eu_y, vma = ([], [], [], [], [], [])
        for idm, ma in enumerate(match):
            if True:
                pt0 = np.array(detect_src[0][ma.queryIdx].pt)
                pt1 = np.array(detect_dst[0][ma.trainIdx].pt)
                pt_diff = pt0 - pt1
                d = np.sqrt(np.sum(pt_diff**2))
                r = ma.distance
                if (d < d_th if d_th else True and r < r_th if r_th else True):
                    p_src.append(detect_src[0][ma.queryIdx].pt)
                    p_dst.append(detect_dst[0][ma.trainIdx].pt)
                    eu_d.append(d)
                    eu_x.append(pt_diff[0])
                    eu_y.append(pt_diff[1])
                    vma.append(ma)
        if not len(vma) > 0:
            set_trace()
            print("unable to find valid match for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        p_src, p_dst, vma = np.array(p_src), np.array(p_dst), np.array(vma)
        eu_d, eu_x, eu_y = np.array(eu_d), np.array(eu_x), np.array(eu_y)
        if z_th:
            eu_x_z_mask = np.absolute(zscore(eu_x)) < z_th
            eu_y_z_mask = np.absolute(zscore(eu_y)) < z_th
            eu_dist_z_mask = np.absolute(zscore(eu_d)) < z_th
            dist_z_mask = np.logical_and.reduce(
                [eu_dist_z_mask, eu_x_z_mask, eu_y_z_mask])
        else:
            dist_z_mask = np.ones_like(eu_d)
        if q_th:
            x_h_ma = eu_x < np.percentile(eu_x, q_th)
            x_l_ma = eu_x > np.percentile(eu_x, 100 - q_th)
            y_h_ma = eu_y < np.percentile(eu_y, q_th)
            y_l_ma = eu_y > np.percentile(eu_y, 100 - q_th)
            d_h_ma = eu_d < np.percentile(eu_d, q_th)
            d_l_ma = eu_d > np.percentile(eu_d, 100 - q_th)
            dist_q_mask = np.logical_and.reduce(
                [x_h_ma, x_l_ma, y_h_ma, y_l_ma, d_h_ma, d_l_ma])
        else:
            dist_q_mask = np.ones_like(eu_d)
        mask = np.logical_and(dist_z_mask, dist_q_mask)
        p_src, p_dst, vma = p_src[mask], p_dst[mask], vma[mask]
        eu_d, eu_x, eu_y = eu_d[mask], eu_x[mask], eu_y[mask]
        if not len(vma) > 0:
            sys.stdout.write("\033[K")
            print("No matches passed consistency test for frame {} and {}".
                  format(last_registered, fid))
            drop_idx.append(fid)
            continue
        trans, hmask = cv2.findHomography(
            p_src, p_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
        hmask = hmask.squeeze().astype(bool)
        p_src, p_dst, vma = p_src[hmask], p_dst[hmask], vma[hmask]
        eu_d, eu_x, eu_y = eu_d[hmask], eu_x[hmask], eu_y[hmask]
        if not len(vma) > 0:
            sys.stdout.write("\033[K")
            print("no matches formed a homography for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        elif np.std(eu_d) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("dist variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_d)))
            drop_idx.append(fid)
            continue
        elif np.std(eu_x) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("x variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_x)))
            drop_idx.append(fid)
            continue
        elif np.std(eu_y) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("y variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_y)))
            drop_idx.append(fid)
            continue
        est_shift = np.median(p_dst - p_src, axis=0)
        pts_src = cv2.goodFeaturesToTrack(im_src, 100, 0.5, 3, blockSize=3)
        if pts_src is None or not len(pts_src) > 1:
            sys.stdout.write("\033[K")
            print(
                "not enough good corners for frame {}".format(last_registered))
            drop_idx.append(fid)
            continue
        pts_dst = cv2.goodFeaturesToTrack(im_dst, 100, 0.5, 3, blockSize=3)
        if pts_dst is None or not len(pts_dst) > 1:
            sys.stdout.write("\033[K")
            print("not enough good corners for frame {}".format(fid))
            drop_idx.append(fid)
            continue
        pts_src = np.array(pts_src).squeeze().astype(np.float32)
        pts_dst = pts_src + est_shift
        pts_dst = np.array(pts_dst).astype(np.float32)
        vld_mask = pts_dst.min(axis=1) > 0
        if not vld_mask.sum() > 0:
            sys.stdout.write("\033[K")
            print("no valid corners for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        pts_src, pts_dst = pts_src[vld_mask], pts_dst[vld_mask]
        p1, st0, err0 = cv2.calcOpticalFlowPyrLK(im_src, im_dst,
                                                 pts_src.copy(),
                                                 pts_dst.copy(), **lk_params)
        p0r, st1, err1 = cv2.calcOpticalFlowPyrLK(im_dst, im_src,
                                                  p1.copy(),
                                                  pts_src.copy(), **lk_params)
        d_mask = np.absolute(pts_src - p0r).reshape(-1, 2).max(-1) < 1
        st0 = st0.squeeze().astype(bool)
        st1 = st1.squeeze().astype(bool)
        optmask = np.logical_and.reduce([st0, st1, d_mask])
        if not np.any(optmask):
            sys.stdout.write("\033[K")
            print(("no valid optical flow matching was found "
                   "for frame {} and {}").format(last_registered, fid))
            drop_idx.append(fid)
            continue
        pts_src, pts_dst, err0 = p0r[optmask], p1[optmask], err0[optmask]
        if err0.mean() > opt_err_thres:
            sys.stdout.write("\033[K")
            print(("optical flow error too high "
                   "for frame {} and {}. error: {}").format(
                       last_registered, fid, err0.mean()))
            drop_idx.append(fid)
            continue
        # consmask = np.absolute(pts_src - pts_dst - est_shift).max(
        #     axis=1) < opt_restr
        # if not consmask.sum() > 0:
        #     print(("no optical flow was found consitent with surf result "
        #            "for frame {} and {}").format(last_registered, fid))
        #     drop_idx.append(fid)
        #     continue
        # pts_src, pts_dst = pts_src[consmask], pts_dst[consmask]
        if len(pts_src) > 3:
            trans, hmask = cv2.findHomography(
                pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=3)
            hmask = hmask.squeeze().astype(bool)
        else:
            hmask = np.ones(len(pts_src), dtype=bool)
        if hmask.sum() < opt_h_prop * len(hmask):
            sys.stdout.write("\033[K")
            print(("too many optical flow matches were outliers "
                   "for frame {} and {}").format(last_registered, fid))
            hmask = np.ones(len(pts_src), dtype=bool)
        pts_src = pts_src[hmask]
        pts_dst = pts_dst[hmask]
        pts_diff = pts_src - pts_dst
        pts_dist = np.sqrt((pts_diff**2).sum(axis=1))
        if np.std(pts_dist) > opt_std_thres:
            sys.stdout.write("\033[K")
            print(("optical flow distance variance too high "
                   "for frame {} and {}. std:{}").format(
                       last_registered, fid, np.std(pts_dist)))
            drop_idx.append(fid)
            continue
        pts_src = pts_src.reshape((-1, 2))
        pts_dst = pts_dst.reshape((-1, 2))
        if method == 'opencv':
            trans = cv2.estimateRigidTransform(pts_dst, pts_src, False)
            if trans is not None:
                varr_mc.loc[dict(frame=fid)] = cv2.warpAffine(
                    varr_mc.sel(frame=fid).values,
                    trans,
                    varr_mc.sel(frame=fid).values.shape[::-1])
            else:
                print("unable to find transform for frame {}".format(fid))
        elif method == 'translation':
            if weight and len(pts_dist) > 1:
                weights = np.exp(-np.array(np.absolute(zscore(pts_dist))) * 10)
                weights = weights / np.sum(weights)
            else:
                weights = None
            shift = estimate_translation(pts_src, pts_dst, weights)
            shifts.append(shift)
            shifts_idx.append(fid)
        elif method == 'skimage':
            trans = tf.estimate_transform('similarity', pts_src, pts_dst)
            varr_mc.loc[dict(frame=fid)] = tf.warp(
                varr_mc.sel(frame=fid), trans.inverse)
        print(
            ("processing frame {:5d} of {:5d}, "
             "current features: {:3d}, current err: {:06.4f}").format(
                 i, len(frm_idx), len(pts_src), err0.mean()),
            end='\r')
        last_registered = fid
        match_dict[fid] = dict(
            src=detect_src,
            dst=detect_dst,
            match=vma,
            src_fid=last_registered,
            upsample=upsample if upsample else 1)
    if method == 'translation':
        shifts = xr.DataArray(
            shifts,
            coords=dict(frame=shifts_idx),
            dims=['frame', 'shift_dims'])
        shifts_final = []
        for fid in frm_idx[1:].values:
            cur_sh_hist = shifts.sel(frame=slice(frm_idx[0], fid))
            cur_shift = cur_sh_hist.sum('frame')
            cur_shift = cur_shift.values.astype(int)
            shifts_final.append(cur_shift)
            varr_mc.loc[dict(frame=fid)] = apply_translation(
                varr_mc.sel(frame=fid), cur_shift)
        shifts_final = xr.DataArray(
            shifts_final,
            coords=dict(frame=frm_idx[1:]),
            dims=['frame', 'shift_dims'])
    else:
        shifts_final = None
    varr_mc = varr_mc.reindex(
        method='nearest',
        width=varray.coords['width'],
        height=varray.coords['height'])
    return varr_mc.rename(varray.name + "_MotionCorrected"
                          ), match_dict, np.array(drop_idx), shifts_final


def remove_duplicate_keypoints(detect, threshold=2):
    remv_idx = []
    kps = detect[0]
    des = detect[1]
    for kp0, kp1 in itt.combinations(enumerate(kps), 2):
        if not (kp0[0] in remv_idx or kp1[0] in remv_idx):
            dist = np.sqrt(
                np.sum(np.array(kp0[1].pt) - np.array(kp1[1].pt))**2)
            if dist < threshold:
                remv_idx.append(kp0[0])
    kps = [kp for ikp, kp in enumerate(kps) if ikp not in remv_idx]
    des = np.delete(des, remv_idx, axis=0)
    return (kps, des)


def estimate_translation(pts_src, pts_dst, weights=None):
    return np.average(pts_src - pts_dst, axis=0, weights=weights)
    # return np.median(pts_src - pts_dst, axis=0)


def apply_translation(img, shift):
    return np.roll(img, shift, axis=(1, 0))


def estimate_shift_fft(varr, dim='frame', on='first', pad_f=1, pct_thres=None):
    varr = varr.chunk(dict(height=-1, width=-1))
    dims = list(varr.dims)
    dims.remove(dim)
    sizes = [varr.sizes[d] for d in ['height', 'width']]
    if not pct_thres:
        pct_thres = (1 - 10 / (sizes[0] * sizes[1])) * 100
    print(pct_thres)
    pad_s = np.array(sizes) * pad_f
    pad_s = pad_s.astype(int)
    results = []
    print("computing fft on array")
    varr_fft = xr.apply_ufunc(
        darr.fft.fft2,
        varr,
        input_core_dims=[[dim, 'height', 'width']],
        output_core_dims=[[dim, 'height', 'width']],
        dask='allowed',
        kwargs=dict(s=pad_s),
        output_dtypes=[np.complex64])
    if on == 'mean':
        meanfm = varr.mean(dim)
        src_fft = xr.apply_ufunc(
            darr.fft.fft2,
            meanfm,
            input_core_dims=[['height', 'width']],
            output_core_dims=[['height', 'width']],
            dask='allowed',
            kwargs=dict(s=pad_s),
            output_dtypes=[np.complex64])
    elif on == 'first':
        src_fft = varr_fft.isel(**{dim: 0})
    elif on == 'last':
        src_fft = varr_fft.isel(**{dim: -1})
    elif on == 'perframe':
        src_fft = varr_fft.shift(**{dim: 1})
    else:
        try:
            src_fft = varr_fft.isel(**{dim: on})
        except TypeError:
            print("template not understood. returning")
            return
    print("estimating shifts")
    res = xr.apply_ufunc(
        shift_fft,
        src_fft,
        varr_fft,
        input_core_dims=[['height', 'width'], ['height', 'width']],
        output_core_dims=[['variable']],
        kwargs=dict(pct_thres=pct_thres),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes=dict(variable=3))
    res = res.assign_coords(variable=['height', 'width', 'corr'])
    return res


def mask_shifts(varr_fft, corr, shifts, z_thres, perframe=True, pad_f=1):
    dims = list(varr_fft.dims)
    dims.remove('frame')
    sizes = [varr_fft.sizes[d] for d in dims]
    pad_s = np.array(sizes) * pad_f
    pad_s = pad_s.astype(int)
    mask = xr.apply_ufunc(zscore, corr.fillna(0)) > z_thres
    shifts = shifts.where(mask)
    if perframe:
        mask_diff = xr.DataArray(
            np.diff(mask.astype(int)),
            coords=dict(frame=mask.coords['frame'][1:]),
            dims=['frame'])
        good_idx = mask.coords['frame'].where(mask > 0, drop=True)
        bad_idx = mask_diff.coords['frame'].where(mask_diff == -1, drop=True)
        for cur_bad in bad_idx:
            gb_diff = good_idx - cur_bad
            try:
                next_good = gb_diff[gb_diff > 0].min() + cur_bad
                last_good = gb_diff[gb_diff < 0].max() + cur_bad
                cur_src = varr_fft.sel(frame=last_good)
                cur_dst = varr_fft.sel(frame=next_good)
                res = shift_fft(cur_src, cur_dst, pad_s, pad_f)
                shifts.loc[dict(frame=next_good.values)] = res[0:2]
            except (KeyError, ValueError):
                print("unable to correct for bad frame: {}".format(int(cur_bad)))
    return shifts, mask


def shift_fft(fft_src, fft_dst, pad_s=None, pad_f=1, pct_thres=99.99):
    if not np.iscomplexobj(fft_src):
        fft_src = np.fft.fft2(fft_src)
    if not np.iscomplexobj(fft_dst):
        fft_dst = np.fft.fft2(fft_dst)
    if np.isnan(fft_src).any() or np.isnan(fft_dst).any():
        return np.array([0, 0, np.nan])
    dims = fft_dst.shape
    prod = fft_src * np.conj(fft_dst)
    iprod = np.fft.ifft2(prod)
    iprod_sh = np.fft.fftshift(iprod)
    cor = iprod_sh.real
    # cor = np.log(np.where(iprod_sh.real > 1, iprod_sh.real, 1))
    cor_cent = np.where(cor > np.percentile(cor, pct_thres), cor, 0)
    sh = center_of_mass(cor_cent) - np.ceil(np.array(dims) / 2.0 * pad_f)
    # sh = np.unravel_index(cor.argmax(), cor.shape) - np.ceil(np.array(dims) / 2.0 * pad_f)
    corr = np.max(iprod.real)
    return np.concatenate([sh, corr], axis=None)


def apply_shifts(varr, shifts):
    sh_dim = shifts.coords['variable'].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ['variable']],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask = 'parallelized',
        output_dtypes = [varr.dtype])
    return varr_sh


def shift_perframe(fm, sh):
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    index = [slice(None) for _ in range(fm.ndim)]
    for ish, s in enumerate(sh):
        index = [slice(None) for _ in range(fm.ndim)]
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = np.nan
        elif s == 0:
            continue
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = np.nan
    return fm


def interpolate_frame(varr, drop_idx):
    if drop_idx.dtype == bool:
        drop_idx = drop_idx.coords['frame'].where(~drop_idx, drop=True).values
    if not set(drop_idx):
        print("no bad frame to interpolate, returning input")
        return varr
    keep_idx = np.array(list(set(varr.coords['frame'].values) - set(drop_idx)))
    varr_int = varr.copy()
    for i, fid in enumerate(drop_idx):
        print(
            "processing frame: {} progress: {}/{}".format(
                fid, i, len(drop_idx)),
            end='\r')
        diff = keep_idx - fid
        try:
            fid_fwd = diff[diff > 0].min() + fid
        except ValueError:
            fid_fwd = keep_idx.max()
        try:
            fid_bak = diff[diff < 0].max() + fid
        except ValueError:
            fid_bak = keep_idx.min()
        int_src = xr.concat(
            [varr.sel(frame=fid_fwd),
             varr.sel(frame=fid_bak)], dim='frame')
        varr_int.loc[dict(frame=fid)] = int_src.mean('frame')
    print("\ninterpolation done")
    return varr_int.rename(varr.name + "_Interpolated")

##

class MotionCorrect(object):
    """
        class implementing motion correction operations
       """

    def __init__(self, varr, min_mov=None, dview=None, max_shifts=(6, 6), niter_rig=1, splits_rig=14, num_splits_to_process_rig=None,
                 strides=(96, 96), overlaps=(32, 32), splits_els=14, num_splits_to_process_els=[7, None],
                 upsample_factor_grid=4, max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True, gSig_filt=None,
                 use_cuda=False, border_nan=True, pw_rigid=False, num_frames_split=80, var_name_hdf5='mov'):
        """
        Constructor class for motion correction operations
        Args:
           fname: str
               path to file to motion correct
           min_mov: int16 or float32
               estimated minimum value of the movie to produce an output that is positive
           dview: ipyparallel view object list
               to perform parallel computing, if NOne will operate in single thread
           max_shifts: tuple
               maximum allow rigid shift
           niter_rig':int
               maximum number of iterations rigid motion correction, in general is 1. 0
               will quickly initialize a template with the first frames
           splits_rig': int
            for parallelization split the movies in  num_splits chuncks across time
           num_splits_to_process_rig:list,
               if none all the splits are processed and the movie is saved, otherwise at each iteration
               num_splits_to_process_rig are considered
           strides: tuple
               intervals at which patches are laid out for motion correction
           overlaps: tuple
               overlap between pathes (size of patch strides+overlaps)
           pw_rigig: bool, default: False
               flag for performing motion correction when calling motion_correct
           splits_els':list
               for parallelization split the movies in  num_splits chuncks across time
           num_splits_to_process_els:list,
               if none all the splits are processed and the movie is saved  otherwise at each iteration
                num_splits_to_process_els are considered
           upsample_factor_grid:int,
               upsample factor of shifts per patches to avoid smearing when merging patches
           max_deviation_rigid:int
               maximum deviation allowed for patch with respect to rigid shift
           shifts_opencv: Bool
               apply shifts fast way (but smoothing results)
           nonneg_movie: boolean
               make the SAVED movie and template mostly nonnegative by removing min_mov from movie
           use_cuda : bool, optional
               Use skcuda.fft (if available). Default: False
           border_nan : bool or string, optional
               Specifies how to deal with borders. (True, False, 'copy', 'min')
           num_frames_split: int, default: 80
               Number of frames in each batch. Used when cosntructing the options
               through the params object
           var_name_hdf5: str, default: 'mov'
               If loading from hdf5, name of the variable to load
       Returns:
           self
        """
        # if 'ndarray' in str(type(fname)):
        #     logging.info('Creating file for motion correction "tmp_mov_mot_corr.hdf5"')
        #     cm.movie(fname).save('./tmp_mov_mot_corr.hdf5')
        #     fname = ['./tmp_mov_mot_corr.hdf5']

        if type(fname) is not list:
            fname = [fname]

        self.varr = varr
        self.dview = dview
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.shifts_opencv = shifts_opencv
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.gSig_filt = gSig_filt
        self.use_cuda = use_cuda
        self.border_nan = border_nan
        self.pw_rigid = pw_rigid
        self.var_name_hdf5 = var_name_hdf5
        # if self.use_cuda and not HAS_CUDA:
        #     logging.debug("pycuda is unavailable. Falling back to default FFT.")

    def motion_correct(self, template=None, save_movie=False):
        """general function for performing all types of motion correction. The
        function will perform either rigid or piecewise rigid motion correction
        depending on the attribute self.pw_rigid and will perform high pass
        spatial filtering for determining the motion (used in 1p data) if the
        attribute self.gSig_filt is not None. A template can be passed, and the
        output can be saved as a memory mapped file.
        Args:
            template: nd.array, default: None
                template provided by user for motion correction
            save_movie: bool, default: False
                flag for saving motion corrected file(s) as memory mapped file(s)
        Returns:
            self
        """
        # TODO: Review the docs here, and also why we would ever return self
        #       from a method that is not a constructor
        if self.min_mov is None:
            if self.gSig_filt is None:
                self.min_mov = np.array([cm.load(self.fname[0],
                                                 var_name_hdf5=self.var_name_hdf5,
                                                 subindices=slice(400))]).min()
            else:
                self.min_mov = np.array([high_pass_filter_space(m_, self.gSig_filt)
                    for m_ in cm.load(self.fname[0], var_name_hdf5=self.var_name_hdf5,
                                      subindices=slice(400))]).min()

        if self.pw_rigid:
            self.motion_correct_pwrigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                    np.max(np.abs(self.y_shifts_els))))
        else:
            self.motion_correct_rigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))
        self.border_to_0 = b0.astype(np.int)
        self.mmap_file = self.fname_tot_els if self.pw_rigid else self.fname_tot_rig
        return self

    def motion_correct_rigid(self, template=None, save_movie=False):
        """
        Perform rigid motion correction
        Args:
            template: ndarray 2D
                if known, one can pass a template to register the frames to
            save_movie_rigid:Bool
                save the movies vs just get the template
        Returns:
            self
        Important Fields:
            self.fname_tot_rig: name of the mmap file saved
            self.total_template_rig: template updated by iterating  over the chunks
            self.templates_rig: list of templates. one for each chunk
            self.shifts_rig: shifts in x and y per frame
        """
        self.total_template_rig = template
        self.templates_rig:List = []
        self.fname_tot_rig:List = []
        self.shifts_rig:List = []

        for fname_cur in self.fname:
            _fname_tot_rig, _total_template_rig, _templates_rig, _shifts_rig = motion_correct_batch_rigid(
                fname_cur,
                self.max_shifts,
                dview=self.dview,
                splits=self.splits_rig,
                num_splits_to_process=self.num_splits_to_process_rig,
                num_iter=self.niter_rig,
                template=self.total_template_rig,
                shifts_opencv=self.shifts_opencv,
                save_movie_rigid=save_movie,
                add_to_movie=-self.min_mov,
                nonneg_movie=self.nonneg_movie,
                gSig_filt=self.gSig_filt,
                use_cuda=self.use_cuda,
                border_nan=self.border_nan,
                var_name_hdf5=self.var_name_hdf5)
            if template is None:
                self.total_template_rig = _total_template_rig

            self.templates_rig += _templates_rig
            self.fname_tot_rig += [_fname_tot_rig]
            self.shifts_rig += _shifts_rig

        return self

    def motion_correct_pwrigid(
            self,
            save_movie=True,
            template=None,
            show_template=False):
        """Perform pw-rigid motion correction
        Args:
            template: ndarray 2D
                if known, one can pass a template to register the frames to
            save_movie:Bool
                save the movies vs just get the template
            show_template: boolean
                whether to show the updated template at each iteration
        Returns:
            self
        Important Fields:
            self.fname_tot_els: name of the mmap file saved
            self.templates_els: template updated by iterating  over the chunks
            self.x_shifts_els: shifts in x per frame per patch
            self.y_shifts_els: shifts in y per frame per patch
            self.coord_shifts_els: coordinates associated to the patch for
            values in x_shifts_els and y_shifts_els
            self.total_template_els: list of templates. one for each chunk
        Raises:
            Exception: 'Error: Template contains NaNs, Please review the parameters'
        """

        num_iter = 1
        if template is None:
            #logging.info('Generating template by rigid motion correction')
            self = self.motion_correct_rigid()
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.fname_tot_els:List = []
        self.templates_els:List = []
        self.x_shifts_els:List = []
        self.y_shifts_els:List = []
        self.coord_shifts_els:List = []
        for name_cur in self.fname:
            for num_splits_to_process in self.num_splits_to_process_els:
                _fname_tot_els, new_template_els, _templates_els,\
                    _x_shifts_els, _y_shifts_els, _coord_shifts_els = motion_correct_batch_pwrigid(
                        name_cur, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
                        dview=self.dview, upsample_factor_grid=self.upsample_factor_grid,
                        max_deviation_rigid=self.max_deviation_rigid, splits=self.splits_els,
                        num_splits_to_process=num_splits_to_process, num_iter=num_iter, template=self.total_template_els,
                        shifts_opencv=self.shifts_opencv, save_movie=save_movie, nonneg_movie=self.nonneg_movie, gSig_filt=self.gSig_filt,
                        use_cuda=self.use_cuda, border_nan=self.border_nan, var_name_hdf5=self.var_name_hdf5)
                if show_template:
                    pl.imshow(new_template_els)
                    pl.pause(.5)
                if np.isnan(np.sum(new_template_els)):
                    raise Exception(
                        'Template contains NaNs, something went wrong. Reconsider the parameters')

            if template is None:
                self.total_template_els = new_template_els

            self.fname_tot_els += [_fname_tot_els]
            self.templates_els += _templates_els
            self.x_shifts_els += _x_shifts_els
            self.y_shifts_els += _y_shifts_els
            self.coord_shifts_els += _coord_shifts_els
        return self

    def apply_shifts_movie(self, fname, rigid_shifts=True, border_nan=True):
        """
        Applies shifts found by registering one file to a different file. Useful
        for cases when shifts computed from a structural channel are applied to a
        functional channel. Currently only application of shifts through openCV is
        supported.
        Args:
            fname: str
                name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable
            rigid_shifts: bool
                apply rigid or pw-rigid shifts (must exist in the mc object)
        Returns:
            m_reg: caiman movie object
                caiman movie object with applied shifts (not memory mapped)
        """

        Y = cm.load(fname).astype(np.float32)

        if rigid_shifts is True:
            if self.shifts_opencv:
                m_reg = [apply_shift_iteration(img, shift, border_nan=border_nan)
                         for img, shift in zip(Y, self.shifts_rig)]
            else:
                m_reg = [apply_shifts_dft(img, (
                    sh[0], sh[1]), 0, is_freq=False, border_nan=border_nan) for img, sh in zip(
                    Y, self.shifts_rig)]
        else:
            dims_grid = tuple(np.max(np.stack(self.coord_shifts_els[0], axis=1), axis=1) - np.min(
                np.stack(self.coord_shifts_els[0], axis=1), axis=1) + 1)
            shifts_x = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                np.float32) for _sh_ in self.x_shifts_els], axis=0)
            shifts_y = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                np.float32) for _sh_ in self.y_shifts_els], axis=0)
            dims = Y.shape[1:]
            x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(
                np.float32), np.arange(0., dims[1]).astype(np.float32))
            m_reg = [cv2.remap(img,
                               -cv2.resize(shiftY, dims) + x_grid, -cv2.resize(shiftX, dims) + y_grid, cv2.INTER_CUBIC)
                     for img, shiftX, shiftY in zip(Y, shifts_x, shifts_y)]

        return cm.movie(np.stack(m_reg, axis=0))






#%%
def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(np.int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img


#%%

def motion_correct_oneP_rigid(
        filename,
        gSig_filt,
        max_shifts,
        dview=None,
        splits_rig=10,
        save_movie=True,
        border_nan=True):
    '''Perform rigid motion correction on one photon imaging movies
    Args:
        filename: str
            name of the file to correct
        gSig_filt:
            size of the filter. If algorithm does not work change this parameters
        max_shifts: tuple of ints
            max shifts in x and y allowed
        dview:
            handle to cluster
        splits_rig: int
            number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100)
        save_movie: bool
            whether to save the movie in memory mapped format
        border_nan : bool or string, optional
            Specifies how to deal with borders. (True, False, 'copy', 'min')
    Returns:
        Motion correction object
    '''
    min_mov = np.array([caiman.motion_correction.high_pass_filter_space(
        m_, gSig_filt) for m_ in cm.load(filename[0], subindices=range(400))]).min()
    new_templ = None

    # TODO: needinfo how the classes works
    mc = MotionCorrect(
        filename,
        min_mov,
        dview=dview,
        max_shifts=max_shifts,
        niter_rig=1,
        splits_rig=splits_rig,
        num_splits_to_process_rig=None,
        shifts_opencv=True,
        nonneg_movie=True,
        gSig_filt=gSig_filt,
        border_nan=border_nan)

    mc.motion_correct_rigid(save_movie=save_movie, template=new_templ)

    return mc

def motion_correct_oneP_nonrigid(
        filename,
        gSig_filt,
        max_shifts,
        strides,
        overlaps,
        splits_els,
        upsample_factor_grid,
        max_deviation_rigid,
        dview=None,
        splits_rig=10,
        save_movie=True,
        new_templ=None,
        border_nan=True):
    '''Perform rigid motion correction on one photon imaging movies
    Args:
        filename: str
            name of the file to correct
        gSig_filt:
            size of the filter. If algorithm does not work change this parameters
        max_shifts: tuple of ints
            max shifts in x and y allowed
        dview:
            handle to cluster
        splits_rig: int
            number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100)
        save_movie: bool
            whether to save the movie in memory mapped format
        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')
    Returns:
        Motion correction object
    '''

    if new_templ is None:
        min_mov = np.array([high_pass_filter_space(
            m_, gSig_filt) for m_ in cm.load(filename, subindices=range(400))]).min()
    else:
        min_mov = np.min(new_templ)

    # TODO: needinfo how the classes works
    mc = MotionCorrect(
        filename,
        min_mov,
        dview=dview,
        max_shifts=max_shifts,
        niter_rig=1,
        splits_rig=splits_rig,
        num_splits_to_process_rig=None,
        shifts_opencv=True,
        nonneg_movie=True,
        gSig_filt=gSig_filt,
        strides=strides,
        overlaps=overlaps,
        splits_els=splits_els,
        upsample_factor_grid=upsample_factor_grid,
        max_deviation_rigid=max_deviation_rigid,
        border_nan=border_nan)

    mc.motion_correct_pwrigid(save_movie=True, template=new_templ)
    return mc


def motion_correct_batch_rigid(fname, max_shifts, dview=None, splits=56, num_splits_to_process=None, num_iter=1,
                               template=None, shifts_opencv=False, save_movie_rigid=False, add_to_movie=None,
                               nonneg_movie=False, gSig_filt=None, subidx=slice(None, None, 1), use_cuda=False,
                               border_nan=True, var_name_hdf5='mov'):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file
    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable
        max_shifts: tuple
            x and y maximum allowd shifts
        dview: ipyparallel view
            used to perform parallel computing
        splits: int
            number of batches in which the movies is subdivided
        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed
        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.
        template: ndarray
            if a good approximation of the template to register is available, it can be used
        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing
        save_movie_rigid: boolean
             toggle save movie
        subidx: slice
            Indices to slice
        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False
    Returns:
         fname_tot_rig: str
         total_template:ndarray
         templates:list
              list of produced templates, one per batch
         shifts: list
              inferred rigid shifts to correct the movie
    Raises:
        Exception 'The movie contains nans. Nans are not allowed!'
    """
    corrected_slicer = slice(subidx.start, subidx.stop, subidx.step * 10)
    m = cm.load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)

    if m.shape[0] < 300:
        m = cm.load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)
    elif m.shape[0] < 500:
        corrected_slicer = slice(subidx.start, subidx.stop, subidx.step * 5)
        m = cm.load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)
    else:
        corrected_slicer = slice(subidx.start, subidx.stop, subidx.step * 30)
        m = cm.load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)

    if len(m.shape) < 3:
        m = cm.load(fname, var_name_hdf5=var_name_hdf5)
        m = m[corrected_slicer]
        logging.warning("Your original file was saved as a single page " +
                        "file. Consider saving it in multiple smaller files" +
                        "with size smaller than 4GB (if it is a .tif file)")
    if template is None:
        if gSig_filt is not None:
            m = cm.movie(
                np.array([high_pass_filter_space(m_, gSig_filt) for m_ in m]))

        template = bin_median(
            m.motion_correct(max_shifts[1], max_shifts[0], template=None)[0])

    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        #logging.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    # else:
    #     logging.debug('Adding to movie ' + str(add_to_movie))

    save_movie = False
    fname_tot_rig = None
    res_rig: List = []
    for iter_ in range(num_iter):
        #logging.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1:
            save_movie = save_movie_rigid
            #logging.debug('saving!')

        fname_tot_rig, res_rig = motion_correction_piecewise(fname, splits, strides=None, overlaps=None,
                                                             add_to_movie=add_to_movie, template=old_templ,
                                                             max_shifts=max_shifts, max_deviation_rigid=0,
                                                             dview=dview, save_movie=save_movie,
                                                             base_name=os.path.split(
                                                                 fname)[-1][:-4] + '_rig_', subidx=subidx,
                                                             num_splits=num_splits_to_process,
                                                             shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie,
                                                             gSig_filt=gSig_filt,
                                                             use_cuda=use_cuda, border_nan=border_nan,
                                                             var_name_hdf5=var_name_hdf5)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_rig]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

        #logging.debug((old_div(np.linalg.norm(new_templ - old_templ), np.linalg.norm(old_templ))))

    total_template = new_templ
    templates = []
    shifts: List = []
    for rr in res_rig:
        shift_info, idxs, tmpl = rr
        templates.append(tmpl)
        shifts += [[sh[0][0], sh[0][1]] for sh in shift_info[:len(idxs)]]

    return fname_tot_rig, total_template, templates, shifts

def motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None, newstrides=None,
                                 dview=None, upsample_factor_grid=4, max_deviation_rigid=3,
                                 splits=56, num_splits_to_process=None, num_iter=1,
                                 template=None, shifts_opencv=False, save_movie=False, nonneg_movie=False, gSig_filt=None,
                                 use_cuda=False, border_nan=True, var_name_hdf5='mov'):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file
    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable
        strides: tuple
            strides of patches along x and y
        overlaps:
            overlaps of patches along x and y. exmaple. If strides = (64,64) and overlaps (32,32) patches will be (96,96)
        newstrides: tuple
            overlaps after upsampling
        newoverlaps: tuple
            strides after upsampling
        max_shifts: tuple
            x and y maximum allowd shifts
        dview: ipyparallel view
            used to perform parallel computing
        splits: int
            number of batches in which the movies is subdivided
        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed
        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.
        template: ndarray
            if a good approximation of the template to register is available, it can be used
        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing
        save_movie_rigid: boolean
             toggle save movie
        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False
    Returns:
        fname_tot_rig: str
        total_template:ndarray
        templates:list
            list of produced templates, one per batch
        shifts: list
            inferred rigid shifts to corrrect the movie
    Raises:
        Exception 'You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function'
    """
    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        logging.error('The template contains NaNs. NaNs are not allowed!')
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()

        if iter_ == num_iter - 1:
            save_movie = save_movie
            if save_movie:
                logging.debug('saving mmap of ' + fname)

        fname_tot_els, res_el = motion_correction_piecewise(fname, splits, strides, overlaps,
                                                            add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts,
                                                            max_deviation_rigid=max_deviation_rigid,
                                                            newoverlaps=newoverlaps, newstrides=newstrides,
                                                            upsample_factor_grid=upsample_factor_grid, order='F', dview=dview, save_movie=save_movie,
                                                            base_name=os.path.split(fname)[-1][:-4] + '_els_', num_splits=num_splits_to_process,
                                                            shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                                            use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_el]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    x_shifts = []
    y_shifts = []
    coord_shifts = []
    for rr in res_el:
        shift_info_chunk, idxs_chunk, tmpl_chunk = rr
        templates.append(tmpl_chunk)
        for shift_info, _ in zip(shift_info_chunk, idxs_chunk):
            total_shift, _, xy_grid = shift_info
            x_shifts.append(np.array([sh[0] for sh in total_shift]))
            y_shifts.append(np.array([sh[1] for sh in total_shift]))
            coord_shifts.append(xy_grid)

    return fname_tot_els, total_template, templates, x_shifts, y_shifts, coord_shifts

def motion_correction_piecewise(varr, splits, strides, overlaps, add_to_movie=0, template=None,
                                max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4,  dview=None, subidx = None, num_splits=None,
                                shifts_opencv=False, nonneg_movie=False, gSig_filt=None,
                                use_cuda=False, border_nan=True, var_name_hdf5='mov'):
    """
    """
    # todo todocument
    # name, extension = os.path.splitext(fname)[:2]
    # extension = extension.lower()
    # is_fiji = False

    dims = varr.shape[0:2]
    T = varr.shape[2]
    d1, d2 = dims

    if type(splits) is int:
        if subidx is None:
            rng = range(T)
        else:
            rng = range(T)[subidx]

        idxs = np.array_split(list(rng), splits)

    else:
        idxs = splits
        # save_movie = False
    if template is None:
        raise Exception('Not implemented')

    shape_mov = (d1 * d2, T)

    #dims = d1, d2
    if num_splits is not None:
        idxs = np.array(idxs)[np.random.randint(0, len(idxs), num_splits)]
        # save_movie = False
        #logging.warning('**** MOVIE NOT SAVED BECAUSE num_splits is not None ****')

    # if save_movie:
    #     if base_name is None:
    #         base_name = os.path.split(fname)[1][:-4]
    #     fname_tot:Optional[str] = memmap_frames_filename(base_name, dims, T, order)
    #     fname_tot = os.path.join(os.path.split(fname)[0], fname_tot)
    #     np.memmap(fname_tot, mode='w+', dtype=np.float32,
    #               shape=prepare_shape(shape_mov), order=order)
    #     logging.info('Saving file as {}'.format(fname_tot))
    # else:
    #     fname_tot = None

    pars = []
    for idx in idxs:
        pars.append([fname, fname_tot, idx, shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
            newoverlaps, newstrides, shifts_opencv, nonneg_movie, gSig_filt, is_fiji,
            use_cuda, border_nan, var_name_hdf5])

    if dview is not None:
        #logging.info('** Starting parallel motion correction **')
        if HAS_CUDA and use_cuda:
            res = dview.map(tile_and_correct_wrapper,pars)
            dview.map(close_cuda_process, range(len(pars)))
        elif 'multiprocessing' in str(type(dview)):
            res = dview.map_async(tile_and_correct_wrapper, pars).get(4294967)
        else:
            res = dview.map_sync(tile_and_correct_wrapper, pars)
        #logging.info('** Finished parallel motion correction **')
    else:
        res = list(map(tile_and_correct_wrapper, pars))

    return fname_tot, res

def high_pass_filter_space(img_orig, gSig_filt):
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return cv2.filter2D(np.array(img_orig, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)

def bin_median(mat, window=10, exclude_nans=True):
    """ compute median of 3D array in along axis o by binning values
    Args:
        mat: ndarray
            input 3D matrix, time along first dimension
        window: int
            number of frames in a bin
    Returns:
        img:
            median image
    Raises:
        Exception 'Path to template does not exist:'+template
    """

    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img

def tile_and_correct_wrapper(params):
    """Does motion correction on specified image frames
    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image (to get individ frames, use out_fname to write them to disk)
    Notes:
    Also writes corrected frames to the mmap file specified by out_fname (if not None)
    """
    # todo todocument


    try:
        cv2.setNumThreads(0)
    except:
        pass  # 'Open CV is naturally single threaded'

    img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        shifts_opencv, nonneg_movie, gSig_filt, is_fiji, use_cuda, border_nan, var_name_hdf5 = params

    name, extension = os.path.splitext(img_name)[:2]
    extension = extension.lower()
    shift_info = []

    imgs = cm.load(img_name, subindices=idxs)
    mc = np.zeros(imgs.shape, dtype=np.float32)
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            logging.debug(count)
        mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan)
        shift_info.append([total_shift, start_step, xy_grid])

    if out_fname is not None:
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
                         shape=prepare_shape(shape_mov), order='F')
        if nonneg_movie:
            bias = np.float32(add_to_movie)
        else:
            bias = 0
        outv[:, idxs] = np.reshape(
            mc.astype(np.float32), (len(imgs), -1), order='F').T + bias
    new_temp = np.nanmean(mc, 0)
    new_temp[np.isnan(new_temp)] = np.nanmin(new_temp)
    return shift_info, idxs, new_temp

def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None,
                     use_cuda=False, border_nan=True):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches
    Args:
        img: ndaarray 2D
            image to correct
        template: ndarray
            reference image
        strides: tuple
            strides of the patches in which the FOV is subdivided
        overlaps: tuple
            amount of pixel overlaping between patches along each dimension
        max_shifts: tuple
            max shifts in x and y
        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields
        newoverlaps:tuple
            amount of pixel overlaping between patches along each dimension when upsampling the vector fields
        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field
        upsample_factor_fft: int
            resolution of fractional shifts
        show_movie: boolean whether to visualize the original and corrected frame during motion correction
        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)
        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times
        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data
        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False
        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')
    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image
    """

    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:

        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)

    if max_deviation_rigid == 0:

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            new_img = apply_shift_iteration(
                img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)

        else:

            if gSig_filt is not None:
                raise Exception(
                    'The use of FFT and filtering options have not been tested. Set opencv=True')

            new_img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None
    else:
        # extract patches
        templates = [
            it[-1] for it in sliding_window(template, overlaps=overlaps, strides=strides)]
        xy_grid = [(it[0], it[1]) for it in sliding_window(
            template, overlaps=overlaps, strides=strides)]
        num_tiles = np.prod(np.add(xy_grid[-1], 1))
        imgs = [it[-1]
                for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))

        if max_deviation_rigid is not None:

            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)

        else:

            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            dims = img.shape
            x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                np.float32), np.arange(0., dims[0]).astype(np.float32))
            m_reg = cv2.remap(img, cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
                              cv2.resize(shift_img_x.astype(np.float32), dims[::-1]) + y_grid,
                              cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                             # borderValue=add_to_movie)
            total_shifts = [
                    (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

        newshapes = np.add(newstrides, newoverlaps)

        imgs = [it[-1]
                for it in sliding_window(img, overlaps=newoverlaps, strides=newstrides)]

        xy_grid = [(it[0], it[1]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        start_step = [(it[2], it[3]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        dim_new_grid = tuple(np.add(xy_grid[-1], 1))

        shift_img_x = cv2.resize(
            shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        shift_img_y = cv2.resize(
            shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        diffs_phase_grid_us = cv2.resize(
            diffs_phase_grid, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)

        num_tiles = np.prod(dim_new_grid)

        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig
                imgs = [
                    it[-1] for it in sliding_window(img, overlaps=newoverlaps, strides=newstrides)]

            imgs = [apply_shift_iteration(im, sh, border_nan=border_nan)
                    for im, sh in zip(imgs, total_shifts)]

        else:
            if gSig_filt is not None:
                raise Exception(
                    'The use of FFT and filtering options have not been tested. Set opencv=True')

            imgs = [apply_shifts_dft(im, (
                sh[0], sh[1]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
                imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1]]
                new_img[x:x + newshapes[0], y:y + newshapes[1]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

            new_img = old_div(new_img, normalizer)

        else:  # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
            half_overlap_x = np.int(newoverlaps[0] / 2)
            half_overlap_y = np.int(newoverlaps[1] / 2)
            for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                new_img[x_start:x_end,
                        y_start:y_end] = im[x_start - x:, y_start - y:]

        if show_movie:
            img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)
            img_show = np.vstack([new_img, img])

            img_show = cv2.resize(img_show, None, fx=1, fy=1)

            cv2.imshow('frame', old_div(img_show, np.percentile(template, 99)))
            cv2.waitKey(int(1. / 500 * 1000))

        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        return new_img - add_to_movie, total_shifts, start_step, xy_grid