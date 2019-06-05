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
from past.utils import old_div
from numpy.fft import ifftshift
from past.builtins import basestring
from past.utils import old_div
from numpy.fft import ifftshift


try:
    cv2.setNumThreads(0)
except:
    pass

from cv2 import dft as fftn
from cv2 import idft as ifftn

opencv = True


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
                d = np.sqrt(np.sum(pt_diff ** 2))
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
        pts_dist = np.sqrt((pts_diff ** 2).sum(axis=1))
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
                np.sum(np.array(kp0[1].pt) - np.array(kp1[1].pt)) ** 2)
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
    # results = []
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
        dask='parallelized',
        output_dtypes=[varr.dtype])
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


def normcorre_wrapper(varr_ref, pw_rigid=False, max_shifts=(5, 5), gSig_filt=(3, 3),
                      strides=(48, 48), overlaps=(24, 24), max_deviation_rigid=3,
                      border_nan='copy'):
    """Does motion correction on specified image frames
    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image (to get individ frames, use out_fname to write them to disk)
    Notes:
    Also writes corrected frames to the mmap file specified by out_fname (if not None)
    """
    # todo todocument

    #
    # try:
    #     cv2.setNumThreads(0)
    # except:
    #     pass  # 'Open CV is naturally single threaded'

    # img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
    #     add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
    #     shifts_opencv, nonneg_movie, gSig_filt, is_fiji, use_cuda, border_nan, var_name_hdf5 = params
    if not pw_rigid:
        strides=None
        overlaps=None
        max_deviation_rigid=0

    template = high_pass_filter_space(np.nanmedian(varr_ref, axis=0), gSig_filt)
    shift_info = []
    # imgs = cm.load(img_name, subindices=idxs)
    mc = np.zeros(varr_ref.shape, dtype=np.float32)
    for count, img in enumerate(varr_ref):
        mc[count], total_shift, start_step, xy_grid = \
            tile_and_correct(img, template, strides, overlaps, max_shifts,
                             upsample_factor_fft=10, show_movie=False,
                             max_deviation_rigid=max_deviation_rigid, gSig_filt=gSig_filt,
                             border_nan=border_nan)
        shift_info.append([total_shift, start_step, xy_grid])

    # if out_fname is not None:
    #     outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
    #                      shape=prepare_shape(shape_mov), order='F')
    #     if nonneg_movie:
    #         bias = np.float32(add_to_movie)
    #     else:
    #         bias = 0
    #     outv[:, idxs] = np.reshape(
    #         mc.astype(np.float32), (len(imgs), -1), order='F').T + bias
    new_temp = np.nanmean(mc, 0)
    new_temp[np.isnan(new_temp)] = np.nanmin(new_temp)
    return mc, shift_info, new_temp


def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None,
                     upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0,
                     shifts_opencv=True, gSig_filt=None,
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
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in
            zip(
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
            m_reg = cv2.remap(np.asarray(img), cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
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
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itt.product(
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
            for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts,
                                                                      weight_matrix):

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


def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.
    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is
    Copyright (C) 2011, the scikit-image team
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = fftn(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        shifts = shifts[::-1]
        nc, nr = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nr, Nc = np.meshgrid(Nr, Nc)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * 1. * Nr / nr - shifts[1] * 1. * Nc / nc))
    else:
        # shifts = np.array([*shifts[:-1][::-1],shifts[-1]])
        shifts = np.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        nc, nr, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nr, Nc, Nd = np.meshgrid(Nr, Nc, Nd)
        Greg = src_freq * np.exp(-1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = ifftn(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(np.int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(np.int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(np.int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h - 1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w - 1, np.newaxis]
            if is3D:
                new_img[:, :, :max_d] = new_img[:, :, max_d]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d - 1]

    return new_img


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
                img[min_h:] = img[min_h - 1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w - 1, np.newaxis]

    return img


def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image
    Args:
        img:ndarray 2D
            image that needs to be slices
        windowSize: tuple
            dimension of the patch
        strides: tuple
            stride in wach dimension
     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix
              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])


def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.
    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is
    Copyright (C) 2011, the scikit-image team
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Args:
        src_image : ndarray
            Reference image.
        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.
        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)
        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.
        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False
    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.
        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).
    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"
     ValueError "Error: images must really be same size for "
                         "register_translation"
     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."
    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if opencv:
            src_freq_1 = fftn(
                src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = fftn(
                np.asarray(target_image), flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
            target_freq = np.array(
                target_freq, dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(
                src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(
                target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:
        image_product_cv = np.dstack(
            [np.real(image_product), np.imag(image_product)])
        cross_correlation = fftn(
            image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,
                            :, 0] + 1j * cross_correlation[:, :, 1]
    else:
        cross_correlation = ifftn(image_product)

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2))
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(
            np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).
    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.
    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is
    Copyright (C) 2011, the scikit-image team
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.
        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.
        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.
        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)
    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
            np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
            (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
            (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(old_div(data.shape[2], 2))))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes=[1, 0])
    output = np.tensordot(output, col_kernel, axes=[1, 0])

    if data.ndim > 2:
        # import pdb
        # pdb.set_trace()
        output = np.tensordot(output, pln_kernel, axes=[1, 1])
    # output = row_kernel.dot(data).dot(col_kernel)
    return output
