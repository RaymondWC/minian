import sys
import os
import gc
import psutil
import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import bokeh.plotting as bpl
import dask.array as da
import pandas as pd
import dask
import datashader as ds
import itertools as itt
import papermill as pm
import ast
import functools as fct
from holoviews.operation.datashader import datashade, regrid, dynspread
from datashader.colors import Sets1to3
from dask.diagnostics import ProgressBar
from IPython.core.display import display, HTML
from dask.distributed import Client, progress, LocalCluster, fire_and_forget

minian_path = r'C:\Users\William Mau\Documents\GitHub\minian'
#dpath = r"C:\Users\William Mau\Documents\GitHub\minian\demo_movies"
dpath = r"C:\Users\William Mau\Documents\Projects\Cai lab\minian\Videos"
chunks = {"frame": 1000, "height": 50, "width": 50, "unit_id": 100}
subset = None
subset_mc = None
in_memory = True
interactive = True
output_size = 60
param_load_videos = {
    'pattern': 'msCam[0-9]+\.avi$',
    'dtype': np.float32,
    'in_memory': in_memory,
    'downsample': dict(frame=2),
    'downsample_strategy': 'subset'}
param_glow_removal = {
    'method': 'uniform',
    'wnd': 51}
param_brightspot_removal = {
    'thres': 2}
param_first_denoise = {
    'method': 'median',
    'ksize': 5}
param_second_denoise = {
    'method': 'gaussian',
    'sigmaX': 0,
    'ksize': (5, 5)}
param_estimate_shift = {
    'dim': 'frame',
    'on': 'first',
    'pad_f': 1,
    'pct_thres': 99.99}
param_normcorre = {
    'pw_rigid': False,
    'max_shifts': (5, 5),
    'gSig_filt': (3, 3),
    'strides': (48, 48),
    'overlaps': (24, 24),
    'max_deviation_rigid': 3,
    'border_nan': 'copy'
}
param_normcorre_pw = {
    'pw_rigid': True,
    'max_shifts': (5, 5),
    'gSig_filt': (3, 3),
    'strides': (48, 48),
    'overlaps': (24, 24),
    'max_deviation_rigid': 3,
    'border_nan': 'copy'
}
param_background_removal = {
    'method': 'tophat',
    'wnd': 10}
param_seeds_init = {
    'wnd_size': 2000,
    'method': 'rolling',
    'stp_size': 1000,
    'nchunk': 100,
    'max_wnd': 10}
param_gmm_refine = {
    'q': (0.1, 99.9),
    'n_components': 2,
    'valid_components': 1,
    'mean_mask': True}
param_pnr_refine = {
    'noise_freq': 0.06,
    'thres': 'auto'}
param_ks_refine = {
    'sig': 0.05}
param_seeds_merge = {
    'thres_dist': 5,
    'thres_corr': 0.7,
    'noise_freq': 'envelope'}
param_initialize = {
    'thres_corr': 0.8,
    'wnd': 10}
param_get_noise = {
    'noise_range': (0.06, 0.5),
    'noise_method': 'logmexp'}
param_first_spatial = {
    'dl_wnd': 5,
    'sparse_penal': 0.1,
    'update_background': False,
    'post_scal': True,
    'zero_thres': 'eps'}
param_first_temporal = {
    'noise_freq': 0.06,
    'sparse_penal': 1,
    'p': 2,
    'add_lag': 20,
    'use_spatial': False,
    'chk': chunks,
    'jac_thres': 0.1,
    'zero_thres': 1e-8,
    'max_iters': 200,
    'use_smooth': True,
    'scs_fallback': False,
    'post_scal': True}
param_first_merge = {
    'thres_corr': 0.9}
param_second_spatial = {
    'dl_wnd': 5,
    'sparse_penal': 0.05,
    'update_background': False,
    'post_scal': True,
    'zero_thres': 'eps'}
param_second_temporal = {
    'noise_freq': 0.06,
    'sparse_penal': 1,
    'p': 2,
    'add_lag': 20,
    'use_spatial': False,
    'chk': chunks,
    'jac_thres': 0.1,
    'zero_thres': 1e-8,
    'max_iters': 500,
    'use_smooth': True,
    'scs_fallback': False,
    'post_scal': True}
param_second_merge = {
    'thres_corr': 0.9}
param_save_minian = {
    'dpath': dpath,
    'fname': 'minian',
    'backend': 'zarr',
    'meta_dict': dict(session=-1, animal=-2),
    'overwrite': True}

sys.path.append(minian_path)
from minian.utilities import load_params, load_videos, scale_varr, scale_varr_da, save_variable, open_minian, save_minian, handle_crash
from minian.preprocessing import remove_brightspot, gradient_norm, denoise, remove_background, stripe_correction
from minian.motion_correction import estimate_shift_fft, apply_shifts, interpolate_frame, mask_shifts, normcorre_wrapper
from minian.initialization import seeds_init, gmm_refine, pnr_refine, intensity_refine, ks_refine, seeds_merge, initialize
from minian.cnmf import psd_welch, psd_fft, get_noise, update_spatial, update_temporal, unit_merge, smooth_sig
from minian.visualization import VArrayViewer, CNMFViewer, generate_videos, visualize_seeds, visualize_gmm_fit, visualize_spatial_update, visualize_temporal_update, roi_draw


dpath = os.path.abspath(dpath)
para_norm_list = ['meta_dict', 'chunks', 'subset']
for par_key in list(globals().keys()):
    if par_key in para_norm_list or par_key.startswith('param_'):
        globals()[par_key] = load_params(globals()[par_key])
if interactive:
    hv.notebook_extension('bokeh', width=100)
    pbar = ProgressBar()
    pbar.register()
else:
    hv.notebook_extension('matplotlib')

varr = load_videos(dpath, **param_load_videos)
varr_ref = varr
varr_ref = varr_ref.chunk(dict(frame=int(chunks['frame']/10), height=-1, width=-1))
varr_ref = remove_background(varr_ref, **param_glow_removal)
varr_ref = remove_brightspot(varr_ref, **param_brightspot_removal)
varr_ref = denoise(varr_ref, **param_first_denoise)

# Estimate shifts.
res = estimate_shift_fft(varr_ref.sel(subset_mc), **param_estimate_shift)
if in_memory:
    res = res.compute()
shifts = res.sel(variable = ['height', 'width'])
corr = res.sel(variable='corr')

#Apply shifts
varr_mc = apply_shifts(varr_ref, shifts)
varr_mc = varr_mc.ffill('height').bfill('height').ffill('width').bfill('width')
if in_memory:
    varr_mc = varr_mc.persist()

# Alternatively, use NoRMCorre
varr_nce_rig, shift_info, new_temp = normcorre_wrapper(np.asarray(varr_ref), **param_normcorre)
# varr_nce = xr.DataArray(varr_nce,dims=('frame','height','width'),
#                         coords=dict(frame=np.arange(varr_nce.shape[0]),
#                                     height=np.arange(varr_nce.shape[1]),
#                                     width=np.arange(varr_nce.shape[2])))

varr_nce_pw, _, _ = normcorre_wrapper(np.asarray(varr_ref), **param_normcorre_pw)

varr_ref = np.asarray(varr_ref)
varr_mc = np.asarray(varr_mc)

from pylab import tight_layout

raw_max = varr_ref.max(axis=0)
mc_max = varr_mc.max(axis=0)
rigid_max = varr_nce_rig.max(axis=0)
pw_max = varr_nce_pw.max(axis=0)

fig_max, ax_max = plt.subplots(2,2)
tight_layout()
ax_max[0,0].imshow(raw_max, vmin=-7, vmax=50)
ax_max[0,0].set_title('Raw')

ax_max[0,1].imshow(mc_max, vmin=-7, vmax=50)
ax_max[0,1].set_title('minian')

ax_max[1,0].imshow(rigid_max, vmin=-7, vmax=50)
ax_max[1,0].set_title('NoRMCorre, rigid')

ax_max[1,1].imshow(pw_max, vmin=-7, vmax=50)
ax_max[1,1].set_title('NoRMCorre, piecewise')

[axi.set_axis_off() for axi in ax_max.ravel()]

fig_diff, ax_diff = plt.subplots(3,2)
tight_layout()
ax_diff[0,0].imshow(raw_max - mc_max, vmin=-3, vmax=3)
ax_diff[0,0].set_title('Raw minus minian-mc')

ax_diff[1,0].imshow(raw_max - rigid_max, vmin=-3, vmax=3)
ax_diff[1,0].set_title('Raw minus NoRMCorre, rigid')

ax_diff[2,0].imshow(raw_max - pw_max, vmin=-3, vmax=3)
ax_diff[2,0].set_title('Raw minus NoRMCorre, piecewise')

ax_diff[1,1].imshow(mc_max - rigid_max, vmin=-3, vmax=3)
ax_diff[1,1].set_title('minian-mc minus NoRMCorre, rigid')

ax_diff[2,1].imshow(mc_max - pw_max, vmin=-3, vmax=3)
ax_diff[2,1].set_title('minian-mc minus NoRMCorre, piecewise')

ax_diff[0,1].set_axis_off()

[axi.set_axis_off() for axi in ax_diff.ravel()]

import matplotlib.animation as animation
fig, ax = plt.subplots(2,2)
tight_layout()

ims = []
for raw_frame, mc_frame, nce_frame, nce_pw_frame in zip(varr_ref, varr_mc, varr_nce_rig, varr_nce_pw):
    raw = ax[0,0].imshow(raw_frame, cmap='grey', animated=True)
    ax[0,0].set_title('Raw')

    mc = ax[0,1].imshow(mc_frame, cmap='grey', animated=True)
    ax[0,1].set_title('minian')

    nce_rig = ax[1,0].imshow(nce_frame, cmap='grey', animated=True)
    ax[1,0].set_title('NoRMCorre, rigid')

    nce_pw = ax[1,1].imshow(nce_pw_frame, cmap='grey', animated=True)
    ax[1,1].set_title('NoRMCorre, piecewise')

    [axi.set_axis_off() for axi in ax.ravel()]

    ims.append([raw, mc, nce_rig, nce_pw])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
ani.save('movies.mp4')
pass