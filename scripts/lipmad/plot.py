#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
import os
import numpy as np
# from pathlib import Path
import pathlib
#
import pandas as pd
# from photutils.segmentation import SegmentationImage

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # GRIDSPEC !

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpl_patches

import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import TwoSlopeNorm

import io as sysio
from urllib.request import urlopen, Request
from PIL import Image as PILImage

from pyproj import Geod
import cartopy.crs as ccrs

import cartopy.io.img_tiles as cimgt
from shapely.geometry.point import Point
import geopandas as gpd

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import Normalize
from astropy.nddata import extract_array

from . import base_conf
from . import io

from .conversions import convert_mnk_to_kmn, convert_kmn_to_mnk

# matplotlib parameter
mpl.use('Qt5Agg')
mpl.rc("lines", linewidth=1.2)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('figure', dpi=150, facecolor='w', edgecolor='k')
mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')
plt.rcParams['text.usetex'] = True


def image_spoof(self, tile):  # this function pretends not to be a Python script
    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header('User-agent', 'Anaconda 3')  # add user agent to request
    fh = urlopen(req)
    im_data = sysio.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = PILImage.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), 'lower'  # reformat for cartopy


def add_compass(ax, location, size=0.1):
    """ Add a simple compass to the ax """

    # North label
    ax.text(location[0], location[1] + size + 0.02, 'N', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.arrow(location[0], location[1], 0, 0.045, length_includes_head=True,
             head_width=0.02, head_length=0.1, overhang=.1,
             facecolor='k',
             transform=ax.transAxes)


def get_dist_at_loc(bounds=None, pos=None):
    patch = None
    if bounds is not None:

        # Manually create the coordinates of the bounding box
        bbox_coords = np.array([
            [bounds[0], bounds[1]],  # Bottom-left
            [bounds[2], bounds[1]],  # Bottom-right
            [bounds[2], bounds[3]],  # Top-right
            [bounds[0], bounds[3]],  # Top-left
            [bounds[0], bounds[1]]  # Close the loop
        ])

        # Convert coordinates to a Path, making sure to close the loop
        path = Path(bbox_coords, closed=True)
        patch = PathPatch(path, facecolor='none', edgecolor='k', alpha=1.)  # No fill, just the edge

        lon_avg = (bounds[0] + bounds[2]) / 2
        lat_avg = (bounds[1] + bounds[3]) / 2
    else:
        lon_avg = pos[0]
        lat_avg = pos[1]

    points = gpd.GeoSeries(data=[Point(lon_avg - 0.5, lat_avg),
                                 Point(lon_avg + 0.5, lat_avg)],
                           crs=4326)  # Geographic WGS 84 - degrees

    points = points.to_crs(32619)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])

    return patch, distance_meters


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


class PlotResults:
    def __init__(self,
                 log: logging.Logger = base_conf.log,
                 log_level: int = 20):
        """ Constructor with default values """

        # set log level
        log.setLevel(log_level)

        self.log = log
        anf_shp_folder = pathlib.Path('../../data/shp_files/R02/')
        anf_shp_file1 = anf_shp_folder / 'MANZANA_SIN_INF_C17.shp'
        anf_shp_file2 = anf_shp_folder / 'MANZANA_IND_C17.shp'
        anf_shp_file3 = anf_shp_folder / 'CALLES_PAIS_C17.shp'

        gdf_shp_1 = io.read_geojson(anf_shp_file1)
        gdf_shp_2 = io.read_geojson(anf_shp_file2)
        gdf_shp_3 = io.read_geojson(anf_shp_file3)

        self.anf_gdf = pd.concat([gdf_shp_1.geometry,
                                  gdf_shp_2.geometry])

        self.anf_gdf_incl_streets = pd.concat([gdf_shp_1.geometry,
                                               gdf_shp_2.geometry,
                                               gdf_shp_3.geometry])

    def plot_coastal_image(self, image):
        # load image
        image.load_image_calibrated()
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        print(img_cal_undist.shape)
        import tifffile
        tifffile.imwrite(image.plot_dir / f'output_{image.name}.tif',
                         convert_mnk_to_kmn(img_cal_undist.astype(np.float32)))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        img = self.contrast_stretch(img_cal_undist, (1., 99.2), clip=True)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()

        fname = image.plot_dir / f'coastal_view_image_{image.name}.png'
        self.save_plot(fig, fname)

    def plot_coastal_image_and_map(self, image):

        # load image
        image.load_image_calibrated()
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)

        self.log.info(f'> Plot on OSM map')

        # load area geojson
        gdf = io.read_geojson(image.area_geojson)
        # print(gdf.columns)

        img_mask = (gdf.File_Name == image.name) & (gdf.geometry.geom_type == 'Point')
        img_gdf = gdf[img_mask]
        drone_pos = [img_gdf.geometry.x.values[0], img_gdf.geometry.y.values[0]]
        drone_orient = img_gdf.FlightYawDegree.values[0]
        fov = image.h_fov_deg

        _, distance_meters = get_dist_at_loc(None, drone_pos)

        # get map extent
        expanded_extent = calculate_expanded_bbox(gdf, 2500,
                                                  drone_pos)
        # print(expanded_extent)
        # load the image
        cimgt.OSM.get_image = image_spoof
        osm_tiles = cimgt.OSM()

        # SATELLITE STYLE
        # cimgt.QuadtreeTiles.get_image = image_spoof
        # osm_tiles = cimgt.QuadtreeTiles()
        # Create a figure
        fig = plt.figure(figsize=(12, 6))
        # Define the GridSpec layout
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.8])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

        img = self.contrast_stretch(img_cal_undist, (1., 99.), clip=True)
        ax1.imshow(img)
        ax1.axis('off')

        ax2.add_image(osm_tiles, 16)
        ax2.set_extent(expanded_extent,
                       crs=ccrs.PlateCarree())

        # Plot drone's position
        ax2.plot(drone_pos[0], drone_pos[1], 'bo',
                 markersize=10, transform=ccrs.PlateCarree())

        # Compute the FoV polygon (a sector of a circle)
        num_points = 100
        angles = np.linspace(drone_orient - fov / 2, drone_orient + fov / 2, num_points)
        radii = 1500 / distance_meters  # Approximate radius for visualization purposes
        x_fov = drone_pos[0] + radii * np.sin(np.radians(angles))
        y_fov = drone_pos[1] + radii * np.cos(np.radians(angles))

        # Plot the FoV polygon on the map
        ax2.fill(np.concatenate([[drone_pos[0]], x_fov, [drone_pos[0]]]),
                 np.concatenate([[drone_pos[1]], y_fov, [drone_pos[1]]]),
                 color='green', alpha=0.3, transform=ccrs.PlateCarree())

        # Add gridlines with labels
        gl = ax2.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Add a scale bar
        scalebar = ScaleBar(distance_meters, units='m', location='lower left', scale_loc='bottom', length_fraction=0.1)
        ax2.add_artist(scalebar)

        # Add compass
        add_compass(ax2, (0.055, 0.135), size=0.045)  # Top-right corner

        plt.tight_layout()
        # plt.show()
        fname = image.plot_dir / f'combined_coastal_view_image_map_{image.name}.png'
        self.save_plot(fig, fname)

    def plot_table_data(self, image_idx, image, suffix='_image'):
        self.log.info(f'  Load processed data')
        data_file = image.cache_dir / f'area_data_processed_{image.folder_name}.tab'
        data = io.load_data(data_file)

        if suffix == '_image':
            data_df = data.query(f'imageID == "{image.name}"')
            fname1 = image.plot_dir / f'colors_and_outliers_image_{image.name}.png'
            fname1_avg = image.plot_dir / f'colors_and_outliers_avg_image_{image.name}.png'
            fname2 = image.plot_dir / f'mags_image_{image.name}.png'
        else:
            data_df = data
            fname1 = image.plot_dir / f'colors_and_outliers_area_{image.folder_name}.png'
            fname1_avg = image.plot_dir / f'colors_and_outliers_avg_area_{image.name}.png'

        df = data_df.copy()

        formula = "df['B_flux'] / (df['G_flux'] + df['R_flux'])"
        df = df.assign(sum=eval(formula))

        formula = "df['B_avg'] / (df['G_avg'] + df['R_avg'])"
        df = df.assign(sum_avg=eval(formula))
        self.plot_color_ratio(df, fname1, suffix)

        if suffix == '_image':
            # self.plot_mags(df, fname2, suffix)
            # self.plot_color_ratio(df, fname1_avg, suffix, is_avg=True)
            # self.plot_outlier_in_image(df, image)
            # self.plot_color_sum_image(df, image)
            #
            # data = df.query(f'outlier{suffix} != 1')
            # self.plot_image_outlier_on_map(data, image)
            self.plot_color_B_over_R_image(df, image)

        else:
            data = df.query(f'outlier{suffix} != 1')
            self.plot_area_outlier_on_map(data, image)

    def plot_mags(self, df, fname, suffix, is_avg=False):

        x_col = 'B-G'
        y_col = 'B_mag'
        z_col = 'B-R'
        xlabel = '$(B-G)$ (mag)'
        ylabel = r'$B$ (mag)'
        zlabel = r'$(B-R)$ (mag)'

        self.log.info(f'  Plot flux vs colors')

        df.dropna(subset=[x_col, y_col, z_col], inplace=True)

        inliers_df = df.query(f'outlier{suffix} == 1')
        ooi_df = df.query(f'outlier{suffix} != 1')
        brightest_source_df = df.query(f'outlier{suffix} == 3')

        vmin, vmax = df[z_col].min(), df[z_col].max()
        center_point = 0.

        if vmax <= center_point or vmin >= center_point:
            center_point = (vmax + vmin) / 2.
        # print(vmin, vmax, center_point)

        norm = MidpointNormalize(vcenter=center_point, vmin=vmin,
                                 vmax=vmax)
        cmap = plt.get_cmap('bwr')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Scatter plot for inliers
        scat = ax.scatter(inliers_df[x_col], inliers_df[y_col],
                          c=inliers_df[z_col], cmap=cmap, norm=norm,
                          marker='o', s=25, edgecolor='k', linewidth=0.5)

        # Scatter plot for outliers
        ax.scatter(ooi_df[x_col], ooi_df[y_col],
                   c=ooi_df[z_col], cmap=cmap, norm=norm,
                   marker='s', s=25, edgecolor='k', linewidth=0.5)

        ax.scatter(brightest_source_df[x_col], brightest_source_df[y_col], c=brightest_source_df[z_col],
                   norm=norm, cmap=cmap,
                   marker='D', s=50, edgecolor='k', linewidth=1.)

        # -----------------------------------------------------------------------------
        # Make color bar
        # -----------------------------------------------------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075)
        cbar = plt.colorbar(scat, cax=cax, orientation='horizontal',
                            extend='both'
                            )

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1.,
            color='k')

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        cbar.ax.set_xlabel(zlabel)

        plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
        cbar.update_ticks()

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )
        ax.set_xlim(xmin=np.min(df[x_col]) - 0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.yaxis.set_inverted(True)
        self.save_plot(fig, fname)
        # plt.show()

    def plot_color_ratio(self, df, fname, suffix, is_avg=False):

        x_col = 'B/G'
        y_col = 'G/R'
        z_col = 'B/R'
        xlabel = 'B/G'
        ylabel = 'G/R'
        zlabel = 'B/R'
        if is_avg:
            x_col = 'B/G_avg'
            y_col = 'G/R_avg'
            z_col = 'B/R_avg'
            xlabel = r'$\left< \mathrm{B/G} \right>$'
            ylabel = r'$\left< \mathrm{G/R} \right>$'
            zlabel = r'$\left< \mathrm{B/R} \right>$'

        df.dropna(subset=[x_col, y_col, z_col], inplace=True)

        inliers_df = df.query(f'outlier{suffix} == 1')
        ooi_df = df.query(f'outlier{suffix} != 1')
        brightest_source_df = df.query(f'outlier{suffix} == 3')

        self.log.info(f'  Plot color information and outlier data')
        # Define the position for the arrow to start at the top-left corner
        x_min = np.min(df[x_col])
        x_max = np.max(df[x_col])
        y_max = np.max(df[y_col])
        y_min = np.min(df[y_col])
        s = 2 if suffix == '_image' else 4

        vmin, vmax = df[z_col].min(), df[z_col].max()
        center_point = 1.

        if vmax <= center_point or vmin >= center_point:
            center_point = (vmax + vmin) / 2.
        # print(vmin, vmax, center_point)
        # norm = TwoSlopeNorm(vcenter=center_point, vmin=vmin, vmax=vmax)
        norm = MidpointNormalize(vcenter=center_point, vmin=vmin,
                                 vmax=vmax)
        cmap = plt.get_cmap('bwr_r')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Scatter plot for inliers
        scat = ax.scatter(inliers_df[x_col], inliers_df[y_col],
                          c=inliers_df[z_col], cmap=cmap, norm=norm,
                          marker='o', s=25, edgecolor='k', linewidth=0.5)

        # Scatter plot for outliers
        ax.scatter(ooi_df[x_col], ooi_df[y_col],
                   c=ooi_df[z_col], cmap=cmap, norm=norm,
                   marker='s', s=25, edgecolor='k', linewidth=0.5)

        ax.scatter(brightest_source_df[x_col], brightest_source_df[y_col], c=brightest_source_df[z_col],
                   norm=norm, cmap=cmap,
                   marker='D', s=50, edgecolor='k', linewidth=1.)

        # Adding arrows
        # Example fixed positions in axis coordinates (0, 0 is bottom-left and 1, 1 is top-right)
        arrow1_x = 0.025
        arrow1_y1 = 0.75
        arrow1_y2 = 0.95

        arrow2_x1 = 0.2
        arrow2_x2 = 0.025
        arrow2_y = 0.95

        # Adding arrows with fixed positions relative to the axes
        ax.annotate('', xy=(arrow1_x, arrow1_y1), xytext=(arrow1_x, arrow1_y2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    ha='center', va='top', xycoords='axes fraction')

        ax.annotate('', xy=(arrow2_x1, arrow2_y), xytext=(arrow2_x2, arrow2_y),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    ha='left', va='center', xycoords='axes fraction')

        # Adding labels to arrows with fixed positions relative to the axes
        ax.text(arrow1_x, arrow1_y1 - 0.015, 'Redder', rotation=90, va='top', ha='center', transform=ax.transAxes)
        ax.text(arrow2_x1 + 0.015, arrow2_y, 'Bluer', va='center', ha='left', transform=ax.transAxes)

        # -----------------------------------------------------------------------------
        # Make color bar
        # -----------------------------------------------------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075)
        cbar = plt.colorbar(scat, cax=cax, orientation='horizontal',
                            extend='both'
                            )

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1.,
            color='k')

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        cbar.ax.set_xlabel(zlabel)

        plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
        cbar.update_ticks()

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )
        ax.set_xlim(xmin=np.min(df[x_col]) - 0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # fig.tight_layout()

        self.save_plot(fig, fname)
        # plt.show()

    def plot_segm_outlier_closeup(self, image):
        """"""

        self.log.info(f'  Load processed data')
        data_file = image.cache_dir / f'area_data_processed_{image.folder_name}.tab'
        data = io.load_data(data_file)

        df = data[data.imageID == image.name]

        self.log.info(f'> Plot segment closeup')
        area_gdf = io.read_geojson(image.area_geojson)

        # get bounds from gdf
        area_bounds = area_gdf.total_bounds
        _, distance_meters = get_dist_at_loc(area_bounds)

        # get map extent
        area_extent = calculate_expanded_bbox(area_gdf, 1000)

        segm_gdf = io.read_geojson(image.segm_img_geojson)

        outlier_mask_image = df.outlier_image != 1
        outlier_mask_area = df.outlier_area != 1

        # print(df[outlier_mask_image])
        # print(df[outlier_mask_area])
        # print(segm_gdf)
        #
        cutout_width = cutout_height = 417  # ~25m at 120m altitude
        cutout_shape = (cutout_height, cutout_width)

        image.load_image_calibrated()
        image.load_detection_data()
        img_cal = image.image_RGB_cal
        segm_map = image.segment_map
        segm_map_undist = image.segment_map_undist
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        segm_data = img_cal_undist
        # print(segm_data.shape)

        mask_list = [(outlier_mask_image, 'outlier_image'), (outlier_mask_area, 'outlier_area')]
        for m in mask_list:
            df2 = df[m[0]].reset_index(drop=True)
            current_df = (df2.sort_values([m[1], 'G_flux', 'area'], ascending=[False, False, False]))
            # print(current_df)
            for index, row in current_df.head(25).iterrows():
                fname = image.segm_plot_dir / f'segm_{m[1]}_{row[m[1]]}_{row.label}_{row.imageID}.png'

                row_segm_mask = (segm_gdf.SegmentLabel == row.label) & (segm_gdf.geometry.geom_type == 'Point')
                row_segm_gdf = segm_gdf[row_segm_mask]
                # print(row_segm_gdf)

                drone_pos = [row_segm_gdf.geometry.x.values[0], row_segm_gdf.geometry.y.values[0]]

                _, distance_meters = get_dist_at_loc(None, drone_pos)

                # get map extent
                expanded_extent = calculate_expanded_bbox(segm_gdf, 150,
                                                          drone_pos)

                # load the image
                cimgt.OSM.get_image = image_spoof
                osm_tiles = cimgt.OSM()

                # SATELLITE STYLE
                # cimgt.QuadtreeTiles.get_image = image_spoof
                # osm_tiles = cimgt.QuadtreeTiles()

                # print(expanded_extent)
                label_id = row['label']
                segmobj = segm_map.segments[segm_map.get_index(label_id)]
                mask = segmobj.data_ma
                sliced = (slice(None),) + segmobj.slices

                # print(img_cal.shape)
                if image.rgb_mask is not None:
                    img_cal[:, image.rgb_mask] = np.nan

                segm_image_in = img_cal[sliced]
                segm_image_in[segm_image_in < 0.] = np.nan

                segm_image = segm_image_in * mask
                segm_image = np.where(mask, segm_image, np.nan)
                segm_image = np.where(segm_image != 0., segm_image, np.nan)
                segm_mean = np.nanmean(segm_image, axis=(1, 2))
                # print(segm_mean)
                # ratio_B2R = np.divide(segm_image[2, :, :], segm_image[0, :, :])
                # ratio_B2R_avg = np.nanmean(ratio_B2R, axis=None)
                # # print(ratio_B2R_avg)
                # plt.figure()
                # plt.imshow(self.contrast_stretch(convert_kmn_to_mnk(segm_image_in), (1., 99.5), clip=True))
                mosaic = """
                        AB
                        AC
                        """
                fig = plt.figure(figsize=(10, 6), layout='compressed')
                # gs = gridspec.GridSpec(2, 2)
                ax_dict = fig.subplot_mosaic(mosaic, width_ratios=[2, 1], per_subplot_kw={
                    "C": {"projection": ccrs.PlateCarree()}})

                # plt.show()
                # Get center position (in pixel coordinates)
                center_y, center_x = row.ycentroid_undist, row.xcentroid_undist

                bbox_xmin = row.bbox_xmin_undist
                bbox_xmax = row.bbox_xmax_undist
                bbox_ymin = row.bbox_ymin_undist
                bbox_ymax = row.bbox_ymax_undist

                # Center position as a tuple
                center = (center_y, center_x)
                # print(center, cutout_shape)
                # Initialize a list to store cutouts for each band
                combined_cutout = np.zeros((segm_data.shape[2], *cutout_shape), dtype=segm_data.dtype)

                # Extract the larger cutout for each band
                for i in range(segm_data.shape[2]):
                    combined_cutout[i] = extract_array(segm_data[:, :, i], cutout_shape, center)

                # Display the results
                # fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                combined_cutout = convert_kmn_to_mnk(combined_cutout)
                img = self.contrast_stretch(combined_cutout, (1., 99.5), clip=True)

                # Subplot for the image
                # ax0 = fig.add_subplot(gs[:2, 0])
                ax_dict['A'].imshow(img, zorder=-1)

                ax_dict['A'].axes.get_xaxis().set_visible(False)
                ax_dict['A'].axes.get_yaxis().set_visible(False)

                # Calculate the shift in coordinates
                shift_x = center_x - cutout_width // 2
                shift_y = center_y - cutout_height // 2

                # Adjust the bounding box coordinates for the cutout
                adjusted_xmin = bbox_xmin - shift_x
                adjusted_ymin = bbox_ymin - shift_y
                adjusted_width = bbox_xmax - bbox_xmin
                adjusted_height = bbox_ymax - bbox_ymin

                # Create a rectangle patch
                rect = mpl_patches.Rectangle(
                    (adjusted_xmin, adjusted_ymin),  # (x, y)
                    adjusted_width,  # width
                    adjusted_height,  # height
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                # print(rect)
                # rect.set_edgecolor('r')

                # Add the rectangle to the Axes
                ax_dict['A'].add_patch(rect)

                # ax1 = fig.add_subplot(gs[1, 1:2])
                segm_image *= 1000.
                kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=30, ec="k")
                ax_dict['B'].hist(segm_image[0, :, :].flatten(), color='r', **kwargs)
                ax_dict['B'].hist(segm_image[1, :, :].flatten(), color='g', **kwargs)
                ax_dict['B'].hist(segm_image[2, :, :].flatten(), color='b', **kwargs)
                ax_dict['B'].set_xlabel(r'Irradiance (mW/m²)')
                ax_dict['B'].set_ylabel(r'Frequency')

                # ax2 = fig.add_subplot(gs[0:1, 1])
                ax_dict['C'].add_image(osm_tiles, 19)
                ax_dict['C'].set_extent(expanded_extent, crs=ccrs.PlateCarree())
                for idx, row2 in row_segm_gdf.iterrows():
                    if row2.geometry.geom_type == 'Point':
                        # Calculate the circle in the projection space
                        radius_deg = 10. / distance_meters

                        # Draw a circle around the point with the calculated radius
                        circle = Point(row2.geometry.x, row2.geometry.y).buffer(radius_deg,
                                                                                resolution=16)

                        # Add circle to plot
                        ax_dict['C'].add_geometries([circle], ccrs.PlateCarree(),
                                                    facecolor='red',
                                                    edgecolor='r', alpha=0.5)

                # Add gridlines with labels
                gl = ax_dict['C'].gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False

                # Add a scale bar
                scalebar = ScaleBar(distance_meters, units='m', location='lower left', scale_loc='bottom',
                                    length_fraction=0.1)
                ax_dict['C'].add_artist(scalebar)
                # Add compass
                add_compass(ax_dict['C'], (0.055, 0.2), size=0.045)  # Top-right corner

                self.save_plot(fig, fname)

    def plot_area_outlier_on_map(self, df, image):
        """"""
        self.log.info(f'  Plot area outlier on map')

        area_gdf = io.read_geojson(image.area_geojson)

        area_bounds = area_gdf.total_bounds
        image_extent = calculate_expanded_bbox(area_gdf, 150)
        patch, distance_meters = get_dist_at_loc(area_bounds)

        segm_gdf = io.read_geojson(image.segm_area_geojson)
        segm_outlier_gdf = segm_gdf[segm_gdf['SegmentLabel'].isin(df['label'].values)]

        # load the image
        cimgt.OSM.get_image = image_spoof
        osm_tiles = cimgt.OSM()

        # SATELLITE STYLE
        # cimgt.QuadtreeTiles.get_image = image_spoof
        # osm_tiles = cimgt.QuadtreeTiles()

        # Set up the plot
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=(10, 6))

        ax.add_image(osm_tiles, 19)
        ax.set_extent(image_extent, crs=ccrs.PlateCarree())
        ax.add_patch(patch)
        for img_idx, img_row in segm_outlier_gdf.iterrows():
            if img_row.geometry.geom_type == 'Point':
                # Calculate the circle in the projection space
                radius_deg = 5. / distance_meters  # Very rough approximation: degrees = meters / (meters per degree)

                # Draw a circle around the point with the calculated radius
                circle = Point(img_row.geometry.x, img_row.geometry.y).buffer(radius_deg,
                                                                              resolution=16)

                # Add circle to plot
                ax.add_geometries([circle], ccrs.PlateCarree(),
                                  facecolor='red',
                                  edgecolor='r', alpha=0.5)

        # add scalebar
        ax.add_artist(ScaleBar(distance_meters, location="lower left"))

        # Add compass
        add_compass(ax, (0.075, 0.135), size=0.045)  # Top-right corner

        a = inset_axes(ax, width="100%", height='100%', loc='lower right',
                       bbox_to_anchor=(.5, .0, 1., 1.),
                       bbox_transform=ax.transAxes,
                       borderpad=0.)

        self.anf_gdf.plot(figsize=(20, 20), cmap='Blues', ax=a)

        # Add the bounding box patch to the plot
        patch, distance_meters = get_dist_at_loc(area_bounds)
        a.add_patch(patch)

        a.set_xlim(-70.45, -70.36)
        a.set_ylim(-23.75, -23.475)
        a.set_xticks([])
        a.set_yticks([])

        # fig.tight_layout()
        # plt.show()

        fname = image.plot_dir / f'outlier_geolocation_area_{image.folder_name}.png'
        self.save_plot(fig, fname)

    def plot_image_outlier_on_map(self, df, image):
        """"""
        # print(df['label'])
        self.log.info(f'  Plot image outlier on map')

        area_gdf = io.read_geojson(image.area_geojson)

        # get current image data
        img_mask = area_gdf.File_Name == image.name
        img_gdf = area_gdf[img_mask]

        # get bounds from gdf
        img_bounds = img_gdf.total_bounds
        area_bounds = area_gdf.total_bounds
        _, distance_meters = get_dist_at_loc(img_bounds)

        segm_gdf = io.read_geojson(image.segm_img_geojson)
        # outlier_mask = segm_gdf.Outlier == -1
        # segm_outlier_gdf = segm_gdf[outlier_mask]
        segm_outlier_gdf = segm_gdf[segm_gdf['SegmentLabel'].isin(df['label'].values)]
        # print(segm_gdf)
        # print(segm_outlier_gdf)

        # get map extent
        image_extent = calculate_expanded_bbox(img_gdf, 100)
        area_extent = calculate_expanded_bbox(area_gdf, 2000)
        #
        # print(img_bounds, image_extent)
        # print(area_bounds, area_extent)

        # load the image
        cimgt.OSM.get_image = image_spoof
        osm_tiles = cimgt.OSM()

        # SATELLITE STYLE
        # cimgt.QuadtreeTiles.get_image = image_spoof
        # osm_tiles = cimgt.QuadtreeTiles()

        # Set up the plot
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=(10, 6))

        ax.add_image(osm_tiles, 19)
        ax.set_extent(image_extent, crs=ccrs.PlateCarree())
        for idx, row in segm_outlier_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                # Calculate the circle in the projection space
                radius_deg = 5. / distance_meters  # Very rough approximation: degrees = meters / (meters per degree)

                # Draw a circle around the point with the calculated radius
                circle = Point(row.geometry.x, row.geometry.y).buffer(radius_deg,
                                                                      resolution=16)

                # Add circle to plot
                ax.add_geometries([circle], ccrs.PlateCarree(),
                                  facecolor='red',
                                  edgecolor='r', alpha=0.5)
        for idx, row in img_gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                ax.plot(x, y, 'k--', alpha=0.75, lw=1, transform=ccrs.Geodetic())

        # add scalebar
        ax.add_artist(ScaleBar(distance_meters, location="lower left"))

        # Add compass
        add_compass(ax, (0.075, 0.135), size=0.045)  # Top-right corner

        a = inset_axes(ax, width="100%", height='50%', loc='upper right',
                       bbox_to_anchor=(.5, .0, 1., 1.),
                       bbox_transform=ax.transAxes,
                       borderpad=0.)

        self.anf_gdf_incl_streets.plot(figsize=(20, 20), cmap='YlOrRd', ax=a, zorder=-1)

        # Add the bounding box patch to the plot
        patch, distance_meters = get_dist_at_loc(img_bounds)
        a.add_patch(patch)

        a.set_xlim(area_extent[0], area_extent[1])
        a.set_ylim(area_extent[2], area_extent[3])
        a.set_xticks([])
        a.set_yticks([])

        a = inset_axes(ax, width="100%", height='50%', loc='lower right',
                       bbox_to_anchor=(.5, .0, 1., 1.),
                       bbox_transform=ax.transAxes,
                       borderpad=0.)

        self.anf_gdf.plot(figsize=(20, 20), cmap='Blues', ax=a)

        # Add the bounding box patch to the plot
        patch, distance_meters = get_dist_at_loc(area_bounds)
        a.add_patch(patch)

        a.set_xlim(-70.45, -70.36)
        a.set_ylim(-23.75, -23.475)
        a.set_xticks([])
        a.set_yticks([])

        # plt.show()
        fname = image.plot_dir / f'outlier_geolocation_image_{image.name}.png'
        self.save_plot(fig, fname)

    def plot_color_sum_image(self, df, image):
        self.log.info(f'  Plot B/(R+G) image')

        # load image
        image.load_image_calibrated()
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        img_in = self.contrast_stretch(img_cal_undist[:, :, 1], (1., 99.5), clip=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        img_cal_undist = np.where(img_cal_undist <= 0., np.nan, img_cal_undist)
        z_img = img_cal_undist[:, :, 2] / (img_cal_undist[:, :, 0] + img_cal_undist[:, :, 1])
        # z_img = img_cal_undist[:, :, 2] / (img_cal_undist[:, :, 0])
        vmin, vmax = 0, 1

        center_point = 0.25  #np.nanmedian(z_img)  # Change this to any value you need as the center
        norm = TwoSlopeNorm(vcenter=center_point, vmin=vmin, vmax=vmax)
        # norm = MidpointNormalize(vcenter=center_point, vmin=vmin,
        #                          vmax=vmax)
        cmap = plt.get_cmap('coolwarm_r')
        ax.imshow(img_in, cmap='Greys_r')
        img = ax.imshow(z_img, cmap=cmap, norm=norm, zorder=2)
        # -----------------------------------------------------------------------------
        # Make color bar
        # -----------------------------------------------------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075)
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal',
                            extend='both'
                            )

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1.,
            color='k')

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        cbar.ax.set_xlabel('B/(R+G)')

        plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
        cbar.update_ticks()

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()

        fname = image.plot_dir / f'B_over_R_plus_G_{image.name}.png'
        # self.save_plot(fig, fname)
        # plt.show()

    def plot_color_B_over_R_image(self, df, image):
        self.log.info(f'  Plot B/R image')

        # load image
        image.load_image_calibrated()
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        img_in = self.contrast_stretch(img_cal_undist[:, :, 1], (1., 99.5), clip=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        img_cal_undist = np.where(img_cal_undist <= 0., np.nan, img_cal_undist)
        z_img = img_cal_undist[:, :, 2] / img_cal_undist[:, :, 0]
        # z_img = img_cal_undist[:, :, 2] / (img_cal_undist[:, :, 0])
        vmin, vmax = 0, 2

        center_point = 0.5  # np.nanmedian(z_img)  # Change this to any value you need as the center
        norm = TwoSlopeNorm(vcenter=center_point, vmin=vmin, vmax=vmax)
        # norm = MidpointNormalize(vcenter=center_point, vmin=vmin,
        #                          vmax=vmax)
        cmap = plt.get_cmap('RdYlBu')
        img = ax.imshow(z_img, cmap=cmap, norm=norm)

        ax.imshow(img_in, cmap='Greys_r', zorder=-1)

        # -----------------------------------------------------------------------------
        # Make color bar
        # -----------------------------------------------------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075)
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal',
                            extend='both'
                            )

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1.,
            color='k')

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        cbar.ax.set_xlabel('B/R')

        plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
        cbar.update_ticks()

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()

        fname = image.plot_dir / f'B_over_R_{image.name}.png'
        # plt.show()
        self.save_plot(fig, fname)

    def plot_outlier_in_image(self, df, image):
        """"""

        # load image
        image.load_image_calibrated()
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # img_cal_undist = np.where(img_cal_undist <= 0., np.nan, img_cal_undist)
        # ax.imshow(img_cal_undist[:, :, 2] / (img_cal_undist[:, :, 0] + img_cal_undist[:, :, 1]))
        # plt.show()
        self.log.info(f'  Plot outlier in image')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        img = self.contrast_stretch(img_cal_undist, (1., 99.5), clip=True)
        ax.imshow(img)

        # Loop through the DataFrame and draw each rectangle
        for index, row in df.iterrows():
            # Create a rectangle patch
            rect = mpl_patches.Rectangle(
                (row['bbox_xmin_undist'], row['bbox_ymin_undist']),  # (x, y)
                row['bbox_xmax_undist'] - row['bbox_xmin_undist'],  # width
                row['bbox_ymax_undist'] - row['bbox_ymin_undist'],  # height
                linewidth=1, facecolor='none'
            )

            if row[f'outlier_image'] == -1:
                rect.set_edgecolor('r')
            elif row[f'outlier_image'] == 2:
                rect.set_edgecolor('y')
                rect.set_linestyle('--')
                rect.set_linewidth(1.5)
            elif row[f'outlier_image'] == 4:
                rect.set_edgecolor('b')
                rect.set_linestyle('-.')
                rect.set_linewidth(1.5)
            else:
                rect.set_edgecolor('magenta')
                # rect.set_alpha(0.75)

            # Add the rectangle to the Axes
            ax.add_patch(rect)

        ax.axis('off')
        fig.tight_layout()

        # plt.show()

        fname = image.plot_dir / f'outlier_image_{image.name}.png'
        self.save_plot(fig, fname)

    def plot_detection(self, image):
        """"""
        img_cal = image.image_RGB_cal
        img_gray = image.image_XYZ
        # print(img_gray.shape)
        img_bkg_gray = image.bkg_image
        segm_map = image.segment_map
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        img_gray_undist = image.undistort_data(img_gray, is_segm=True)
        img_bkg_gray_undist = image.undistort_data(img_bkg_gray, is_segm=True)
        segm_map_undist = image.segment_map_undist
        # print(img_gray_undist.shape, img_bkg_gray_undist.shape)

        fig, ax = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(10, 5))

        ax[0].imshow(img_gray_undist[:, :, 1], cmap='grey',
                     vmin=np.nanpercentile(img_gray_undist[:, :, 1], 1.5),
                     vmax=np.nanpercentile(img_gray_undist[:, :, 1], 99.5))
        ax[0].axis('off')
        ax[1].imshow(img_bkg_gray_undist,
                     vmin=np.nanpercentile(img_bkg_gray_undist, 1.5),
                     vmax=np.nanpercentile(img_bkg_gray_undist, 99.5))
        ax[1].axis('off')

        # ax[2].imshow(img_cal_undist[:, :, 1], cmap='grey',
        #              vmin=np.nanpercentile(img_cal_undist[:, :, 1], 1.5),
        #              vmax=np.nanpercentile(img_cal_undist[:, :, 1], 99.5))
        cmap = segm_map_undist.make_cmap(seed=12345)
        segm_map_data = segm_map_undist.data.astype(float)
        # segm_map_data[segm_map_data == 0] = np.nan
        ax[2].imshow(segm_map_data, cmap=cmap, interpolation='nearest')
        ax[2].axis('off')

        fig.tight_layout()

        # plt.show()

        fname = image.plot_dir / f'detection_combined_{image.name}.png'
        self.save_plot(fig, fname)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(img_gray_undist[:, :, 1], cmap='grey',
                  vmin=np.nanpercentile(img_gray_undist[:, :, 1], 1.5),
                  vmax=np.nanpercentile(img_gray_undist[:, :, 1], 99.5))
        ax.axis('off')
        fig.tight_layout()
        fname = image.plot_dir / f'detection_image_gray_{image.name}.png'
        self.save_plot(fig, fname)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(img_bkg_gray_undist,
                  vmin=np.nanpercentile(img_bkg_gray_undist, 1.5),
                  vmax=np.nanpercentile(img_bkg_gray_undist, 99.5))
        ax.axis('off')
        fig.tight_layout()
        fname = image.plot_dir / f'detection_image_bkg_{image.name}.png'
        self.save_plot(fig, fname)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        cmap = segm_map_undist.make_cmap(seed=12345)
        segm_map_data = segm_map_undist.data.astype(float)
        # segm_map_data[segm_map_data == 0] = np.nan
        ax.imshow(segm_map_data, cmap=cmap, interpolation='nearest')
        ax.axis('off')

        fig.tight_layout()
        fname = image.plot_dir / f'detection_image_segments_{image.name}.png'
        self.save_plot(fig, fname)

        # plt.show()

    def plot_as_rgb_image(self, image, include_segm=False):

        self.log.info(f'  Load processed data')
        data_file = image.cache_dir / f'area_data_processed_{image.folder_name}.tab'
        data = io.load_data(data_file)
        # print(data.columns)

        df = data.query(f'imageID == "{image.name}"')

        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        img = self.contrast_stretch(img_cal_undist, (1., 99.5), clip=True)
        ax.imshow(img)
        ax.axis('off')
        if include_segm:

            fname = image.plot_dir / f'image_color_with_segments_{image.name}.png'
            # Loop through the DataFrame and draw each rectangle
            for index, row in df.iterrows():
                # Create a rectangle patch
                rect = mpl_patches.Rectangle(
                    (row['bbox_xmin_undist'], row['bbox_ymin_undist']),  # (x, y)
                    row['bbox_xmax_undist'] - row['bbox_xmin_undist'],  # width
                    row['bbox_ymax_undist'] - row['bbox_ymin_undist'],  # height
                    linewidth=1, facecolor='none'
                )

                rect.set_edgecolor('magenta')
                # rect.set_alpha(0.75)

                # Add the rectangle to the Axes
                ax.add_patch(rect)

            fig.tight_layout()
        else:
            fname = image.plot_dir / f'detection_image_color_{image.name}.png'

        # plt.show()
        self.save_plot(fig, fname)

    def plot_image_loc_on_map(self, image_idx, image):
        """"""
        self.log.info(f'> Plot on OSM map')

        # load area geojson
        gdf = io.read_geojson(image.area_geojson)

        # get line string row
        line_mask = gdf.geometry.type == 'LineString'

        # get current image data
        current_img_mask = gdf.File_Name == image.name

        current_gdf = gdf[current_img_mask]
        remaining_gdf = gdf[~current_img_mask & ~line_mask]

        #
        line_gdf = gdf[line_mask]

        # get area bounds
        bounds = gdf.total_bounds

        patch, distance_meters = get_dist_at_loc(bounds)

        # get map extent
        expanded_extent = calculate_expanded_bbox(gdf, 150)

        # load the image
        cimgt.OSM.get_image = image_spoof
        osm_tiles = cimgt.OSM()

        # SATELLITE STYLE
        # cimgt.QuadtreeTiles.get_image = image_spoof
        # osm_tiles = cimgt.QuadtreeTiles()

        # Set up the plot
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=(10, 6))

        ax.add_image(osm_tiles, 19)
        ax.set_extent(expanded_extent, crs=ccrs.PlateCarree())

        # plot the mission path
        for idx, row in line_gdf.iterrows():
            x, y = row.geometry.xy
            ax.plot(x, y, 'y-', transform=ccrs.Geodetic())

        # first plot remaining polygons
        for idx, row in remaining_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                # print(row.geometry)
                ax.plot(row.geometry.x, row.geometry.y, 'ko', markersize=5,
                        transform=ccrs.Geodetic())
            elif row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                ax.plot(x, y, 'k-', alpha=0.125, transform=ccrs.Geodetic())
                ax.fill(x[::-1], y[::-1], 'black', alpha=0.125, transform=ccrs.Geodetic())

        # plot current image polygon
        for idx, row in current_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                # print(row.geometry)
                ax.plot(row.geometry.x, row.geometry.y, 'r*', markersize=10,
                        transform=ccrs.Geodetic())
            elif row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                ax.plot(x, y, 'r-', alpha=0.75, lw=3, transform=ccrs.Geodetic())
                ax.fill(x[::-1], y[::-1], 'b', alpha=0.25, transform=ccrs.Geodetic())

        # add scalebar
        ax.add_artist(ScaleBar(distance_meters, location="lower left"))

        # Add compass
        add_compass(ax, (0.075, 0.135), size=0.045)  # Top-right corner

        a = inset_axes(ax, width="100%", height='100%', loc='lower right',
                       bbox_to_anchor=(.5, .0, 1., 1.),
                       bbox_transform=ax.transAxes,
                       borderpad=0.)

        self.anf_gdf.plot(figsize=(20, 20), cmap='Blues', ax=a)

        # Add the bounding box patch to the plot
        a.add_patch(patch)

        a.set_xlim(-70.45, -70.36)
        a.set_ylim(-23.75, -23.475)
        a.set_xticks([])
        a.set_yticks([])
        # a.set_axis_off()

        # plt.show()

        fname = image.plot_dir / f'geolocation_image_{image.name}.png'
        self.save_plot(fig, fname)

    def plot_colors_and_outlier(self, df, mask, fname, suffix='_image'):

        param_list = [('coolwarm_r', 'B/R', 'B/R'),
                      ('plasma_r', 'G_mag_surf', 'G band surface brightness')]

        inliers_df = df[~mask]
        outlier_df = df[mask]

        x_col = 'B/G'
        y_col = 'G/R'

        # Define the position for the arrow to start at the top-left corner
        x_min = np.min(df[x_col])
        x_max = np.max(df[x_col])
        y_max = np.max(df[y_col])
        y_min = np.min(df[y_col])
        s = 2 if suffix == '_image' else 4
        self.log.info(f'  Plot color information and outlier data')

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
        for i, ax in enumerate(axs):
            color_map = param_list[i][0]
            z_col = param_list[i][1]

            # Define color map and norm for consistent coloring based on 'z'
            cmap = plt.cm.get_cmap(color_map, np.unique(df[z_col]).size)
            norm = plt.Normalize(vmin=df[z_col].min(), vmax=df[z_col].max())

            # Scatter plot for inliers
            scat = ax.scatter(inliers_df[x_col], inliers_df[y_col],
                              c=inliers_df[z_col], cmap=cmap, norm=norm,
                              marker='o', s=25, edgecolor=None)

            # Scatter plot for outliers
            ax.scatter(outlier_df[x_col], outlier_df[y_col],
                       c=outlier_df[z_col], cmap=cmap, norm=norm,
                       marker='*', s=100, edgecolor='k')

            if i == 0:
                # Adding arrows
                ax.annotate('', xy=(x_min - 0.05, y_max * 0.85),
                            xytext=(x_min - 0.05, y_max * 1.01),
                            arrowprops=dict(facecolor='black', shrink=0.05,
                                            width=2, headwidth=8),
                            ha='center', va='top')

                ax.annotate('', xy=(x_min * s, y_max * 1.01),
                            xytext=(x_min * 0.99, y_max * 1.01),
                            arrowprops=dict(facecolor='black', shrink=0.05,
                                            width=2, headwidth=8),
                            ha='left', va='center')

                # Adding labels to arrows
                ax.text(x_min - 0.05, y_max * 0.845, 'Redder', rotation=90,
                        va='top', ha='center')
                ax.text(x_min * s, y_max * 1.01, 'Bluer',
                        va='center', ha='left')

            # -----------------------------------------------------------------------------
            # Make color bar
            # -----------------------------------------------------------------------------
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.075)
            cbar = plt.colorbar(scat, cax=cax, orientation='horizontal',
                                extend='both'
                                )

            cbar.ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',
                left=True,  # ticks along the bottom edge are off
                right=True,  # ticks along the top edge are off
                top=True,
                bottom=True,
                width=1.,
                color='k')

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")

            cbar.ax.set_xlabel(param_list[i][2], fontsize='smaller')

            plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
            cbar.update_ticks()

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',  # inward or outward ticks
                left=True,  # ticks along the left edge are off
                right=True,  # ticks along the right edge are off
                top=True,  # ticks along the top edge are off
                bottom=True,  # ticks along the bottom edge are off
                width=1.,
                color='k'  # tick color
            )

            ax.set_xlim(xmin=np.min(df[x_col]) - 0.1)
            ax.set_xlabel(x_col)
            if i == 0:
                ax.set_ylabel(y_col)

        # plt.figure(figsize=(8, 6))
        # plt.scatter(df['B/G'], df['G/R'], c=df['B/R'], cmap='coolwarm_r', s=20)
        # # plt.title('Clusters formed by HDBSCAN after UMAP Reduction')
        # plt.xlabel('B/G')
        # plt.ylabel('G/R')
        # plt.colorbar()
        fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        self.save_plot(fig, fname)
        # plt.show()

        # # df.plot(ax=ax, kind='scatter', x='B/G_avg', y='G/R_avg')
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # df.plot(ax=ax, kind='scatter', y='B-R', x='G-R')
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # df.plot(ax=ax, kind='scatter', y='B-G', x='G-R')
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # df.plot(ax=ax, kind='scatter', y=['R_flux', 'B_flux'], x=['G_flux', 'G_flux'])
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # df.plot(ax=ax, kind='scatter', y='B_avg', x='G_avg')
        # plt.show()

        # # Unique cluster labels
        # cluster_labels = np.unique(df[f'cluster{suffix}'])
        # colors = ['blue', 'green', 'red']
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # for i, cluster in enumerate(cluster_labels):
        #     # Data for current cluster inliers
        #     cluster_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == 1)]
        #     ax.scatter(cluster_data['G-R'], cluster_data['B-G'],
        #                color=colors[i % len(colors)],  # Cycle through colors list
        #                label=f'Cluster {cluster}', alpha=0.5)
        #
        #     # Data for current cluster outliers
        #     outliers_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == -1)]
        #     ax.scatter(outliers_data['G-R'], outliers_data['B-G'],
        #                marker='x',  # X marker for outliers
        #                s=50,  # Size of the marker
        #                color=colors[i % len(colors)],  # Cycle through colors list
        #                label=f'Outliers in Cluster {cluster}')  # Label only once for legend
        #
        # ax.set_xlabel('B-G')
        # ax.set_ylabel('G-R')
        # plt.show()

    @staticmethod
    def save_plot(fig, fname):

        fig.savefig(fname, format='png')
        os.system(f'mogrify -trim {fname}')
        plt.close(fig)

    @staticmethod
    def contrast_stretch(data, q=(2, 98), clip=False):

        # data[data < 0] = 0.

        # Simple contrast stretching
        p2, p98 = np.nanpercentile(data, q=q)
        data_stretched = (data - p2) / (p98 - p2)

        if clip:
            return np.clip(data_stretched, 0, 1)
        else:
            return data_stretched

    def plot_g_band(self, image):
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)
        g_image = img_cal_undist[:, :, 1] * 1e6

        lin_cmap = cm.get_cmap('gist_ncar')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        img = ax.imshow(g_image, cmap=lin_cmap, vmin=0, vmax=4.)

        # -----------------------------------------------------------------------------
        # Make color bar
        # -----------------------------------------------------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075)
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal',
                            extend='max'
                            )

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1.,
            color='k')

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        cbar.ax.set_xlabel(r'Irradiance ($\mu Wm^{-2}$)')

        plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
        cbar.update_ticks()

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        fig.tight_layout()
        # plt.show()
        fname = image.plot_dir / f'irradiance_G_{image.name}.png'
        # plt.show()
        self.save_plot(fig, fname)

    def plot_images_colored(self, image):
        img_cal = image.image_RGB_cal
        img_cal_undist = image.undistort_data(img_cal, is_segm=False)

        X, Y = np.meshgrid(np.arange(img_cal_undist.shape[1] + 1),
                           np.arange(img_cal_undist.shape[0] + 1))

        # data_stretched = np.clip((data - p2) / (p98 - p2), 0, 1)
        cmap_list = ['Reds_r', 'Greens_r', 'Blues_r']
        for i in range(img_cal_undist.shape[2]):
            single_color = img_cal_undist[:, :, i]
            cmap = cmap_list[i]
            lin_cmap = cm.get_cmap(cmap)
            ############################################
            # prepare nonlinear norm (good for data viz)
            gamma = 0.1
            nonlin_norm = mcolors.PowerNorm(gamma=gamma,
                                            vmin=np.nanpercentile(single_color, 5),
                                            vmax=np.nanpercentile(single_color, 95))
            nonlin_norm.autoscale(single_color)

            # prepare nonlinear norm b (better for colorbar)
            gamma_b = 0.40
            nonlin_norm_b = mcolors.PowerNorm(gamma=gamma_b,
                                              vmin=np.nanpercentile(single_color, 1.5),
                                              vmax=np.nanpercentile(single_color, 99.5))
            nonlin_norm.autoscale(single_color)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            ax.imshow(single_color, cmap=lin_cmap, norm=nonlin_norm)
            # ax.pcolormesh(X, Y, single_color,
            #               cmap=lin_cmap,
            #               norm=nonlin_norm_b,
            #               shading='auto')
            # ax.axis('off')
            fig.tight_layout()
        plt.show()


def calculate_expanded_bbox(gdf, distance=1000, pos=None):
    """ Calculate and expand the bounding box of the GeoDataFrame by a given distance in meters using more accurate
    geodetic calculations."""
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    if pos is not None:
        bounds = [pos[0], pos[1], pos[0], pos[1]]

    geod = Geod(ellps='WGS84')  # Use WGS84 ellipsoid.

    # Calculate new bounds by moving outwards by 'distance' meters
    def expand_point(lon, lat, bearing, distance):
        new_lon, new_lat, _ = geod.fwd(lon, lat, bearing, distance)
        return new_lon, new_lat

    # Expand each corner of the bounding box
    min_lon, min_lat = expand_point(bounds[0], bounds[1], 225, distance)  # South-West corner
    max_lon, max_lat = expand_point(bounds[2], bounds[3], 45, distance)  # North-East corner

    return [min_lon, max_lon, min_lat, max_lat]
