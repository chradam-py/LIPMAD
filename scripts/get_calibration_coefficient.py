#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import copy
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy import constants as const
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from pathlib import Path
from astropy.wcs import FITSFixedWarning
import warnings
from scipy.interpolate import interp1d

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Qt5Agg')
mpl.rc('figure', dpi=150, facecolor='w', edgecolor='k')
mpl.rc("lines", linewidth=1.2)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')

warnings.filterwarnings(action='ignore', category=FITSFixedWarning)

# Define the data
ref_data = {
    'Band': ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'g', 'r', 'i', 'z'],
    'lambda_c': [0.36, 0.44, 0.55, 0.64, 0.79, 1.26, 1.60, 2.22, 0.52, 0.67, 0.79, 0.91],
    'dlambda/lambda': [0.15, 0.22, 0.16, 0.23, 0.19, 0.16, 0.23, 0.23, 0.14, 0.14, 0.16, 0.13],
    'F_0': [1810, 4260, 3640, 3080, 2550, 1600, 1080, 670, 3730, 4490, 4760, 4810],
}

# Create the DataFrame
ref_flux_df = pd.DataFrame(ref_data)

# Flux at 0 magnitude for each filter
flux_zero_values = ref_flux_df.set_index('Band')['F_0']
lambda_c_vals = ref_flux_df.set_index('Band')['lambda_c']
dlambda_lambda_vals = ref_flux_df.set_index('Band')['dlambda/lambda']

p = 101  # Atmospheric pressure in kPa

tau_a_04 = 0.110
tau_a_05 = 0.089  # Vertical aerosol optical depth at 0.5 μm
tau_a_06 = 0.056

tau_a_dict = dict(R=0.056, G=0.089, B=0.110)
alpha = 1.378  # Ångstrom coefficient

z_a = 0.965  # km
z_star = 0.095  # km

H_m = 8  # Scale height for molecules in km
H_a = 2  # Scale height for aerosols in km

transmittance_dict = dict(R=0.056, G=0.089, B=0.110,
                          alpha=1.378,  # Ångstrom coefficient
                          p=101,  # Atmospheric pressure in kPa
                          z_a=0.965,  # km
                          z_star=0.095,  # km
                          H_m=8,  # Scale height for molecules in km
                          H_a=2  # Scale height for aerosols in km
                          )


def main():
    """"""

    channels = ['R', 'G', 'B']
    results = {channel: {} for channel in channels}  # Dictionary to store results

    aperture_radius = 7  # Example radius
    annulus_radii = (1.5 * aperture_radius, 2. * aperture_radius)  # Inner and outer radii for annulus

    work_dir = Path('../../data/Camera_Calibration/')
    fits_dir = work_dir / '2024-02-08_starfield/iso800_t_5/ImageAnalysis/iso800_t_5/processed/pipelineout/'
    jpg_dir = work_dir / '2024-02-08_starfield/iso800_t_5/ImageAnalysis/iso800_t_5/processed/'

    base_filenames = sorted(get_base_filenames(fits_dir))
    # print(base_filenames)

    # load average RGB response
    response_filename = 'avg_RGB_response.fit'
    response_file_path = work_dir / response_filename
    response_df = load_response(response_file_path)
    print(response_df)
    resp_wl = response_df['lambda'].values

    # load color matching function
    cmf_filename = 'CIE_xyz_1931_2deg.csv'
    cmf_path = Path('../../data/Camera_Calibration/CIE1931') / cmf_filename

    cmf_df = pd.read_csv(cmf_path, names=['lambda', 'x_bar', 'y_bar', 'z_bar'],
                         header=None)
    print(cmf_df)
    ncmf_df = interpolate_data(cmf_df, 1, resp_wl/10)
    print(ncmf_df)
    # Dummy spectral data for demonstration
    wavelengths = resp_wl / 10
    V_lambda = ncmf_df['y_bar'].values

    # Example spectral sensitivity curves for R, G, B
    S_r = response_df['stdR']
    S_g = response_df['stdG']
    S_b = response_df['stdB']

    # Calculate coefficients
    C_r = np.trapz(S_r * V_lambda, wavelengths) / np.trapz(V_lambda, wavelengths)
    C_g = np.trapz(S_g * V_lambda, wavelengths) / np.trapz(V_lambda, wavelengths)
    C_b = np.trapz(S_b * V_lambda, wavelengths) / np.trapz(V_lambda, wavelengths)

    print(f"Red Coefficient: {C_r:.4f}, Green Coefficient: {C_g:.4f}, Blue Coefficient: {C_b:.4f}")
    rgb_coeff_dict = {'C_r': [C_r],
                      'C_g': [C_g],
                      'C_b': [C_b]}
    rgb_coeff_df = pd.DataFrame.from_dict(rgb_coeff_dict, orient='columns')

    # # Optionally, rename columns for clarity
    # coeff_df.rename(columns={'slope': 'c1',
    #                          'slope_err': 'c1_err',
    #                          # 'intercept': 'c0',
    #                          # 'intercept_err': 'c0_err'
    #                          }, inplace=True)

    # rgb_coeff_df = rgb_coeff_df.reset_index(names=['Band'])
    print(rgb_coeff_df)

    # save to disk
    rgb_coeff_df.to_csv(work_dir / f'luminance_coeff_RGB_mavic3.csv', index=False)

    # plot sensitivity curves
    fig, ax = plt.subplots(1, 1, sharex="all", sharey="all",
                           figsize=(10, 6))
    for channel in channels:
        resp_data = response_df[f'std{channel}'].values
        ax.plot(resp_wl / 10., resp_data, '-', color=f'{channel.lower()}')
    ax.tick_params(axis='both', direction='in', labelsize=14)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Rel. spectral sensitivity', fontsize=14)
    fig.tight_layout()
    fname = work_dir / 'avg_RGB_response.png'
    plt.savefig(fname, format='png', dpi=300)
    os.system(f'mogrify -trim {fname}')
    plt.close(fig)

    # load and prepare reference spectra
    spec_names_df, spec_data_df = load_and_prepare_spectra(work_dir)
    print(spec_names_df)
    print(spec_data_df)
    print(response_df)
    spec_data_cut = spec_data_df[spec_data_df['lambda'].isin(resp_wl)]

    # Colors for the fill plots
    fill_colors = ['red', 'green', 'blue']
    num_datasets = spec_names_df.shape[0]
    num_columns = 2
    import math
    num_rows = math.ceil(num_datasets / num_columns)  # Calculate number of rows needed

    # Setting up the figure and subplots
    fig, axs = plt.subplots(num_rows, num_columns, sharex='all', figsize=(12, 8))  # Creates a 4x2 grid of subplots
    # Flatten the axes array for easier iteration
    axs = axs.ravel()
    # Ensure data types are numpy arrays for matplotlib compatibility
    x_values = spec_data_df['lambda'].values / 10.

    # Plotting each subplot
    for i, ax in enumerate(axs.ravel()):
        if i < num_datasets:
            # Column name from df3
            column_name = spec_names_df.at[i, 'Name']
            ax2 = ax.twinx()  # Creates a twin of the original
            # Plotting and filling from df2
            for j, channel in enumerate(channels):
                y2_values = response_df[f'std{channel}'].values
                x2_values = response_df['lambda'].values / 10.
                ax2.fill_between(x2_values, 0, y2_values,
                                 color=channel.lower(), alpha=0.5)
                ax2.set_yticks([])  # Disable y-ticks for secondary axis

            # Plotting from df1
            y1_values = spec_data_df[f'm_{column_name}_'].values * 1e10
            ax.plot(x_values, y1_values, label='Kiehling (1987)', color='black')
            # Annotation instead of legend
            ax.annotate(column_name, xy=(0.025, 0.95), xycoords='axes fraction',
                        verticalalignment='top', horizontalalignment='left', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0))

            # Legend configuration
            if i == 0:
                ax.legend(loc='upper right', frameon=True, numpoints=14)
            ax.tick_params(axis='both', direction='in', labelsize=14)
            # Set title for each subplot using names from df3
            # ax.set_title(column_name)
        else:
            # Remove unused axes
            ax.set_visible(False)

    # Set a common Y label
    fig.text(0.025, 0.5, r'Flux (10$^{-10}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)',
             va='center', rotation='vertical', fontsize=14)
    # Set a common X label
    fig.text(0.5, 0.01, r'Wavelength (nm)', ha='center', fontsize=14)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust the left margin

    # Display the figure
    fname = work_dir / 'spectra.png'
    plt.savefig(fname, format='png')
    os.system(f'mogrify -trim {fname}')
    plt.close(fig)

    # loop files
    for i, base_filename in enumerate(base_filenames):
        jpg_img_file = jpg_dir / f'{base_filename}.jpg'
        jpg_out = jpg_dir / f'{base_filename}_annotated.png'
        info_fits = fits_dir / f'{base_filename}_G_out.fits'
        _, info_header = load_fits_image(info_fits)

        observation_time = info_header['DATE-OBS']  # Example time in ISO format
        latitude = info_header['SITELAT']
        longitude = info_header['SITELONG']
        spec_names_df['theta_z'] = spec_names_df.apply(calculate_zenith_angle_for_row,
                                                       args=(observation_time, latitude, longitude),
                                                       axis=1)
        resp_wl = response_df['lambda'].values

        spec_data_cut = spec_data_df[spec_data_df['lambda'].isin(resp_wl)]

        # loop channels
        for channel in channels:
            file_path = fits_dir / f'{base_filename}_{channel}_out.fits'
            image_data, header = load_fits_image(file_path)
            wcs_header = WCS(info_header)

            resp_data = response_df[f'std{channel}'].values
            # print(resp_wl, resp_data)

            T_m = calculate_molecular_transmittance(p, resp_wl * 1e-4)
            # print(T_m)
            T_a = calculate_aerosol_transmittance(tau_a_dict[channel], alpha, resp_wl * 1e-4)
            T_a_infinity = calculate_aerosol_transmittance_infinity(T_a, z_a, H_a)
            # print(T_a, T_a_infinity)

            # get total corrected flux
            spec_names_df[f'{channel}_flux_total'] = spec_names_df.apply(
                lambda row: calculate_total_flux(f'm_{row["Name"]}_', spec_data_cut,
                                                 resp_data, resp_wl, T_m, T_a_infinity,
                                                 row['theta_z'], z_star, H_a,
                                                 H_m),
                axis=1
            )

            spec_names_df = spec_names_df.reset_index(drop=True)
            spec_names_df['StarID'] = range(1, len(spec_names_df) + 1)
            pix_coords = convert_ra_dec_to_pixel(spec_names_df[['_RAJ2000', '_DEJ2000']].values,
                                                 wcs_header)
            from photutils.centroids import centroid_sources, centroid_com
            x, y = centroid_sources(image_data, pix_coords[:, 0], pix_coords[:, 1], box_size=11,
                                    centroid_func=centroid_com)
            pix_coords_centered = np.array(list(zip(x, y)))
            if channel == 'G' and i == 0:

                # Load the image from disk
                image = Image.open(jpg_img_file)
                p2, p98 = np.nanpercentile(np.array(image), (5, 95))

                fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(10, 6))

                ax.imshow(np.array(image), origin='lower')
                # ax.scatter(pix_coords_centered[:, 0], pix_coords_centered[:, 1])
                circle_rad = 10  # This is the radius, in points
                for idx, point in enumerate(pix_coords_centered):
                    name = spec_names_df.at[idx, 'Name']
                    offset = (-20, -20) if name == 'HR1654' else (10, -5)

                    ax.plot(point[0], point[1], 'o',
                            ms=circle_rad, mec='b', mfc='none', mew=2)
                    ax.annotate(name, xy=point, xytext=offset,
                                textcoords='offset points',
                                color='w', size='medium')
                add_points = [['Rigel', 1270, 1728],
                              ['Sirius', 1872, 1187],
                              ['Canopus', 1217, 136]]
                for idx, point in enumerate(add_points):
                    name = point[0]
                    offset = (12, 0)

                    ax.plot(point[1], point[2], 'o',
                            ms=circle_rad, mec='orange', mfc='none', mew=2)
                    ax.annotate(name, xy=point[1:], xytext=offset,
                                textcoords='offset points',
                                color='orange', size='medium')
                # ax.scatter(x, y, marker='+')
                ax.axis("off")
                fig.tight_layout()

                plt.savefig(jpg_out, format='png')
                os.system(f'mogrify -trim {jpg_out}')
                plt.close()

            imask = (image_data < 0)
            image_data[imask] = 0
            phot_table = perform_aperture_photometry(image_data, pix_coords_centered, aperture_radius, annulus_radii)

            for star_id, measured_flux in zip(spec_names_df['StarID'], phot_table['aper_sum_bkgsub']):
                # Retrieve reference flux for the star_id from std_df
                ref_flux = spec_names_df.loc[spec_names_df['StarID'] == star_id, f'{channel}_flux_total'].iloc[0]
                if star_id not in results[channel]:
                    results[channel][star_id] = []

                # Store both reference and measured fluxes
                results[channel][star_id].append([ref_flux, measured_flux])

        #
        # print(spec_names_df)

    # print(results)
    # Calculating mean values
    mean_results = {channel: {} for channel in channels}
    for channel in channels:
        for star_id, fluxes in results[channel].items():
            fluxes = np.array(fluxes)
            # print(fluxes)
            mean_results[channel][star_id] = [np.median(fluxes, axis=0),
                                              np.std(fluxes, axis=0)]
    # print(mean_results)
    # star_ids = set(star_id for channel in channels for star_id in results[channel])  # Get all unique star IDs
    #
    # for star_id in star_ids:
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #     fig.suptitle(f'Star ID: {star_id}')
    #
    #     for i, channel in enumerate(channels):
    #         if star_id in results[channel]:
    #             ref_fluxes, measured_fluxes = zip(*results[channel][star_id])
    #             axs[i].scatter(ref_fluxes, measured_fluxes, alpha=0.7)
    #             axs[i].set_title(f'Channel {channel}')
    #             axs[i].set_xlabel('Reference Flux')
    #             axs[i].set_ylabel('Measured Flux')
    #             axs[i].grid(True)
    #
    #     plt.tight_layout()
    #     plt.show()

    # # Prepare figure
    # plt.figure(figsize=(15, 5))
    #
    # # Iterate over each channel to create a subplot
    # for i, channel in enumerate(results.keys(), 1):
    #     ref_fluxes = []
    #     measured_fluxes = []
    #     # Create subplot for the channel
    #     plt.subplot(1, len(results), i)
    #
    #     # Extract fluxes for each star in the channel
    #     for star_id, flux_pairs in results[channel].items():
    #         for ref_flux, measured_flux in flux_pairs:
    #             ref_fluxes.append(ref_flux)
    #             measured_fluxes.append(measured_flux)
    #
    #     plt.scatter(ref_fluxes, measured_fluxes, alpha=0.7)
    #     # Extract fluxes for each star in the channel
    #     for star_id, flux_pairs in mean_results[channel].items():
    #         plt.scatter(flux_pairs[0][0], flux_pairs[0][1], color='k')
    #
    #     plt.title(f'{channel}-band Flux Comparison')
    #     plt.xlabel('Reference Flux')
    #     plt.ylabel('Measured Flux')
    #     plt.grid(True)
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    import lmfit
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    calibration_coefficients = {}
    for c, channel in enumerate(channels):
        ref_fluxes = []
        measured_fluxes = []
        y_errs = []  # Assuming you have uncertainties for measured_fluxes

        for star_id in mean_results[channel]:
            values = mean_results[channel][star_id]
            ref_fluxes.append(values[0][0])
            measured_fluxes.append(values[0][1])
            y_errs.append(values[1][1])  # Append uncertainty

        y = np.array(ref_fluxes)
        x = np.array(measured_fluxes)

        params = lmfit.Parameters()
        params.add('slope', value=1e-12)  # Initial value and a constraint to be positive
        params.add('intercept', value=0., vary=False)  # Initial value and a constraint to be positive

        result = lmfit.minimize(linear_model_lmfit, params, method='leastsq',
                                args=(x,), kws={'data': y, 'eps': None})
        print(lmfit.fit_report(result))

        # plt.figure()
        ax.plot(x, y * 1e7, 'o', label=f'{channel.upper()}', c=f'{channel.lower()}')
        ax.plot(x, linear_model_lmfit(result.params, x) * 1e7, 'k', c=f'{channel.lower()}')

        # inverse_coeff = 1 / coeff
        # inverse_error = coeff_error / (coeff ** 2)
        # Store the calibration coefficient and its error
        calibration_coefficients[channel] = {'c1': result.params['slope'].value,
                                             'c1_err': result.params['slope'].stderr,
                                             'c0': result.params['intercept'].value,
                                             'c0_err': result.params['intercept'].stderr
                                             }

    ax.set_xlabel(r'Norm. digital number (ADU sr$^{-1}$ s$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'Total flux ($10^{-7}$ W/m²)', fontsize=14)
    ax.legend(numpoints=1, loc='upper left')
    ax.tick_params(axis='both', direction='in', labelsize=14)
    fig.tight_layout()
    fname = work_dir / 'calibration_coefficient_fit_result.png'
    plt.savefig(fname, format='png')
    os.system(f'mogrify -trim {fname}')
    plt.close()
    # # Print out the calibration coefficients and their errors
    # for channel, values in calibration_coefficients.items():
    #     print(f"Channel {channel}: Coefficient = {values['coefficient']}, Error = {values['error']}")
    #     print(f"Channel {channel}: Inverse Coefficient = {values['inverse_coefficient']},"
    #           f" Inverse Error = {values['inverse_error']}")

    # Convert to DataFrame
    coeff_df = pd.DataFrame.from_dict(calibration_coefficients, orient='index')

    # # Optionally, rename columns for clarity
    # coeff_df.rename(columns={'slope': 'c1',
    #                          'slope_err': 'c1_err',
    #                          # 'intercept': 'c0',
    #                          # 'intercept_err': 'c0_err'
    #                          }, inplace=True)

    coeff_df = coeff_df.reset_index(names=['Band'])
    print(coeff_df)

    # save to disk
    coeff_df.to_csv(work_dir / f'calibration_coeff_RGB_mavic3.csv', index=False)

    plt.show()
    # spec_data_df.plot(kind='line', x='lambda', y='m_HR2326_')
    # plt.show()


def calculate_total_flux(name, spec_data, resp_data, resp_wl, T_m, T_a_infinity, theta_z, z_star, H_a, H_m):
    """"""

    T_star_m = calculate_total_transmittance(T_m, theta_z, z_star, H_m)
    T_star_a = calculate_total_transmittance(T_a_infinity, theta_z, z_star, H_a)

    flux = spec_data[name]
    flux_integrated = np.trapz(flux * (T_star_m * T_star_a) * resp_data,
                               x=resp_wl)
    wl_eff = np.trapz(resp_data, x=resp_wl)

    flux_avg = flux_integrated / wl_eff
    return flux_integrated


def load_and_prepare_spectra(file_path):
    """"""
    spec_names_file_name = file_path / '2024-02-08_ref_field_spectra_names.fit'
    spec_data_file_name = file_path / '2024-02-08_ref_field_spectra_data.fit'

    spec_names_df = read_fits_to_dataframe(str(spec_names_file_name))

    spec_names_df.rename(columns={"Vmag": "V"}, inplace=True)

    spec_names_df['Name'] = spec_names_df['Name'].replace(r"\s+", "", regex=True)
    spec_names_df['Sp'] = spec_names_df['Sp'].str.strip()

    spec_names_df.query("Name != 'HR2693'", inplace=True)
    spec_names_df.query("Name != 'HR2326'", inplace=True)
    spec_names_df.query("Name != 'HR3043'", inplace=True)
    spec_names_df.query("Name != 'HR1654'", inplace=True)
    spec_names_df.query("Name != 'HR2580'", inplace=True)
    spec_names_df.query("Name != 'HR2646'", inplace=True)
    spec_names_df.query("Name != 'HR1325'", inplace=True)
    spec_names_df.query("Name != 'HR1865'", inplace=True)

    t = Table.read(str(spec_data_file_name))

    spec_data_df = t.to_pandas()
    spec_data_df = spec_data_df.drop(['recno'], axis=1)

    # convert to flux
    spec_data_df = convert_mag_to_flux(spec_data_df, spec_names_df)

    # interpolate spectra, the step size is 1 because the wl is in nm
    spec_data_df = interpolate_data(spec_data_df, 1)

    spec_names_df.reset_index(inplace=True)
    return spec_names_df, spec_data_df


def interpolate_data(df, step_size=10, lambda_in=None):
    """
    Interpolates the data in the DataFrame based on the 'lambda' column.

    :param df: DataFrame containing the data to be interpolated.
    :param step_size: The step size for interpolation in Angstrom.
    :return: Interpolated DataFrame.
    """
    interpolated_dfs = []
    lambda_min, lambda_max = df['lambda'].min(), df['lambda'].max()

    new_lambda = np.arange(lambda_min, lambda_max + step_size, step_size)
    if lambda_in is not None:
        new_lambda = lambda_in

    interpolated_dfs.append(pd.DataFrame({'lambda': new_lambda}))

    # Interpolating for each column (star)
    for column in df.columns:
        if column != 'lambda':
            # Using linear interpolation
            interp_func = interp1d(df['lambda'], df[column], bounds_error=False, fill_value="NaN")
            interpolated_values = interp_func(new_lambda)
            interpolated_dfs.append(pd.DataFrame({column: interpolated_values}))

    # Concatenating all interpolated data
    interpolated_df = pd.concat(interpolated_dfs, axis=1)
    return interpolated_df


def convert_mag_to_flux(df, original_df):
    """
    Convert magnitudes to fluxes in a DataFrame using specific zero-point fluxes for each star.

    :param df: DataFrame with magnitudes.
    :param original_df: Original DataFrame containing zero-point fluxes in 10**-11 W/m²/nm.
    :return: DataFrame with converted fluxes.
    """
    flux_df = df.copy()

    flux_df['lambda'] = df['lambda'] * 10

    conversion_factor = 10 ** 7 / 10 ** 4 / 10  # Conversion factor

    for name in original_df['Name']:
        # Adjusting the name to match the column naming in the magnitude DataFrame
        adjusted_name = f'm_{name}_'
        if adjusted_name in df.columns:
            # Retrieve the zero-point flux from the original DataFrame (assuming it's in 10**-11 W/m²/nm)
            zero_point_flux = original_df.loc[original_df['Name'] == name, 'F_555nm_'].iloc[0] * 10 ** -11
            zero_point_flux = zero_point_flux * conversion_factor
            flux_df[adjusted_name] = zero_point_flux * 10 ** (-0.4 * df[adjusted_name])

    return flux_df


def load_response(file_path):
    """"""

    df = read_fits_to_dataframe(str(file_path))

    interpolated_df = interpolate_data(df, 10)

    return interpolated_df


def main_v1():
    """"""

    work_dir = Path('../../data/Camera_Calibration/')
    std_tbl_out_dir = work_dir / 'std_stars_data/'
    fits_dir = work_dir / '2024-02-08_starfield/iso800_t_5/ImageAnalysis/iso800_t_5/processed/pipelineout/'
    std_star_file_path = work_dir / '2024-02-08_ref_field_mavic3.fit'

    raw_std_table = load_ref_star_table(std_star_file_path)

    # Example usage
    base_filenames = sorted(get_base_filenames(fits_dir))
    print(base_filenames)

    channels = ['R', 'G', 'B']
    results = {channel: {} for channel in channels}  # Dictionary to store results

    for base_filename in base_filenames:
        info_fits = fits_dir / f'{base_filename}_G_out.fits'
        _, info_header = load_fits_image(info_fits)

        observation_time = info_header['DATE-OBS']  # Example time in ISO format
        latitude = info_header['SITELAT']
        longitude = info_header['SITELONG']
        std_df_cor = correct_std_flux(raw_std_table, observation_time, latitude, longitude, work_dir, base_filename)

        std_df = std_df_cor.query("2.1 < V <= 4.")

        # std_df = std_df_cor
        for channel in channels:
            file_path = fits_dir / f'{base_filename}_{channel}_out.fits'
            image_data, header = load_fits_image(file_path)
            wcs_header = WCS(info_header)

            pix_coords = convert_ra_dec_to_pixel(std_df[['_RAJ2000', '_DEJ2000']].values, wcs_header)
            mask = create_mask_for_phot_table(pix_coords, image_data.shape)

            std_df_filtered = std_df[mask]
            std_df_filtered = std_df_filtered.reset_index(drop=True)

            std_df_filtered['StarID'] = range(1, len(std_df_filtered) + 1)
            pix_coords_filtered = pix_coords[mask]

            aperture_radius = 7  # Example radius
            annulus_radii = (1.5 * aperture_radius, 2. * aperture_radius)  # Inner and outer radii for annulus

            phot_table = perform_aperture_photometry(image_data, pix_coords_filtered, aperture_radius, annulus_radii)
            # print(phot_table)
            # print(std_df_filtered)
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(10, 6))
            # vmin = 0
            # vmax = 1500
            # ax.imshow(image_data, vmin=vmin, interpolation='nearest')
            # ax.scatter(pix_coords_filtered[:,0], pix_coords_filtered[:,1])
            # plt.show()
            ref_band = channel if channel != 'G' else 'V'
            for star_id, measured_flux in zip(std_df_filtered['StarID'], phot_table['aper_sum_bkgsub']):
                # Retrieve reference flux for the star_id from std_df
                ref_flux = std_df_filtered.loc[std_df_filtered['StarID'] == star_id,
                f'{ref_band}_flux_corrected'].iloc[0]
                # if ref_band == 'B':
                #     ref_flux = ref_flux + std_df_filtered.loc[std_df_filtered['StarID'] == star_id,
                #                                               f'U_flux_corrected'].iloc[0]
                if star_id not in results[channel]:
                    results[channel][star_id] = []

                # Store both reference and measured fluxes
                results[channel][star_id].append([ref_flux, measured_flux])

    # print(results)
    # Calculating mean values
    mean_results = {channel: {} for channel in channels}
    for channel in channels:
        for star_id, fluxes in results[channel].items():
            fluxes = np.array(fluxes)
            # print(fluxes)
            mean_results[channel][star_id] = [np.median(fluxes, axis=0), np.std(fluxes, axis=0)]
    # print(mean_results)

    import matplotlib.pyplot as plt
    # star_ids = set(star_id for channel in channels for star_id in results[channel])  # Get all unique star IDs
    #
    # for star_id in star_ids:
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #     fig.suptitle(f'Star ID: {star_id}')
    #
    #     for i, channel in enumerate(channels):
    #         if star_id in results[channel]:
    #             ref_fluxes, measured_fluxes = zip(*results[channel][star_id])
    #             axs[i].scatter(ref_fluxes, measured_fluxes, alpha=0.7)
    #             axs[i].set_title(f'Channel {channel}')
    #             axs[i].set_xlabel('Reference Flux')
    #             axs[i].set_ylabel('Measured Flux')
    #             axs[i].grid(True)
    #
    #     plt.tight_layout()
    #     plt.show()

    # Prepare figure
    # plt.figure(figsize=(15, 5))
    #
    # # Iterate over each channel to create a subplot
    # for i, channel in enumerate(results.keys(), 1):
    #     ref_fluxes = []
    #     measured_fluxes = []
    #     # Create subplot for the channel
    #     plt.subplot(1, len(results), i)
    #
    #     # Extract fluxes for each star in the channel
    #     for star_id, flux_pairs in results[channel].items():
    #         for ref_flux, measured_flux in flux_pairs:
    #             ref_fluxes.append(ref_flux)
    #             measured_fluxes.append(measured_flux)
    #
    #     plt.scatter(ref_fluxes, measured_fluxes, alpha=0.7)
    #     # Extract fluxes for each star in the channel
    #     for star_id, flux_pairs in mean_results[channel].items():
    #         plt.scatter(flux_pairs[0][0], flux_pairs[0][1], color='k')
    #
    #     plt.title(f'{channel}-band Flux Comparison')
    #     plt.xlabel('Reference Flux')
    #     plt.ylabel('Measured Flux')
    #     plt.grid(True)
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    from scipy import stats
    from scipy.optimize import curve_fit
    import lmfit

    calibration_coefficients = {}
    for channel in channels:
        ref_fluxes = []
        measured_fluxes = []
        y_errs = []  # Assuming you have uncertainties for measured_fluxes

        for star_id in mean_results[channel]:
            values = mean_results[channel][star_id]
            ref_fluxes.append(values[0][0])
            measured_fluxes.append(values[0][1])
            y_errs.append(values[1][1])  # Append uncertainty

        x = np.array(ref_fluxes)
        y = np.array(measured_fluxes)

        params = lmfit.Parameters()
        params.add('slope', value=1e20, min=0)  # Initial value and a constraint to be positive

        result = lmfit.minimize(linear_model_lmfit, params, method='leastsq',
                                args=(x,), kws={'data': y, 'eps': y_errs})
        print(lmfit.fit_report(result))
        coeff = result.params['slope'].value
        coeff_error = result.params['slope'].stderr

        plt.figure()
        plt.plot(x, y, 'o', label='original data')
        plt.plot(x, linear_model_lmfit(result.params, x), 'k', label='fitted line 2')
        plt.legend()

        inverse_coeff = 1 / coeff
        inverse_error = coeff_error / (coeff ** 2)
        # Store the calibration coefficient and its error
        calibration_coefficients[channel] = {'coefficient': coeff, 'error': coeff_error,
                                             'inverse_coefficient': inverse_coeff, 'inverse_error': inverse_error}

    # # Print out the calibration coefficients and their errors
    # for channel, values in calibration_coefficients.items():
    #     print(f"Channel {channel}: Coefficient = {values['coefficient']}, Error = {values['error']}")
    #     print(f"Channel {channel}: Inverse Coefficient = {values['inverse_coefficient']},"
    #           f" Inverse Error = {values['inverse_error']}")

    # Convert to DataFrame
    coeff_df = pd.DataFrame.from_dict(calibration_coefficients, orient='index')

    # Optionally, rename columns for clarity
    coeff_df.rename(columns={'coefficient': 'c',
                             'error': 'c_err',
                             'inverse_coefficient': 'c_inv',
                             'inverse_error': 'c_inv_err'}, inplace=True)

    coeff_df = coeff_df.reset_index(names=['Band'])
    print(coeff_df)

    # save to disk
    coeff_df.to_csv(work_dir / f'calibration_coeff_RGB_mavic3.csv', index=False)

    plt.show()
    # for channel in channels:
    #     ref_fluxes = []
    #     measured_fluxes = []
    #
    #     for star_id in mean_results[channel]:
    #         for ref_flux, measured_flux in mean_results[channel][star_id]:
    #             # print(ref_flux, measured_flux)
    #             ref_fluxes.append(ref_flux)
    #             measured_fluxes.append(measured_flux)
    #
    #     # Perform linear regression
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(ref_fluxes, measured_fluxes)
    #     x = np.array(ref_fluxes)
    #     y = np.array(measured_fluxes)
    #     plt.figure()
    #     plt.plot(x, y, 'o', label='original data')
    #     plt.plot(x, intercept + slope * x, 'r', label='fitted line')
    #     plt.legend()
    #     plt.show()
    #     # Store the calibration coefficient (slope) and R-squared value
    #     calibration_coefficients[channel] = {'coefficient': slope, 'r_squared': r_value ** 2}
    #
    # # Print out the calibration coefficients and R-squared values
    # for channel, values in calibration_coefficients.items():
    #     print(f"Channel {channel}: Coefficient = {values['coefficient']}, R-squared = {values['r_squared']}")


def linear_model(x, a):
    return a * x


def linear_model_lmfit(params, x, data=None, eps=None):
    model = params['slope'] * x + params['intercept']
    if data is None:
        return model
    if eps is None:
        return model - data
    return (model - data) / eps


def load_ref_star_table(file_path):
    dataframe = read_fits_to_dataframe(str(file_path))

    df = dataframe[['_RAJ2000', '_DEJ2000', 'Name',
                    'SpType', 'Vmag', 'U-V', 'B-V', 'R-V', 'I-V',
                    '_RA_icrs', '_DE_icrs']]

    # Drop all rows where any element is NaN
    df_cleaned = df.dropna()

    df = df_cleaned.copy()

    # Apply function to each row
    df['U'] = df_cleaned.apply(lambda row: row['Vmag'] + row['U-V'], axis=1)
    df['B'] = df_cleaned.apply(lambda row: row['Vmag'] + row['B-V'], axis=1)
    df['R'] = df_cleaned.apply(lambda row: row['Vmag'] + row['R-V'], axis=1)
    df['I'] = df_cleaned.apply(lambda row: row['Vmag'] + row['I-V'], axis=1)

    df.rename(columns={"Vmag": "V"}, inplace=True)

    df['Name'] = df['Name'].str.strip()
    df['SpType'] = df['SpType'].str.strip()
    # names_to_exclude = ['HD  58350', 'HD  14228', 'HD  15371',
    #                     'HD  17206', 'HD  18322', 'HD  18978',
    #                     'HD  22049', 'HD  27442', 'HD  45348',
    #                     'HD  61935', 'HD  64760']  # List of names to exclude
    # std_df = df.query("Name not in @names_to_exclude")
    std_df = df.query("Name != 'HD  53244'")

    return df


def correct_std_flux(df, observation_time, latitude, longitude, work_dir, file_base_name):
    # Apply the function to each row in the DataFrame
    df['theta_z'] = df.apply(calculate_zenith_angle_for_row,
                             args=(observation_time, latitude, longitude),
                             axis=1)

    # Calculate flux for each filter
    for band in ['U', 'B', 'V', 'R', 'I']:
        df[f'{band}_flux_total'] = df[f'{band}'].apply(calculate_flux,
                                                       args=(flux_zero_values[band], lambda_c_vals[band],
                                                             dlambda_lambda_vals[band]))

        # Apply atmospheric correction
        df[f'{band}_flux_corrected'] = df.apply(
            lambda row: correct_flux(row[f'{band}_flux_total'], p, tau_a_05, alpha,
                                     lambda_c_vals[band], row['theta_z'], z_star, z_a, H_m, H_a),
            axis=1
        )

    # save to disk
    df.to_csv(work_dir / f'std_star_flux_corrected_{file_base_name}.csv', index=False)

    return df


def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header
    return image_data, header


def convert_ra_dec_to_pixel(sky_coords, wcs_header):
    # sky_coord = SkyCoord(ra, dec, unit="deg", frame='fk5')
    # print(sky_coord)
    return wcs_header.all_world2pix(sky_coords, 0, maxiter=20,
                                    tolerance=1e-4, ra_dec_order=True, quiet=True, adaptive=True,
                                    detect_divergence=True)


def create_mask_for_phot_table(pixel_positions, image_shape):
    max_y, max_x = image_shape
    valid_x = (pixel_positions[:, 0] >= 0) & (pixel_positions[:, 0] < max_x)
    valid_y = (pixel_positions[:, 1] >= 0) & (pixel_positions[:, 1] < max_y)
    return valid_x & valid_y


def perform_aperture_photometry(image_data, pix_coords, aperture_radius, annulus_radii=None):
    apertures = CircularAperture(pix_coords, r=aperture_radius)
    phot_table = aperture_photometry(image_data, apertures)

    if annulus_radii:
        annulus_apertures = CircularAnnulus(pix_coords, r_in=annulus_radii[0], r_out=annulus_radii[1])
        annulus_masks = annulus_apertures.to_mask(method='center')
        bkg_median = []
        bkg_mean = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image_data)
            annulus_data_1d = annulus_data[mask.data > 0]
            mean_sigclip, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            bkg_mean.append(mean_sigclip)

        bkg_median = np.array(bkg_median)
        bkg_mean = np.array(bkg_mean)
        phot_table['annulus_median'] = bkg_median
        phot_table['annulus_mean'] = bkg_mean
        phot_table['aper_bkg'] = bkg_mean * apertures.area
        phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']
        # import matplotlib.pyplot as plt
        # from astropy.visualization import simple_norm
        #
        # plt.figure()
        # norm = simple_norm(image_data, 'sqrt', percent=99)
        # plt.imshow(image_data, norm=norm, interpolation='nearest')
        #
        #
        # ap_patches = apertures.plot(color='white', lw=2,
        #                             label='Photometry aperture')
        # ann_patches = annulus_apertures.plot(color='red', lw=2,
        #                                     label='Background annulus')
        # plt.show()
    return phot_table


def get_base_filenames(directory_path):
    fits_files = Path(directory_path).glob('*_out.fits')
    base_filenames = set()
    for file_path in fits_files:
        filename = file_path.stem  # Get the filename without extension
        base_filename = filename.rsplit('_', 2)[0]  # Split and get the base part
        base_filenames.add(base_filename)
    return list(base_filenames)


def read_fits_to_dataframe(file_path):
    """
    Read a FITS binary table and convert it into a Pandas DataFrame.

    :param file_path: Path to the FITS file.
    :type file_path: str
    :return: Pandas DataFrame containing the data from the FITS table.
    :rtype: DataFrame
    """
    # Open the FITS file
    with fits.open(file_path) as hdulist:
        # Assuming the binary table is in the first extension
        binary_table = hdulist[1].data

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(binary_table)

    return df


def calculate_flux(magnitude, flux_zero, lam_c, dlam_lam):
    return flux_zero * 10 ** (-0.4 * magnitude) * 1e-26 * const.c.value / lam_c * dlam_lam


def calculate_molecular_transmittance(p, wavelength):
    return np.exp(-p / 101.3 * (0.0021520 * (1.0455996 - 341.29061 / wavelength ** 2 - 0.90230850 * wavelength ** 2)
                                / (1 + 0.0027059889 / wavelength ** 2 - 85.968563 * wavelength ** 2)))


def calculate_aerosol_transmittance(tau_a_05, alpha, wavelength):
    return np.exp(-tau_a_05 * (wavelength / 0.5) ** (-alpha))


def calculate_aerosol_transmittance_infinity(T_a, z_a, H):
    return T_a ** np.exp(z_a / H)


def calculate_total_transmittance(t_infinity, theta_z, z_star, H):
    return np.exp(np.log(t_infinity) / np.cos(np.radians(theta_z)) * np.exp(-z_star / H))


def correct_flux(flux, p, tau_a_05, alpha, wavelength, theta_z, z_star, z_a, H_m, H_a):
    T_m = calculate_molecular_transmittance(p, wavelength)
    T_star_m = calculate_total_transmittance(T_m, theta_z, z_star, H_m)

    T_a = calculate_aerosol_transmittance(tau_a_05, alpha, wavelength)
    T_a_infinity = calculate_aerosol_transmittance_infinity(T_a, z_a, H_a)
    T_star_a = calculate_total_transmittance(T_a_infinity, theta_z, z_star, H_a)

    corrected_flux = flux / (T_star_m * T_star_a)

    return corrected_flux


def calculate_zenith_angle(ra, dec, observation_time, latitude, longitude):
    """
    Calculate the zenith angle of a star given its RA, DEC, and the observation location and time.

    :param ra: Right Ascension of the star.
    :param dec: Declination of the star.
    :param observation_time: Time of observation (ISO format string or datetime object).
    :param latitude: Latitude of the observation location in degrees.
    :param longitude: Longitude of the observation location in degrees.
    :return: Zenith angle in radians.
    """
    # Convert RA and DEC to SkyCoord object
    star_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='fk5')

    # Observation location and time
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=95)
    time = Time(observation_time)

    # Convert to AltAz frame
    altaz = star_coord.transform_to(AltAz(obstime=time, location=location))

    # Zenith angle calculation
    zenith_angle = 90 * u.deg - altaz.alt  # Zenith angle is 90 degrees minus the altitude
    return zenith_angle  # .to(u.rad)  # Convert to radians


def calculate_zenith_angle_for_row(row, obs_time, lat, lon):
    """
    Calculate the zenith angle for a row in the DataFrame.

    :param row: DataFrame row containing RA and DEC for a star.
    :param obs_time: Fixed observation time.
    :param lat: Fixed latitude of observation.
    :param lon: Fixed longitude of observation.
    :return: Zenith angle in radians.
    """
    return calculate_zenith_angle(row['_RAJ2000'], row['_DEJ2000'], obs_time, lat, lon).value


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
