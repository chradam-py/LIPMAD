#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import copy
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

from astropy import constants as const

from pathlib import Path

# Define the data
ref_data = {
    'Band': ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'g', 'r', 'i', 'z'],
    'lambda_c': [0.36, 0.44, 0.55, 0.64, 0.79, 1.26, 1.60, 2.22, 0.52, 0.67, 0.79, 0.91],
    'dlambda/lambda': [0.15, 0.22, 0.16, 0.23, 0.19, 0.16, 0.23, 0.23, 0.14, 0.14, 0.16, 0.13],
    'F_0': [1810, 4260, 3640, 3080, 2550, 1600, 1080, 670, 3730, 4490, 4760, 4810],
}

# Create the DataFrame
flux_df = pd.DataFrame(ref_data)


def main():
    """"""
    work_dir = Path('../../data/Camera_Calibration/')
    file_path = work_dir / '2024-02-08_ref_field_mavic3.fit'

    p = 101  # Atmospheric pressure in kPa
    tau_a_05 = 0.089  # Vertical aerosol optical depth at 0.5 μm
    alpha = 1.378  # Ångstrom coefficient
    wavelength = 0.55  # Wavelength in μm
    z_a = 0.965  # km
    z_star = 0.095  # km
    H_m = 8  # Scale height for molecules in km
    H_a = 2  # Scale height for aerosols in km

    observation_time = "2024-02-08T22:31:00.000"  # Example time in ISO format
    latitude = -23.765894444444445  # Example latitude in degrees
    longitude = -70.43574722222223  # Example longitude in degrees

    dataframe = read_fits_to_dataframe(str(file_path))

    df = dataframe[['_RAJ2000', '_DEJ2000', 'Name',
                    'SpType', 'Vmag', 'U-V', 'B-V', 'R-V', 'I-V',
                    '_RA_icrs', '_DE_icrs']]
    # Drop all rows where any element is NaN
    df_cleaned = df.dropna()
    # Use query to filter rows
    df_filtered = df_cleaned.query('2 <= Vmag <= 4')

    df = df_filtered.copy()

    # Apply function to each row
    df['U'] = df_filtered.apply(lambda row: row['Vmag'] + row['U-V'], axis=1)
    df['B'] = df_filtered.apply(lambda row: row['Vmag'] + row['B-V'], axis=1)
    df['R'] = df_filtered.apply(lambda row: row['Vmag'] + row['R-V'], axis=1)
    df['I'] = df_filtered.apply(lambda row: row['Vmag'] + row['I-V'], axis=1)

    df.rename(columns={"Vmag": "V"}, inplace=True)

    # Apply the function to each row in the DataFrame
    df['theta_z'] = df.apply(calculate_zenith_angle_for_row,
                             args=(observation_time, latitude, longitude),
                             axis=1)

    # Flux at 0 magnitude for each filter
    flux_zero_values = flux_df.set_index('Band')['F_0']
    lambda_c_vals = flux_df.set_index('Band')['lambda_c']
    dlambda_lambda_vals = flux_df.set_index('Band')['dlambda/lambda']

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
    df.to_csv(work_dir / 'values.csv', index=False)


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
    return np.exp(np.log(t_infinity) / np.cos(theta_z) * np.exp(-z_star / H))


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
