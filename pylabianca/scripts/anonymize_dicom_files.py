# -*- coding: utf-8 -*-
"""
Anonymize DICOM files
---------------------
Simple script to anonymize DICOM images for multiple patients.
The folders in `input_path` should be folders named as patients (for example
U07, U08, etc.) and each contains `ct` and/or `mri` subfolders with DICOM
images to anonymize.

Created on Fri Nov 11 01:37:51 2022

@author: mmagn
"""
import os
import dicomanonymizer as diano


input_path = r'G:\My Drive\syracuse_anat'
output_path = r'G:\My Drive\syracuse_anat_anon'


subdirs = os.listdir(input_path)
for subdir in subdirs:
    full_subdir = os.path.join(input_path, subdir)
    sub_subdirs = os.listdir(full_subdir)

    has_ct = 'ct' in sub_subdirs
    has_mri = 'mri' in sub_subdirs

    if has_ct or has_mri:
        # make sure subject folder is present in output path
        output_subdir = os.path.join(output_path, subdir)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        print(f'Processing patient {subdir}...')

    if has_ct:
        # anonymize CT scans
        print('Anonymizing CT scans')
        dicom_path_in = os.path.join(full_subdir, 'ct')
        dicom_path_out = os.path.join(output_subdir, 'ct')

        if not os.path.exists(dicom_path_out):
            os.mkdir(dicom_path_out)

        diano.anonymize(dicom_path_in, dicom_path_out, {}, False)

    if has_mri:
        # anonymize mri scans
        print('Anonymizing MRI scans')
        dicom_path_in = os.path.join(full_subdir, 'mri')
        dicom_path_out = os.path.join(output_subdir, 'mri')

        if not os.path.exists(dicom_path_out):
            os.mkdir(dicom_path_out)

        diano.anonymize(dicom_path_in, dicom_path_out, {}, False)
