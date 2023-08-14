"""
Reads all the data and stores paired data into h5 file
"""

import os

from lib.data.dicom import (
    filter_paired,
    group_cts_patient,
    group_lidc_patients,
    group_xrs_patient,
    merge_patient_exams,
)
from lib.io.dicom import load_slices
from lib.utils.dicom import get_folder_plane, get_folder_window, is_lung_axial_ct
from lib.utils.utils import remove_ricord_prefix
from lib.utils.viz import show_ct

ct_dirs = group_cts_patient("data/MIDRC-RICORD-1A")

for key in ct_dirs:
    # print()
    # print()
    for ct_dir in ct_dirs[key]:
        # print(get_folder_plane(ct_dir))
        if get_folder_plane(ct_dir, True) == None:
            # print(key)
            try:
                print(get_folder_window(ct_dir))
                # volume, _, dcm = load_slices(ct_dir)
                # show_ct(volume,False)
            except:
                pass

        """if get_folder_window(ct_dir) == "standard" and get_folder_plane(ct_dir) == "ax":
            try:
                print(get_folder_window(ct_dir))
                volume, _, dcm = load_slices(ct_dir)
                show_ct(volume)
            except:
                pass"""

xr_dirs = group_xrs_patient("data/MIDRC-RICORD-1C")

ricord = merge_patient_exams(ct_dirs, xr_dirs)

ricord, ricord_paired = filter_paired(ricord)

lidc = group_lidc_patients("data/LIDC-IDRI")

lidc, lidc_paired = filter_paired(lidc)

# print(f'Ricord: {len(ricord)}')
print(f"Ricord: {len(ricord)} | Paired: {len(ricord_paired)}")
print(f"LIDC: {len(lidc)} | Paired: {len(lidc_paired)}")

temp = []
for patient in os.listdir("data/MIDRC-RICORD-1A"):
    if patient == "LICENSE":
        continue
    if f"{remove_ricord_prefix(patient)}" in temp:
        continue
    else:
        temp.append(remove_ricord_prefix(patient))

for patient in os.listdir("data/MIDRC-RICORD-1C"):
    if patient == "LICENSE":
        continue
    if f"{remove_ricord_prefix(patient)}" in temp:
        continue
    else:
        temp.append(remove_ricord_prefix(patient))


print(len(temp))
