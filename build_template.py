############################################
# Build a Calgary/Campinas T1 template     #
#                                          #
# - Using 'Syn' for registration           #
# - Normalized intensities between 0 and 1 #
#                                          #
# Sergi Valverde 2019                      #
############################################


import os
import numpy as np
import ants
from processing import normalize_data


# -------------------------------------------------------
# data options
IMAGE_PATH = '/home/sergivalverde/DATA/campinas/all'
T1_NAME = 'T1.nii.gz'
BRAINMASK_NAME = 'brainmask_ss.nii.gz'

# registration options
REG_METHOD = 'Affine'
INTERP = 'linear'

# output names
T1_OUT_TEMPLATE_PATH = 'CC351_lin_t1.nii.gz'
BRAINMASK_OUT_TEMPLATE_PATH = 'CC351_lin_brainmask.nii.gz'
# -------------------------------------------------------

scans = sorted(os.listdir(IMAGE_PATH))
FIXED_SCAN = scans[0]
fixed = ants.image_read(os.path.join(IMAGE_PATH, FIXED_SCAN, 'T1.nii.gz'))

# store accumulated images
t1_template = np.zeros_like(fixed.numpy())
skull_template = np.zeros_like(fixed.numpy())

for MOVING, index in zip(scans[1:], range(1, len(scans))):

    print('Registering scan', MOVING, '--->', FIXED_SCAN)

    t1_moving = ants.image_read(os.path.join(IMAGE_PATH, MOVING, T1_NAME))
    mask_moving = ants.image_read(os.path.join(IMAGE_PATH,
                                               MOVING,
                                               BRAINMASK_NAME))

    mytx = ants.registration(fixed, t1_moving, type_of_transform=REG_METHOD)
    registered_image = mytx['warpedmovout']

    # store the T1-w against the template (optional)
    registered_image.to_filename(os.path.join(IMAGE_PATH, MOVING,
                                              'T1_to_' +
                                              FIXED_SCAN +
                                              '_' + REG_METHOD +
                                              '.nii.gz'))
    t1_template += registered_image.numpy()

    # apply transformation to the brainmask
    moving_mask = ants.apply_transforms(fixed,
                                        mask_moving,
                                        mytx['fwdtransforms'],
                                        interpolator=INTERP)

    moving_mask.to_filename(os.path.join(IMAGE_PATH, MOVING,
                                         'brainmask_to_' +
                                         FIXED_SCAN +
                                         '_' + REG_METHOD +
                                         '.nii.gz'))

    skull_template += moving_mask.numpy()

# build the template
t1_template /= index
skull_template /= index

# normalize the output: here, two solutions can be used.
# 1) each registered image is normalized before accumulation (more blurred)
# 2) only the resulting template is accumulated (preferred, more contrast)
t1_template = normalize_data(t1_template, 'zero_one')

# post process the template: remove non-zero parts outside the head
min_int_brain = t1_template[skull_template > 0].min()
t1_template[t1_template < min_int_brain] = 0

# store the results to disk
template_image = fixed.new_image_like(t1_template)
template_image.to_filename(T1_OUT_TEMPLATE_PATH)

template_image = fixed.new_image_like(skull_template)
template_image.to_filename(BRAINMASK_OUT_TEMPLATE_PATH)
