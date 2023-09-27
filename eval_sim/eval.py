import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
import copy
import pandas as pd
import warnings
from scipy.stats import expon


model = 'stage_2'

# Parameters:
CR_SIGMA_RANGE = [1, 30]
GAUSS_AMPLITUDE_RANGE = [2, 20000]
IMAGE_SIZE = 180
NOISE_SD_RANGE = [0, 30]
BACKGROUND_LUM_EXP_LOC = 1.
BACKGROUND_LUM_EXP_SCALE = 10.
BG_LUMINANCE_RANGE = [.125, .6]


def run(gauss_amp, bg_pos, sx):
    from tensorflow import keras
    import deeptrack as dt
    from deeptrack.extras.radialcenter import radialcenter

    def drawFromRange(range, n=1):
        # range - list of length 2, e.g., [1, 2]
        return range[0] + np.random.rand(n) * np.diff(range)

    # inverse unnormalized Gaussian for scaling
    def gauss1d_inv(v=0,mx=0,sx=1):
        return mx + np.sqrt(2)*sx*np.sqrt(-np.log(v))

    # define 2D Gaussian (without normalization term so peak is 1)
    def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    class GrayBackground(dt.Feature):
        __list_merge_strategy__ = dt.MERGE_STRATEGY_APPEND
        __distributed__ = False
        def get(self, image, pos, ori, lum, bg_lum, smooth_edge, **kwargs):
            x = np.arange(0, IMAGE_SIZE)+((IMAGE_SIZE+1)%2)/2
            y = np.arange(0, IMAGE_SIZE)+((IMAGE_SIZE+1)%2)/2
            x, y = np.meshgrid(x, y)

            x -= pos[0]
            y -= pos[1]

            rx = np.cos(ori) * x + np.sin(ori) * y;
            rx = np.maximum(np.minimum(rx,smooth_edge/2),-smooth_edge/2)/smooth_edge+.5

            # apply raised cosine, from background to foreground luminance
            return (np.cos(rx*np.pi)/2+.5)*(lum-bg_lum)+bg_lum

    class SaturatedGaussian(dt.Feature):
        def get(self, image, position, sigma, gauss_amp, **kwargs):
            x = np.arange(0, image.shape[1])
            y = np.arange(0, image.shape[0])
            x, y = np.meshgrid(x, y)
            fac = gauss1d_inv(1/gauss_amp)
            sd = sigma/fac

            # generate
            feature = gauss_amp*gauss2d(x,y,mx=position[0], my=position[1], sx=sd, sy=sd)*255
            return np.maximum(image,feature)

    class Discretize(dt.Feature):
        # discretize like a real, and convert back to float as i'm not sure this library likes uint8 and don't want to try
        def get(self, image, dtype, **kwargs):
            image = image.astype(dtype)
            image = image.astype(np.float64)
            return image


    background = GrayBackground(
        background_pos=lambda: None,
        background_lum=lambda: None,
        pos=lambda background_pos: background_pos,
        ori=0,
        lum=lambda background_lum: background_lum,
        bg_lum=lambda: expon.rvs(loc=BACKGROUND_LUM_EXP_LOC,scale=BACKGROUND_LUM_EXP_SCALE),
        smooth_edge = 1
    )

    CR = SaturatedGaussian(
        cr_sigma=lambda: None,
        cr_position=lambda: None,
        sigma=lambda cr_sigma: cr_sigma,
        position=lambda cr_position: cr_position,
        gauss_amp=lambda: None
    )

    discretizer = Discretize(dtype=np.uint8)

    # the pipeline we actually want
    image_pipeline = background >> CR
    # Add noise to entire image
    image_pipeline >>= dt.Gaussian(sigma=lambda: drawFromRange(NOISE_SD_RANGE))
    image_pipeline >>= dt.math.Clip(min=0., max=255.) >> discretizer

    image_pipeline >>= dt.NormalizeMinMax(0,1)

    def get_properties(image):
        props = {
        'position': image.get_property("position"),
        'gray_extent': image.get_property("extent"),
        'gray_direction': image.get_property("direction"),
        'cr_sigma': image.get_property("cr_sigma"),
        'gauss_amp': image.get_property("gauss_amp"),
        'noise_sigma': image.get_property("noise_sigma"),
        'bg_luminance': image.get_property("lum")
        }
        return props


    test_model = keras.models.load_model(f'..\\trained_model\\{model}.h5', compile=False)

    def process_image(image, props, plot=False):
        image = np.squeeze(dt.image.strip(image))

        # ground truth
        label = props['position']
        real_position_x = label[0]
        real_position_y = label[1]

        # deep track's position
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore warnings from model.predict and radialcenter
            if plot:
                measured_position = test_model.predict(np.expand_dims(image, axis=0), verbose=0)
                measured_position_x = (measured_position[0, 0] * IMAGE_SIZE)
                measured_position_y = (measured_position[0, 1] * IMAGE_SIZE)
            else:
                measured_position_x,measured_position_y = np.nan,np.nan

            # center of radial symmetry
            radial_x, radial_y = radialcenter(image)

        # simple center of mass calculation
        cog_y, cog_x = ndimage.center_of_mass(image)

        # threshold and center of mass of resulting binary blob
        THRESH = 240./255.  # arbitrary but high threshold, as you would use in an actual eye tracker
        bin_image = copy.deepcopy(image)
        bin_image[bin_image >= THRESH] = THRESH
        bin_image[bin_image < THRESH] = 0
        thresh_y, thresh_x = ndimage.center_of_mass(bin_image)

        if plot:
            plt.imshow(image, cmap='gray', vmin=0, vmax=1)
            plt.scatter(real_position_x, real_position_y, s=70, c='r', marker='x')
            plt.scatter(measured_position_x, measured_position_y, s=100, marker='o', facecolor='none', edgecolors='b')
            plt.scatter(radial_x, radial_y, s=100, marker='>', facecolor='none', edgecolors='g')
            plt.scatter(cog_x, cog_y, s=100, marker='<', facecolor='none', edgecolors='y')
            plt.scatter(thresh_x, thresh_y, s=100, marker='*', facecolor='none', edgecolors='m')
            plt.show()

        return ([real_position_x, real_position_y],
                [measured_position_x, measured_position_y],
                [radial_x, radial_y],
                [cog_x, cog_y],
                [thresh_x, thresh_y])


    # do eval
    # sim parameters
    noise_levels = np.arange(0, 20, 2)  # Image noise
    gray_levels  = [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]

    # vertical position
    offset_y = 0.0
    my = IMAGE_SIZE/2 + offset_y

    step = 0.01  # step size in pixels
    step_range = 0.5  # *2 -> 1 pixel
    mx_fixed = IMAGE_SIZE/2
    steps = np.arange(mx_fixed - step_range,
                    mx_fixed + step_range,
                    step)

    pos = []
    for noise_level in noise_levels:
        for grl in gray_levels:
            print(f'bg_pos: {bg_pos}, gauss_amp: {gauss_amp}, plateau radius: {sx}, noise level: {noise_level}, gray level: {grl}')

            image_getter = lambda mx: image_pipeline(cr_position=np.array([mx, my]),
                                                     cr_sigma=np.array([sx]),
                                                     gauss_amp=np.array([gauss_amp]),
                                                     background_pos=np.array([mx+(bg_pos-.5)*2*sx, my]),
                                                     background_lum=np.array([grl*255]),
                                                     noise_sigma=np.array([noise_level]))
            images = [image_getter(mx) for mx in steps]
            props = [get_properties(im) for im in images]

            measured_positions = test_model.predict(np.array(images), verbose=0)*IMAGE_SIZE
            for k, mx in enumerate(steps):
                # process
                out = process_image(images[k], props[k], plot=False)

                # store
                ref = out[0]
                for m,method in zip(out[1:],['CNN','radial_symm','cog','thresh']):
                    if method=='CNN':
                        m = measured_positions[k,:]
                    pos.append([method, bg_pos, gauss_amp, sx, noise_level, grl, ref[0], ref[1], m[0], m[1]])

    # store results
    df = pd.DataFrame(pos, columns = ['method','bg_pos','gauss_amp', 'size', 'noise_level', 'gray_level',
                                      'ref_x', 'ref_y', 'est_x', 'est_y'])

    df['err_x'] = np.abs(df['est_x']-df['ref_x'])
    df['err_y'] = np.abs(df['est_y']-df['ref_y'])

    df.to_csv(f'CNN_data_{offset_y}_gaussamp{gauss_amp}_bgpos{bg_pos}_sx{sx}.csv',index=False)

    return gauss_amp,bg_pos,sx


if __name__ == "__main__":
    import pebble
    gauss_amps   = [10, 50, 200, 1000, 10000]
    bg_pos       = [-10000, -.25, 0, .25, .5, .75, 1., 1.25]    # -10000 for pure black background
    sigmas_x     = np.arange(2, 20, 2)                          # Blob size (Standard deviation of gaussian). Gaussian is scaled such that SD is radius of saturated plateau

    with pebble.ProcessPool(max_workers=2, max_tasks=1) as pool:
        for result in pool.map(run, *zip(*((a,b,c) for a in gauss_amps for b in bg_pos for c in sigmas_x))).result():
            print(f'done with {result[0]}, {result[1]}, {result[2]}')
