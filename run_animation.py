# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.

import sys
import os
import distutils.util
import re
import numpy as np
from PIL import Image

import sirf.STIR as pet


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r'(\d+)', string)]


# https://stackoverflow.com/questions/36000843/scale-np-array-to-certain-range
def rescale_linear_max_min(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    output = m * array + b

    return output


# https://stackoverflow.com/questions/36000843/scale-np-array-to-certain-range
def rescale_linear_max_min_with_known_max_min(array, new_min, new_max, input_min, input_max):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (input_max - input_min)
    b = new_min - m * input_min
    output = m * array + b

    return output


# https://stackoverflow.com/questions/36000843/scale-np-array-to-certain-range
def rescale_linear_sum_min(array, new_min, new_sum):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.nansum(array)
    m = (new_sum - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    output = m * array + b

    return output


# https://stackoverflow.com/questions/36000843/scale-np-array-to-certain-range
def rescale_linear_sum_min_with_known_sum_min(array, new_min, new_sum, input_min, input_sum):
    """Rescale an arrary linearly."""
    m = (new_sum - new_min) / (input_sum - input_min)
    b = new_min - m * input_min
    output = m * array + b

    return output


def rescale_array_path(data_array, data_type):
    new_data_array = data_array.copy()

    for i in range(len(new_data_array)):
        current_data_array = new_data_array[i]

        current_data_array = rescale_linear_max_min(current_data_array, 0.0, 1.0)

        new_data_array[i] = current_data_array

    return new_data_array


def rescale_array_array_path(data_array):
    new_data_array = data_array.copy()

    current_data_array = new_data_array[0]

    current_max = np.max(current_data_array)
    current_min = np.min(current_data_array)

    for i in range(1, len(new_data_array)):
        current_data_array = new_data_array[i]

        new_max = np.max(current_data_array)
        new_min = np.min(current_data_array)

        if new_max > current_max:
            current_max = new_max

        if new_min < current_min:
            current_min = new_min

    for i in range(len(new_data_array)):
        new_data_array[i] = rescale_linear_max_min_with_known_max_min(new_data_array[i], 0.0, 255.0, current_min, current_max)

    return new_data_array


# https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py#L510-L566
_errstr = "Mode is unknown or incompatible with input array shape."


# https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py#L510-L566
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from run_animation import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


# https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py#L510-L566
def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a np array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The np array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    strdata = None
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


# https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py#L510-L566
def fromimage(im, flatten=False, mode=None):
    """
    Return a copy of a PIL image as a numpy array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
        `imread` docstring for more details.
    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")

    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')

    a = np.array(im)
    return a


# https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py#L510-L566
def imresize(arr, size, interp='bilinear', mode=None):
    """
    Resize an image.
    This function is only available if Python Imaging Library (PIL) is installed.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Parameters
    ----------
    arr : ndarray
        The array of image to be resized.
    size : int, float or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image (height, width).
    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
        'bicubic' or 'cubic').
    mode : str, optional
        The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.
        If ``mode=None`` (the default), 2-D images will be treated like
        ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,
        `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.
    Returns
    -------
    imresize : ndarray
        The resized array of image.
    See Also
    --------
    toimage : Implicitly used to convert `arr` according to `mode`.
    scipy.ndimage.zoom : More generic implementation that does not use PIL.
    """
    im = toimage(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return fromimage(imnew)


def main(input_path=None, ground_truth_path=None, output_path=None, output_name=None, input_prefix_bool=None,
         input_prefix=None, input_file_type=None, ground_truth_prefix_bool=None, ground_truth_prefix=None,
         ground_truth_file_type=None, pixel_width=None, slice_width=None, slice_axis=None, slice_position=None,
         rescale_array_bool=None, frame_time=None):
    np.seterr(all="print")
    
    if input_path == None or ground_truth_path == None or output_path == None or output_name == None or \
        input_prefix_bool == None or input_prefix == None or input_file_type == None or \
        ground_truth_prefix_bool == None or ground_truth_prefix == None or ground_truth_file_type == None or \
        pixel_width == None or slice_width == None or slice_axis == None or slice_position == None or \
        rescale_array_bool == None or frame_time == None:
        input_path = sys.argv[1]
        ground_truth_path = sys.argv[2]
        output_path = sys.argv[3]
        output_name = sys.argv[4]
        input_prefix_bool = bool(distutils.util.strtobool(sys.argv[5]))
        input_prefix = sys.argv[6]
        input_file_type = sys.argv[7]
        ground_truth_prefix_bool = bool(distutils.util.strtobool(sys.argv[8]))
        ground_truth_prefix = sys.argv[9]
        ground_truth_file_type = sys.argv[10]
        pixel_width = float(sys.argv[11])
        slice_width = float(sys.argv[12])
        slice_axis = int(sys.argv[13])
        slice_position = int(sys.argv[14])
        rescale_array_bool = bool(distutils.util.strtobool(sys.argv[15]))
        frame_time = float(sys.argv[16])
    else:
        input_prefix_bool = bool(distutils.util.strtobool(input_prefix_bool))
        ground_truth_prefix_bool = bool(distutils.util.strtobool(ground_truth_prefix_bool))
        pixel_width = float(pixel_width)
        slice_width = float(slice_width)
        slice_axis = int(slice_axis)
        slice_position = int(slice_position)
        rescale_array_bool = bool(distutils.util.strtobool(rescale_array_bool))
        frame_time = float(frame_time)

    all_input_paths = os.listdir(input_path)
    input_paths = []

    # get input data
    for i in range(len(all_input_paths)):
        current_input_path = all_input_paths[i].rstrip()

        if len(current_input_path.split(input_file_type)) > 1:
            if input_prefix_bool:
                if len(current_input_path.split(input_prefix)) > 1:
                    input_paths.append("{0}/{1}".format(input_path, current_input_path))
            else:
                input_paths.append("{0}/{1}".format(input_path, current_input_path))

    input_paths.sort(key=human_sorting)

    ground_truth_paths = None

    # get ground truth path if exists
    if ground_truth_path != "None":
        all_ground_truth_paths = os.listdir(ground_truth_path)
        ground_truth_paths = []

        for i in range(len(all_ground_truth_paths)):
            current_ground_truth_path = all_ground_truth_paths[i].rstrip()

            if len(current_ground_truth_path.split(ground_truth_file_type)) > 1:
                if ground_truth_prefix_bool:
                    if len(current_ground_truth_path.split(ground_truth_prefix)) > 1:
                        ground_truth_paths.append("{0}/{1}".format(ground_truth_path, current_ground_truth_path))
                else:
                    ground_truth_paths.append("{0}/{1}".format(ground_truth_path, current_ground_truth_path))

        ground_truth_paths.sort(key=human_sorting)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frames = []

    if rescale_array_bool:
        # rescale entire array
        input_slice_list = []
        ground_truth_slice_list = []

        for i in range(len(input_paths)):
            input_slice = None

            if slice_axis == 0:
                input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[slice_position, :, :], 0)
                input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * pixel_width)), int(round(input_slice.shape[1] * pixel_width))))
            else:
                if slice_axis == 1:
                    input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[:, slice_position, :], 0)
                    input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * slice_width)), int(round(input_slice.shape[1] * pixel_width))))
                else:
                    if slice_axis == 2:
                        input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[:, :, slice_position], 0)
                        input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * slice_width)), int(round(input_slice.shape[1] * pixel_width))))

            input_slice_list.append(input_slice)

            if ground_truth_paths is not None:
                ground_truth_slice = None

                if slice_axis == 0:
                    ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[slice_position, :, :], 0)
                    ground_truth_slice = imresize(ground_truth_slice, (int(round(ground_truth_slice.shape[0] * pixel_width)), int(round(ground_truth_slice.shape[1] * pixel_width))))
                else:
                    if slice_axis == 1:
                        ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[:, slice_position, :], 0)
                        ground_truth_slice = imresize(ground_truth_slice, (int(round(ground_truth_slice.shape[0] * slice_width)), int(round(ground_truth_slice.shape[1] * pixel_width))))
                    else:
                        if slice_axis == 2:
                            ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[:, :, slice_position], 0)
                            ground_truth_slice = imresize(ground_truth_slice, (int(round(ground_truth_slice.shape[0] * slice_width)), int(round(ground_truth_slice.shape[1] * pixel_width))))

                ground_truth_slice_list.append(ground_truth_slice)

        input_slice_array = np.asfarray(input_slice_list)
        input_slice_array = rescale_array_array_path(input_slice_array)

        if ground_truth_paths is not None:
            ground_truth_slice_array = np.asfarray(ground_truth_slice_list)
            ground_truth_slice_array = rescale_array_array_path(ground_truth_slice_array)

            difference_slice_list = []
            red_difference_slice_list = []
            blue_difference_slice_list = []

            for i in range(len(input_slice_array)):
                difference_slice = ground_truth_slice_array[i] - input_slice_array[i]

                red_difference_slice = difference_slice.copy()
                red_difference_slice[red_difference_slice < 0.0] = 0.0

                red_difference_slice_list.append(red_difference_slice)

                blue_difference_slice = difference_slice.copy()
                blue_difference_slice[blue_difference_slice > 0.0] = 0.0
                blue_difference_slice = np.abs(blue_difference_slice)

                blue_difference_slice_list.append(blue_difference_slice)

                difference_slice = np.zeros(difference_slice.shape)

                difference_slice_list.append(difference_slice)

            difference_slice_array = np.asfarray(difference_slice_list)

            red_difference_slice_array = np.asfarray(red_difference_slice_list)
            red_difference_slice_array = rescale_array_array_path(red_difference_slice_array)

            blue_difference_slice_array = np.asfarray(blue_difference_slice_list)
            blue_difference_slice_array = rescale_array_array_path(blue_difference_slice_array)

            for i in range(len(input_slice_array)):
                red_slices = np.vstack((ground_truth_slice_array[i], red_difference_slice_array[i], input_slice_array[i]))
                green_slices = np.vstack((ground_truth_slice_array[i], difference_slice_array[i], input_slice_array[i]))
                blue_slices = np.vstack((ground_truth_slice_array[i], blue_difference_slice_array[i], input_slice_array[i]))

                frame_image = toimage(np.array((red_slices, green_slices, blue_slices))).convert("RGBA", dither=None, palette="WEB")

                frame_image.save("{0}/{1}.png".format(output_path, str(i)))

                frames.append(frame_image)
        else:
            for i in range(len(input_slice_array)):
                frame_image = toimage(np.array((input_slice_array[i], input_slice_array[i], input_slice_array[i]))).convert("RGBA", dither=None, palette="WEB")

                frame_image.save("{0}/{1}.png".format(output_path, str(i)))

                frames.append(frame_image)
    else:
        # rescale individually
        for i in range(len(input_paths)):
            input_slice = None

            if slice_axis == 0:
                input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[slice_position, :, :], 0)
                input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * pixel_width)), int(round(input_slice.shape[1] * pixel_width))))
            else:
                if slice_axis == 1:
                    input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[:, slice_position, :], 0)
                    input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * slice_width)), int(round(input_slice.shape[1] * pixel_width))))
                else:
                    if slice_axis == 2:
                        input_slice = np.flip(pet.ImageData(input_paths[i]).as_array()[:, :, slice_position], 0)
                        input_slice = imresize(input_slice, (int(round(input_slice.shape[0] * slice_width)), int(round(input_slice.shape[1] * pixel_width))))

            input_slice = rescale_linear_max_min(input_slice, 0, 255)

            if ground_truth_paths is not None:
                ground_truth_slice = None

                if slice_axis == 1:
                    ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[slice_position, :, :], 0)
                    ground_truth_slice = imresize(ground_truth_slice, (
                    int(round(ground_truth_slice.shape[0] * pixel_width)),
                    int(round(ground_truth_slice.shape[1] * pixel_width))))
                else:
                    if slice_axis == 2:
                        ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[:, slice_position, :], 0)
                        ground_truth_slice = imresize(ground_truth_slice, (
                        int(round(ground_truth_slice.shape[0] * slice_width)),
                        int(round(ground_truth_slice.shape[1] * pixel_width))))
                    else:
                        if slice_axis == 3:
                            ground_truth_slice = np.flip(pet.ImageData(ground_truth_paths[i]).as_array()[:, :, slice_position], 0)
                            ground_truth_slice = imresize(ground_truth_slice, (
                            int(round(ground_truth_slice.shape[0] * slice_width)),
                            int(round(ground_truth_slice.shape[1] * pixel_width))))

                ground_truth_slice = rescale_linear_max_min(ground_truth_slice, 0, 255)

                difference_slice = ground_truth_slice - input_slice

                red_difference_slice = difference_slice.copy()
                red_difference_slice[red_difference_slice < 0.0] = 0.0
                red_difference_slice = rescale_linear_max_min(red_difference_slice, 0, 255)

                blue_difference_slice = difference_slice.copy()
                blue_difference_slice[blue_difference_slice > 0.0] = 0.0
                blue_difference_slice = rescale_linear_max_min(np.abs(blue_difference_slice), 0, 255)

                difference_slice = np.zeros(difference_slice.shape)

                red_slices = np.vstack((ground_truth_slice, red_difference_slice, input_slice))
                green_slices = np.vstack((ground_truth_slice, difference_slice, input_slice))
                blue_slices = np.vstack((ground_truth_slice, blue_difference_slice, input_slice))

                frame_image = toimage(np.array((red_slices, green_slices, blue_slices))).convert("RGBA", dither=None, palette="WEB")
            else:
                frame_image = toimage(np.array((input_slice, input_slice, input_slice))).convert("RGBA", dither=None, palette="WEB")

            # output png
            frame_image.save("{0}/{1}.png".format(output_path, str(i)))

            frames.append(frame_image)

    # output gif
    frames[0].save("{0}/{1}".format(output_path, str(output_name)), format="GIF", append_images=frames[1:],
                   save_all=True, duration=int(round(1000 * frame_time)), optimize=False, loop=0)

    np.seterr(all="raise")

    return True


if __name__ == "__main__":
    main()
