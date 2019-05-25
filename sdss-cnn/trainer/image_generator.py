import numpy as np
from PIL import Image

class ImageAugmentor():

    def __init__(self, zoom_factor=0.1):
        self.image_augementation_generator = self.__image_augementation_generator(zoom_factor)

    def __image_augementation_generator(self, zoom_factor):
        def _rotate_angle_generator():
            while True:
                angle = np.random.randint(0, 360)
                yield angle

        def _zoom_generator():
            while True:
                zoom = np.round(zoom_factor * (np.random.random() - 0.5), 3) + 1.0
                yield zoom

        def _boolean_generator():
            while True:
                prob = np.random.random()
                yield prob >= 0.5

        rotate_angle_generator = _rotate_angle_generator()
        zoom_generator = _zoom_generator()
        horiztonal_flip_generator = _boolean_generator()
        vertical_flip_generator = _boolean_generator()
        
        while True:
            rotate_angle = next(rotate_angle_generator)
            zoom = next(zoom_generator)
            horiztonal_flip = next(horiztonal_flip_generator)
            vertical_flip = next(vertical_flip_generator)
            
            yield rotate_angle, zoom, horiztonal_flip, vertical_flip

    def __flip_horizontal(self, data):
        return np.flip(data, 2)

    def __flip_vertical(self, data):
        return np.flip(data, 1)

    def __rotate(self, data, deg):
        for idx in range(data.shape[0]):
            img = Image.fromarray(data[idx])
            img = img.rotate(deg, resample=Image.BICUBIC, expand=True)
            
            if (idx == 0):
                out = np.empty((3,) + img.size)

            out[idx] = np.array(img)

        return out

    def __resize(self, data, factor):
        new_size = int(np.round(factor * data.shape[1]))
        
        out = np.empty((3, new_size, new_size))
        
        for idx in range(data.shape[0]):
            img = Image.fromarray(data[idx])
            img = img.resize((new_size, new_size), resample=Image.BICUBIC)
            out[idx] = np.array(img)
        return out

    def augment_image(self, img_data, verbose=False):
        if np.isnan(img_data).any():
            raise Exception('WARNING: input image has NAN!!! This should not be!!!')

        img_min = np.min(img_data)
        angle, zoom, flip_h, flip_v = next(self.image_augementation_generator)

        if verbose:
            print(f'Rotate: {angle}, Zoom: {zoom}, Flip Horizon.: {flip_h}, Flip Vertical: {flip_v}')

        data = self.__rotate(img_data, angle)
        data = self.__resize(data, zoom)

        if flip_h:
            data = self.__flip_horizontal(data)
        
        if flip_v:
            data = self.__flip_vertical(data)

        nans = np.isnan(data)
        data[nans] = img_min

        return data

# def crop(data, crop_size=42):
#     curr_size = data.shape[1]
#     out = np.empty((3, 42, 42))

#     top = int((curr_size - crop_size)/2)
#     bttm = top + crop_size

#     for idx in range(3):
#         out[idx] = data[idx, top:bttm, top:bttm]

#     return out
