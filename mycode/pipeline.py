from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import color, transform, restoration, io, feature
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
import os


class ImagePipeline(object):

    def __init__(self, parent_dir):
        """
        Manages reading, transforming and saving images

        :param parent_dir: Name of the parent directory containing all the sub directories
        """
        # Define the parent directory
        self.parent_dir = parent_dir

        # Sub directory variables that are filled in when read()
        self.raw_sub_dir_names = None
        self.sub_dirs = None
        self.label_map = None

        # Image variables that are filled in when read() and vectorize()
        self.img_lst2 = []
        self.img_names2 = []
        self.patches_lst2 = []
        self.features = None
        self.labels = None

    def _make_label_map(self):
        """
        Get the sub directory names and map them to numeric values (labels)

        :return: A dictionary of dir names to numeric values
        """
        return {label: i for i, label in enumerate(self.raw_sub_dir_names)}

    def _path_relative_to_parent(self, some_dir):
        """
        Get the full path of a sub directory relative to the parent

        :param some_dir: The name of a sub directory
        :return: Return the full path relative to the parent
        """
        cur_path = os.getcwd()
        return os.path.join(cur_path, self.parent_dir, some_dir)

    def _make_new_dir(self, new_dir):
        """
        Make a new sub directory with fully defined path relative to the parent directory

        :param new_dir: The name of a new sub dir
        """
        # Make a new directory for the new transformed images
        new_dir = self._path_relative_to_parent(new_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        else:
            raise Exception('Directory already exist, please check...')

    def _empty_variables(self):
        """
        Reset all the image related instance variables
        """
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None

    @staticmethod
    def _accpeted_file_format(fname):
        """
        Return boolean of whether the file is of the accepted file format

        :param fname: Name of the file in question
        :return: True or False (if the file is accpeted or not)
        """
        formats = ['.png', '.jpg', '.jpeg', '.tiff']
        for fmt in formats:
            if fname.endswith(fmt):
                return True
        return False

    @staticmethod
    def _accepted_dir_name(dir_name):
        """
        Return boolean of whether the directory is of the accepted name (i.e. no hidden files)

        :param dir_name: Name of the directory in question
        :return: True or False (if the directory is hidden or not)
        """
        if dir_name.startswith('.'):
            return False
        else:
            return True

    def _assign_sub_dirs(self, sub_dirs=tuple('all')):
        """
        Assign the names (self.raw_sub_dir_names) and paths (self.sub_dirs) based on
        the sub dirs that are passed in, otherwise will just take everything in the
        parent dir

        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        """
        # Get the list of raw sub dir names
        if sub_dirs[0] == 'all':
            self.raw_sub_dir_names = os.listdir(self.parent_dir)
        else:
            self.raw_sub_dir_names = sub_dirs
        # Make label to map raw sub dir names to numeric values
        self.label_map = self._make_label_map()

        # Get the full path of the raw sub dirs
        filtered_sub_dir = filter(self._accepted_dir_name, self.raw_sub_dir_names)
        self.sub_dirs = map(self._path_relative_to_parent, filtered_sub_dir)

    def read(self, sub_dirs=tuple('all'), subsample=False):
        """
        Read images from each sub directories into a list of matrix (self.img_lst2)

        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        :param subsample: Only take a certain # of images from each subdirectoy
        """
        # Empty the variables containing the image arrays and image names, features and labels
        self._empty_variables()

        # Assign the sub dir names based on what is passed in
        self._assign_sub_dirs(sub_dirs=sub_dirs)

        if type(subsample) == int:
            for sub_dir in self.sub_dirs:
                img_names = filter(self._accpeted_file_format, os.listdir(sub_dir)[:subsample])
                self.img_names2.append(img_names)

                img_lst = [io.imread(os.path.join(sub_dir, fname)) for fname in img_names]
                self.img_lst2.append(img_lst)
        else:
            for sub_dir in self.sub_dirs:
                img_names = filter(self._accpeted_file_format, os.listdir(sub_dir))
                self.img_names2.append(img_names)

                img_lst = [io.imread(os.path.join(sub_dir, fname)) for fname in img_names]
                self.img_lst2.append(img_lst)

    def save(self, keyword):
        """
        Save the current images into new sub directories

        :param keyword: The string to append to the end of the original names for the
                        new sub directories that we are saving to
        """
        # Use the keyword to make the new names of the sub dirs
        new_sub_dirs = ['%s.%s' % (sub_dir, keyword) for sub_dir in self.sub_dirs]

        # Loop through the sub dirs and loop through images to save images to the respective subdir
        for new_sub_dir, img_names, img_lst in zip(new_sub_dirs, self.img_names2, self.img_lst2):
            new_sub_dir_path = self._path_relative_to_parent(new_sub_dir)
            self._make_new_dir(new_sub_dir_path)

            for fname, img_arr in zip(img_names, img_lst):
                io.imsave(os.path.join(new_sub_dir_path, fname), img_arr)

        self.sub_dirs = new_sub_dirs

    def show(self, sub_dir, img_ind):
        """
        View the nth image in the nth class

        :param sub_dir: The name of the category
        :param img_ind: The index of the category of images
        """
        sub_dir_ind = self.label_map[sub_dir]
        io.imshow(self.img_lst2[sub_dir_ind][img_ind])
        plt.show()

    def transform(self, func, params, sub_dir=None, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr.
        Have to option to transform one as  a test case

        :param sub_dir: The index for the image
        :param img_ind: The index of the category of images
        """
        # Apply to one test case
        if sub_dir is not None and img_ind is not None:
            sub_dir_ind = self.label_map[sub_dir]
            img_arr = self.img_lst2[sub_dir_ind][img_ind]
            img_arr = func(img_arr, **params).astype(float)
            io.imshow(img_arr)
            plt.show()
        # Apply the function and parameters to all the images
        elif isinstance(sub_dir, list):
            if len(sub_dir) == 1:
                sub_dir_ind = self.label_map[sub_dir[0]]
                new_img_lst2 = []
                for img_arr in self.img_lst2[sub_dir_ind]:
                    new_img_lst2.append(func(img_arr, **params).astype(float))
                self.img_lst2[sub_dir_ind] = new_img_lst2
            else:
                for dir in sub_dir:
                    sub_dir_ind = self.label_map[dir]
                    new_img_lst2 = []
                    for img_arr in self.img_lst2[sub_dir_ind]:
                        new_img_lst2.append(func(img_arr, **params).astype(float))
                    self.img_lst2[sub_dir_ind] = new_img_lst2
        else:
            new_img_lst2 = []
            for img_lst in self.img_lst2:
                new_img_lst2.append([func(img_arr, **params).astype(float) for img_arr in img_lst])
            self.img_lst2 = new_img_lst2

    def transform_patches(self, func, params):
        """
        Takes a function and apply to every patch_arr in self.patches_lst2.
        Have to option to transform one as  a test case

        :param sub_dir: The index for the image
        :param img_ind: The index of the category of images
        """
        new_patches_lst2 = []
        for patch_lst in self.patches_lst2:
            new_patches_lst2.append([func(patch_arr, **params).astype(float) for patch_arr in patch_lst])
        self.patches_lst2 = new_patches_lst2

    def grayscale(self, sub_dir=None, img_ind=None, isImage=True):
        """
        Grayscale all the images in self.img_lst2

        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.        
        """
        if isImage:
            self.transform(color.rgb2gray, {}, sub_dir=sub_dir, img_ind=img_ind)
        else:
            self.transform_patches(color.rgb2gray, {})

    def canny(self, sigma=2.5, sub_dir=None, img_ind=None, isImage=True):
        """
        Apply the canny edge detection algorithm to all the images in self.img_lst2

        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.        
        """
        if isImage:
            self.transform(feature.canny, dict(sigma=sigma), sub_dir=sub_dir, img_ind=img_ind)
        else:
            self.transform_patches(feature.canny, dict(sigma=sigma))

    def sobel(self, sub_dir=None, img_ind=None, isImage=True):
        """
        Apply the sobel edge detection algorithm to all the images in self.img_lst2

        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.
        """
        if isImage:
            self.transform(filters.sobel, {}, sub_dir=sub_dir, img_ind=img_ind)
        else:
            self.transform_patches(filters.sobel, {})

    def denoise_bilateral(self, sigma_range=0.15, sub_dir=None, img_ind=None, isImage=True):
        """
        Apply to Bi-lateral denoise to all the images in self.img_lst2

        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.
        """
        if isImage:
            self.transform(restoration.denoise_bilateral,
                        dict(sigma_range = sigma_range),
                        sub_dir=sub_dir, img_ind=img_ind)
        else:
            self.transform_patches(estoration.denoise_bilateral, dict(sigma_range=sigma_range))

    def tv_denoise(self, weight=2, multichannel=True, sub_dir=None, img_ind=None, isImage=True):
        """
        Apply to total variation denoise to all the images in self.img_lst2

        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.
        """
        if isImage:
            self.transform(restoration.denoise_tv_chambolle,
                           dict(weight=weight, multichannel=multichannel),
                           sub_dir=sub_dir, img_ind=img_ind)
        else:
            self.transform_patches(restoration.denoise_tv_chambolle,
                           dict(weight=weight, multichannel=multichannel))

    def resize(self, shape, save=False):
        """
        Resize all images in self.img_lst2 to a uniform shape

        :param shape: A tuple of 2 or 3 dimensions depending on if your images are grayscaled or not
        :param save: Boolean to save the images in new directories or not
        """
        self.transform(transform.resize, dict(output_shape=shape))
        if save:
            shape_str = '_'.join(map(str, shape))
            self.save(shape_str)

    def image_to_pixels(self, image_array):
        """
        Convert Image Array to Pixels
        """
        nrow, ncol, depth = image_array.shape 
        lst_of_pixels = [image_array[irow][icol] for irow in range(nrow) for icol in range(ncol)]
        return lst_of_pixels

    def images_to_dominant_colors(self, n_clusters=3):
        """
        Uses kmeans to extract the most dominant colors
        for all images in self.img_lst2

        :param n_clusters: Designate the number of dominant colors to form as well as 
        the number of centroids to generate.
        """
        print 'Determining the dominant colors for each image in the pipeline'
        image_dominant_colors = []
        for img_list in self.img_lst2:
            for img_arr in img_list:
                img_Pixels = np.r_[self.image_to_pixels(img_arr)]
                KMeans_Model = MiniBatchKMeans(n_clusters=n_clusters)
                KMeans_Model.fit(img_Pixels)
                clusters = KMeans_Model.cluster_centers_
                image_dominant_colors.append(np.ravel(clusters))
        self.dominant_colors = np.r_[image_dominant_colors]

    def merge_features_dominant_colors(self, isImage=True):
        """
        Merge the features matrix with the self.dominant_colors matrix

        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.
        :return numpy array of this combination
        """
        if isImage:
            return np.concatenate((self.features, self.dominant_colors), axis=1)
        else:
            return np.concatenate((self.patch_features, self.patch_dominant_colors), axis=1)

    def extract_patch(self, img_array, patch_size=(80,96), max_patches=30):
        """
        Reshape a 2D image into a collection of patches and extract it into a dedicated array

        :return list of patches extracted from the image
        """
        patches = extract_patches_2d(img_array, patch_size, max_patches)
        self.n_patches = patches.shape[0]
        return list(patches)

    def images_to_patches(self, patch_size=(80,96), max_patches=30):
        """
        Extract patches for all images in the pipeline.
        """
        print 'Extracting patches from all images in the pipeline.'
        for img_list in self.img_lst2:
            for img_arr in img_list:
                patches = self.extract_patch(img_arr, patch_size=patch_size, max_patches=max_patches)
                self.patches_lst2.append(patches)

    def patches_to_dominant_colors(self, n_clusters=3):
        """
        Determine the dominant colors for each patch in the pipeline
        """
        print 'Determining the dominant colors for each patch in the pipeline'
        patch_dominant_colors = []
        for patch_list in self.patches_lst2:
            for patch_arr in patch_list:
                patch_Pixels = np.r_[self.image_to_pixels(patch_arr)]
                KMeans_Model = MiniBatchKMeans(n_clusters = n_clusters)
                KMeans_Model.fit(patch_Pixels)
                clusters = KMeans_Model.cluster_centers_
                patch_dominant_colors.append(np.ravel(clusters))
        self.patch_dominant_colors = np.r_[patch_dominant_colors]

    def _vectorize_features(self):
        """
        Take a list of images and vectorize all the images. Returns a feature matrix where each
        row represents an image
        """
        row_tup = tuple(img_arr.ravel()[np.newaxis, :]
                        for img_lst in self.img_lst2 for img_arr in img_lst)
        self.test = row_tup
        self.features = np.r_[row_tup]

    def _vectorize_patch_features(self):
        """
        Take a list of patches and vectorize all the patches. Returns a feature matrix where each
        row represents a patch image
        """
        row_tup = tuple(patch_arr.ravel()[np.newaxis, :]
                        for patch_lst in self.patches_lst2 for patch_arr in patch_lst)
        self.test = row_tup
        self.patch_features = np.r_[row_tup]    

    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = np.concatenate([np.repeat(i, len(img_names))
                                      for i, img_names in enumerate(self.img_names2)])

    def _vectorize_patch_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        corresponding to patches for each image
        """
        if self.labels is None:
            self._vectorize_labels()
        self.patch_labels = np.repeat(self.labels, self.n_patches)

    def vectorize(self, isImage=True):
        """
        Return (feature matrix, the response) if output is True, otherwise set as instance variable.
        Run at the end of all transformations

        :param isImage: If True, perform operations on images in img_lst2.
         If False, perform operations on patches in patch_lst.
        """
        print 'Generating the Features Matrix'
        if isImage:
            self._vectorize_features()
            self._vectorize_labels()
        else:
            if not self.patches_lst2:
                self.images_to_patches()
                self.patches_to_dominant_colors(n_clusters=3)
            self._vectorize_patch_features()
            self._vectorize_patch_labels()
