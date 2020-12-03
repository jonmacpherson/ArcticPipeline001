import albumentations as albu
import cv2
import functools
from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import pydicom as pydicom
import random
import seaborn as sn
from skimage import exposure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader, Dataset
import tqdm

# A nice debugging tool, just set trace=True during object __init__ 
def trace_pipeline(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if CFG._trace:
            print("Running Function", func)
        return func(*args, **kwargs)
    return wrapper
        

# A way to call functions like they are called in Albumentations compose function:
# https://stackoverflow.com/questions/57378957/how-to-compose-combine-several-function-calls-with-arguments

def make_callable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)
    return wrapper

class CFG():
    
    _trace = False

class ArcticPipeline(Dataset):
    """arctic = ArcticPipeline(df=dataframe_to_use, image_field='filename', prefix_path_with = "") 
    
    """
    
    
    def __init__(self, batch_size=16, warnings=True, dataset_path="", trace=False,\
                 image_field="image", device=None, shuffle=True, pin_memory=True, num_workers=0 ):
        
        """
        image_field is the column name in the dataframe. 
        
        trace can be set to True to produce a log of ArcticPipeline Methods as they run. 
        
        device is passed to pytorch as either "cpu" or "cuda" if available. It can be overridden. 
        """
        
        CFG._trace = trace
        
        
        # PYTORCH STUFFS:
        if device:
            self._device = device
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("CUDA", self._device)
        
        # Dataframe management:
        train_file = os.path.join(dataset_path, "AP_train.csv")
        test_file = os.path.join(dataset_path, "AP_test.csv")
        val_file = os.path.join(dataset_path, "AP_val.csv")
        all_file = os.path.join(dataset_path, "AP_all.csv")
        
        # Build a data structure that defines all of the dataset, with each sub-portion represented by a key, and the values
        # being a tuple containing 0-> The file where the data resides, 1-> the actual dataframe object that contains the data. 
        self._ap_ds_train = pd.DataFrame()
        self._ap_ds_test = pd.DataFrame()
        self._ap_ds_val = pd.DataFrame()
        self._ap_ds_all = pd.DataFrame()
        
        self._ap_dataset_files_n_buckets = {    "train" : [ train_file, self._ap_ds_train ], 
                                                "test" :  [ test_file, self._ap_ds_test ],
                                                "val" :   [ val_file, self._ap_ds_val ],
                                                "all" :   [ all_file, self._ap_ds_all ],
                                           }

        # Data_mode defaults to "all" since it is assumed that any user would need to import the whole dataset, perform actions on it, 
        # and then split that dataset into train, test, val. 
        self._data_mode = "all"
        
        # For image normalization
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.normalize_function = Normalize(self.normalize_mean, self.normalize_std)
                
        # Require warnings unless actively turned off. Just Do It: Always warn of potential erros. 
        self._warnings = warnings
        
        ####################################### BATCHING #############################################################
        # Batch size will be used for generating static datasets OR dynamically generated datasets used by dataloaders. 
        # For dynamic datasets, assume that there are multiple images available to choose for each record, or multiple areas within 
        # the image that could be used. Imagine feeding a 1024 x 1024 image into a CNN ?? yeah, no , 
        # I'd rather send a bunch of segments at 120 x 120  
        # TODO
        self.batch_size = batch_size
        
        ######################### Image related variables ############################################################
        # Image prep pipeline will work similar to a sub-process call with a list of functions to call that will prep the image
        self._img_prep_pipeline = []
              
        # A shortcut to save time passing around the image 
        self._img_field = image_field
        
        # Where image data will be stored when read from files and passed between methods:
        self._img = ''  
        
        # A place to keep track of how many images have been processed/viewed/etc:
        self._num_img = 0
        
        ######################### DATALOADER VARIABLES ###############################################################
        # batch size is defined above
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        
        self._dataset_source = None
    
 
    @trace_pipeline
    def _import_dataframe (self, df_file, bucket, append_mode=False):
        
        # Import a dataframe 
        self.data_mode = bucket
        
        try:
            self.df = pd.read_csv(df_file)            
            
        except Exception as err:
            print("Attempted loading file:", df_file, "into bucket:", bucket, "failed. Error:", err )

            
    @trace_pipeline    
    def _export_dataframe(self, df_file, df_bucket):
        
        self.data_mode = df_bucket
        try:
            self.df.to_csv(df_file, index=False)
            print("Wrote file:", df_file)
        except Exception as err:
            print("Error writting out dataset component file to:", df_file, "Error:", err)
            
        
    @trace_pipeline    
    def import_dataset(self, dataset_folder):
        
        # retain the original mode while loading data, and set it back when done. 
        original_data_mode = self.data_mode
        
        # Only used by __repr__ 
        self._dataset_source = dataset_folder
        
        for bucket in self._ap_dataset_files_n_buckets.keys():
            
            dataframe_file = os.path.join(dataset_folder, self._ap_dataset_files_n_buckets[bucket][0])
             
            if os.path.exists(dataframe_file):
                #print("Found file, importing bucket", bucket)
                self._import_dataframe(dataframe_file, bucket) 
      
        self.data_mode = original_data_mode
        
    @trace_pipeline    
    def export_dataset(self, dataset_folder):
        
        # retain the original mode while loading data, and set it back when done. 
        original_data_mode = self.data_mode
        
        for bucket in self._ap_dataset_files_n_buckets.keys():
            # Check to see if the bucket exists 
            if len(self._ap_dataset_files_n_buckets[bucket][1]) > 0:
                self._export_dataframe(self._ap_dataset_files_n_buckets[bucket][0], bucket) 
                
        self.data_mode = original_data_mode
    
    def __repr__(self):
        
        repr_string =  self.__class__.__name__ + " @: " + self._dataset_source + "\n"
        
        for bucket in self._ap_dataset_files_n_buckets.keys():
            items_in_bucket = len(self._ap_dataset_files_n_buckets[bucket][1])
            repr_string += "Bucket: " + bucket + " Len:" + str(items_in_bucket) + "\n"
        
        return repr_string
    

    
    # Getters and Setters Here:
    @property
    def df(self):
        
        # see __init__ for the useable buckets that are available. They are defined in self._ap_dataset_files_n_buckets
        
        if self.data_mode in self._ap_dataset_files_n_buckets.keys():
            #print("Mode is:", self.data_mode, "length is:", len(self._ap_dataset_files_n_buckets[self.data_mode][1]))
            
            return self._ap_dataset_files_n_buckets[self.data_mode][1]
        
        else:
            print("WARNING: ",self,".data_mode of ", self.data_mode, "is invalid. Possible values are:", \
                  self._ap_dataset_files_n_buckets.keys(), "Defaulting to data_mode of 'all'")
            print("\"all\" Mode Active, len is", len(self._ap_dataset_files_n_buckets['all'][1]))
            self.data_mode = "all"
            
            return self._ap_dataset_files_n_buckets['all'][1]
    
    
    @df.setter
    def df(self, df):
        
        if isinstance(df, pd.DataFrame):
        
            if self.data_mode in self._ap_dataset_files_n_buckets.keys():
                print("Setting the dataframe: ", self.data_mode, "to dataframe containing: ", len(df))

            else:
                print("WARNING: ",self,".data_mode of ", self.data_mode, "is invalid. Possible values are:", \
                  self._ap_dataset_files_n_buckets.keys(), "Defaulting to data_mode of 'all'")
                self.data_mode = "all"
                
            self._ap_dataset_files_n_buckets[self.data_mode][1] = df
                
        else: 
            print("object:", df, "Is not a dataframe.")
    
    @property
    def data_mode(self):
        return self._data_mode
    
    @data_mode.setter
    def data_mode(self, data_mode):
        
        if data_mode in self._ap_dataset_files_n_buckets.keys():
            self._data_mode = data_mode
        else: 
            print("Unknown data mode specified:", data_mode, "Defaulting to \"all\" mode data.")
            self._data_mode = "all"
            # TODO raise a datamode error. 
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size
        
    @property
    def shuffle(self):
        return self._shuffle
    
    @shuffle.setter
    def shuffle(self, state):
        if state is True or state is False:
            self._shuffle = state
        else:
            print("Shuffle must be True or False")
    
    @property
    def num_workers(self):
        return self._num_workers
        
    @num_workers.setter
    def num_workers(self, number):
        self._num_workers = number
        
        
    @property
    def pin_memory(self):
        return self._pin_memory
    
    @pin_memory.setter
    def pin_memory(self, state):
        if state is True or state is False:
            self._pin_memory = state
        else:
            print("Pin Memory must be True or False")
            
    @property
    def shared_work(self):
        # Note, no setter. Should be set on init.
        return self._shared_work
    
    
    @property
    def shared_work_file(self):
        # Note, no setter. Should be set on init.
        return self._shared_work_file
    
    
    @property
    def shared_lock_file(self):
        # Note, no setter. Should be set on init.
        return self._shared_lock_file
    
    
    @property
    def image_field(self):
        return self._image_field
    
    
    @image_field.setter
    def image_field(self, field):
        self._image_field = field
    
    
##################################### PROCESS & SUB PROCESS #######################################################
        
    # The heart of the pipeline
    @trace_pipeline
    def compose_data(self, bucket_list=["all"], command_list=None):
        
        # The initial version of compose_data() will be intended to build a static training/testing set.
        # Later iterations of this may support feeding data directly as a dataloader/dataset combination. 
        
        # to ensure all buckets are processed correctly, iterate through the bucket list, and perform all actions in command list on 
        # each bucket. Only do this in compose_data, not iterate_records, as will be done with the current command list.... 
        
        original_data_mode = self.data_mode
        
        for bucket in bucket_list:
        
            self.data_mode = bucket
            for name, command in command_list:

                command()
                
        self.data_mode = original_data_mode
            
    @make_callable
    @trace_pipeline
    def iterate_records(self, command_list):
        
        # While Arctic_Pipeline.compose_data works on whole columns of data, iterate_records steps through one record at a time, then one
        # command at a time. This is usefull for processing images. For example, you wouldn't open 100 images and store them in memory
        # then resize them all at once, and finally write them out. It would be more memory efficient to do one at a time. 
        
        for index, row in self.df.iterrows():
            
            self._active_id = index
            
            for name, command in command_list:

                # This needs to be updated after each command, as commands may modify it
                self._active_row = self.get_current_record()

                command()
                
                         
    @trace_pipeline
    def process_image(self):

        command_list = self._process_image_command_list
        
        for name, command in command_list:
            
            # This needs to be updated after each command, as commands may modify it
            self._active_row = self.get_current_record()

            command()
            
    @trace_pipeline    
    def define_process_image_command_list(self, command_list):
        self._process_image_command_list = command_list
        
    def define_scaler(self, scaler=MinMaxScaler()):
        self._scaler = scaler
        
        
        
########################################################### RECORDS, DATA, STATE

    @make_callable
    @trace_pipeline
    def show_state(self, n=10, frac=None, active=True):
        """Displays 10 random dataframe records from arctic.df by default. To show 5 records use arctic.show_state(n=5) or to show
        5 percent of records use arctic.show_state(frac=.05). 
        
        active allows this method to be easily turned on and off from the compose list. 
        """
        if active:
            if frac is not None:
                print(self.df.sample(frac=frac))
            else:
                print(self.df.sample(n=n))
                
        elif self._warnings is True:
            print("Arctic_Pipeline.show_state() is set to inactive. To turn it on, use .show_state(active=True) in the compose method")
    

    def get_current_record(self):
        #print(self.df.iloc[self._active_id])
        return self.df.iloc[self._active_id]
    
    # TODO CHECK THIS OUT, add failsafe code with helpful message, and use it. 
    def get_current_record_item(self, item):
        if item in self.df:
            return self.df.iloc[self._active_id][item]
    
    @make_callable
    @trace_pipeline
    def convert_field_to_image(self, field, desired_shape=(10,10)):
        
        field = self.df[field]
        
        if isinstance(field,(np.ndarray)):
            pass
        else: 
            print("Expected object type np.ndarray")
    
    @make_callable
    @trace_pipeline
    def add_field(self, name, prefix='', postfix='', source_field=None, default_value=None):
        
        # If there is a source category that matches 'source_field',
        # pull the value from that, otherwise, use the 'default_value' for all items in this category
        
        if source_field:
            if source_field in self.df:
                self.df[name] = prefix + self.df[source_field] + postfix
                
            else:
                sample_size = min(5, len(self.df))
                print("Source field:", source_field, "does not exist in Dataset:", self.df.sample(n=sample_size))
        else:
            self.df[name] = default_value
    
    @make_callable
    @trace_pipeline
    def combine_two_fields(self, new_field_name, source_fields, field_seperator='_', appendage=None, cast_key=int ):
        """cast_key is how the data should be converted before being joined. Most likely this would be str, but it could be int
        source_fields should be a list of two column names or their interger location. """
        
        extra = ''
        
        if appendage is not None:
            extra = appendage 
        
        if len(source_fields) == 2:
            
            # Try string specified column names first:
            try:
                field_1 = self.df.loc[:,source_fields[0]].astype(str)
                field_2 = self.df.loc[:,source_fields[1]].astype(str)
            
            # If that doesn't work try column numbers: 
            except TypeError:
                
                try:
                    field_1 = self.df.iloc[:, source_fields[0]].astype(str)
                    field_2 = self.df.iloc[:, source_fields[1]].astype(str)
                    
                except KeyError:
                    print("Source Columns Not found", source_fields)
                    
                
            #self.df[new_field_name] = field_1 + field_seperator + field_2 + extra
            self.df[new_field_name] = field_1 + field_seperator + field_2 + extra
           
        else:
            # I'm wondering if there should be a way to turn off this warning message ?? I guess if I come across a serious enough 
            # reason I'll add that functionality later. 
            print("Only two fields can be combined with add_combined_files(). To combine multiple field, use multiple calls to")
            print("add_combined_fields() with two fields specified at a time. Then remove the excess fields with")
            print("remove_fields()")
            
########################################################### IMAGES
            
    @make_callable
    @trace_pipeline
    def random_image_choice_from_folder(self, source_folder_column, column_name_to_create):
        # step through each record, then select a random image from source_folder_column and add that full filename to the dataframe
        
        self.df[column_name_to_create] = ''
        
        for index, row in self.df.iterrows():
            
            folder_of_files = row[source_folder_column]
            files_to_choose_from = os.listdir(folder_of_files)
            
            the_choosen_one = random.choice(files_to_choose_from)
            full_path_and_file = os.path.join(folder_of_files, the_choosen_one)
            
            self.df.loc[index, column_name_to_create] = full_path_and_file

                
    @make_callable
    @trace_pipeline
    def np_resize(self, size, interpolation=cv2.INTER_CUBIC):
        """Arctic_Pipeline.np_resize works on the image object stored in Arctic_Pipeline._img which should be loaded by anothher 
        Arctic_Pipeline method such as Arctic_Pipeline.read_in()
        
        Output is stored in Arctic_Pipeline._img or self._img and should be used by another method to be written or displayed. 
        
        """
        x=size[0]
        y=size[1]
        self._img = cv2.resize(self._img, dsize=(y,x), interpolation=interpolation)
            
        
    @make_callable
    @trace_pipeline
    def crop_out_center(self, crop_size):
        # This utility will allow excess space to be chopped off 
        # Crop size determines the size of the crop, which will be centered on the image. 

        shapes = self._img.shape


        if shapes[0] > crop_size[0] and shapes[1] > crop_size[0]:
            extra_0 = shapes[0] - crop_size[0]
            extra_1 = shapes[1] - crop_size[1]

            start_0 = int(extra_0 / 2)
            start_1 = int(extra_1 / 2)
            end_0 = start_0 + crop_size[0]
            end_1 = start_1 + crop_size[1]
            #print("E0:", extra_0, "E1:", extra_1, "S0:", start_0, "S1:", start_1)
            self._img = self._img[start_0:crop_size[0], start_1:crop_size[1]]
            
        
    @make_callable
    @trace_pipeline
    def crop_random_area(self, t_crop_size):
        
        orig_img_shape = self._img.shape
        
        #print(orig_img_shape)
        
        wiggle_room_x = orig_img_shape[0] - t_crop_size[0]
        wiggle_room_y = orig_img_shape[1] - t_crop_size[1]
        
        start_x = int(random.uniform(0, wiggle_room_x))
        start_y = int(random.uniform(0, wiggle_room_y))
        
        #print("Random crop at", start_x, start_y, "thru", start_x + t_crop_size[0], start_y+t_crop_size[1])
        
        # automagically work on RGB or Gray images:
        if len(orig_img_shape) == 2:
            self._img = self._img[start_x:start_x+t_crop_size[0], start_y:start_y+t_crop_size[1]]
        elif len(orig_img_shape) == 3:
            self._img = self._img[start_x:start_x+t_crop_size[0], start_y:start_y+t_crop_size[1],:]
        else:
            print("Error, Shape of image is ", orig_img_shape, "Only images with 2 or 3 dimensions are supported")
        
        
    @make_callable
    @trace_pipeline    
    def show_x_images(self, x, **show_args):
        #print("Break")
        if self._num_img < x:
            
            plt.imshow(self._img, **show_args)
            plt.show()
            
        self._num_img += 1
            
    
    @make_callable
    def read_in_img_callable(self, image_field, custom_reader=None, color_option=None, equalize=False):
        """many functions that are used in the iterate_records sub processor need @make_callable to function properly,
        however this decorator breaks other parts of the program, so two seperate functions are needed. """
        
        # In fact any function that is called without using the callable syntac will need two different functions.
        # the callable syntax is arctic.read_in_img_callable() as part of the iterate_records sub-processor. 
        
        self.read_in_image(image_field=image_field,
                           custom_reader=custom_reader,
                           color_option=color_option,
                           equalize=equalize)
        
    @trace_pipeline
    def read_in_img(self, image_field, custom_reader=None, color_option=None, equalize=False):

        if custom_reader is not None:
            img = custom_reader
            
        else:
            #print("IN:", image_field, "row", self._active_row)
            image_name = self._active_row[image_field]
            image_type = image_name.split('.')[-1]
            
            #if self._prefix_path_with is not None:
            #    image_name = self._prefix_path_with + image_name
            
            readers = {'dcm' : self.read_dicom(image_name, equalize),
                       'png' : self.read_cv2(image_name, color_option)}
            
            try:
                #if image_type == 'dcm':
                    #self.read_dicom(image_name, equalize)
                #print("Using image reader:", image_type, readers[image_type])
                reader_class = readers[image_type]
                reader_class()
                
            except KeyError:
                # I don't want to kill a long process, so warn and move on. 
                print("Warning: Cannot find image reader for extension:", image_type)
                print("Supported image types are:", readers.keys())
                
                
    @make_callable        
    @trace_pipeline
    def read_dicom(self, file, equalize=False):
        '''should look like this in practice:
        
        #!pip install pydicom # if needed to install pydicom
        
        import pydicom 
        read_dicom(file)
        
        also import numpy as np
        
        
        '''
        # Image decoding methods modeled after:
        # https://www.kaggle.com/havinath/eda-observations-visualizations-pytorch
        
        
        
        try:
            di_img = pydicom.dcmread(file)
            img =di_img.pixel_array
            
            # DICOM uses some odd values to store pixel values. They can range into negative and positive values ranging wildly. 
            # To map these pixel values to a suitable range (0-255)  first determine the range of values in the image ( max_v + min_v)
            # Then determine the a ratio to convert from the pixel ranges to 0-255. 
            # Also note, that before converting with the ratio, the minimum value should be added to all pixel locations. 

            if equalize:
                img = exposure.equalize_hist(img)
            
            dims = img.shape
            ret_img = np.zeros((*dims, 3), dtype=np.float)
            ret_img[:,:,0] = img
            ret_img[:,:,1] = img
            ret_img[:,:,2] = img

            self._img = ret_img
            
        except ZeroDivisionError:
            pass
            print("Division by Zero Error. No Image returned")
            return 0
        
    @make_callable
    @trace_pipeline
    def read_cv2(self, file, color_option=None):
        
        if color_option:
            img = cv2.imread(file, color_option)
        else:
            img = cv2.imread(file)
            
        self._img = img
    
    
    @make_callable
    @trace_pipeline
    def convert_img_color(self, color_option):
        self._img = cv2.cvtColor(self._img, color_option)
    
    
    @make_callable
    @trace_pipeline
    def write_cv2(self, save_name_as=None,  extension=".png",):
        
        image_out_filename = os.path.join(self.dataset_output_path, str(self._active_id)) + extension
        
        if save_name_as:
            
            col_number = self.df.columns.get_loc(save_name_as)
            
            self.df.iat[self._active_id, col_number] = image_out_filename
        
        max_ = np.max(self._img)
        print("writting %s " % image_out_filename, max_)
        cv2.imwrite(image_out_filename, self._img)
    
    @make_callable
    @trace_pipeline
    def convert_field_to_image(self, field, desired_shape=(10,10)):
        
        source_data_field_number = self.df.columns.get_loc(field)    
        raw_image = self.df.iat[self._active_id, source_data_field_number]
        
        self._img = self.convert_array_to_image(raw_image, desired_shape)
        
        
    def convert_array_to_image(self, source_array, desired_shape=(10,10)):

        """Arctic operates best with RGB images as arctic._img, so gray scale images will be produced by coping the gray data to 
        all three color channels. """
        #unit_test1:
        #this = convert_array_to_image(np.array([[0,100,0,0],[0,1,1,0],[1,0,0,1]]), (2,2,3))
        #unit_test2:
        #this = convert_array_to_image(np.array([1,0,0,1]), (2,2))
        #unit_test3:
        #this = convert_array_to_image([1,0,0,1], (2,2))

        # Convert list to numpy first, if there is a problem with the conversion, numpy should warn, otherwise, and data type that can be
        # used in numpy is now compatible with this function.
        source_array = np.array(source_array)

        check_len = source_array.shape

        img = np.reshape(source_array, desired_shape)

        return img
    
    ########################################################### DATALOADERS ETC    


    def _init_learn(self, y_label_column, feature_group_dict, network, loss_function, learning_rate=0.003, data_prewash="Normalize"):

        # So feature_group_hash will have keys of the feature group names, with values that are an 
        # array of names of the pandas df frame that those features should be pulled from. 
        # If you wanted to pull say ["age", "Weight", "Height", "numberToes"] into a single stage NN, just use:
        # arctic._init_learn(y_label_column="label", feature_group_dict={"features":  ["age", "Weight", "Height", "numberToes"]})
        # For multi-stage NNs, use:
        # arctic._init_learn(y_label_column="label", feature_group_dict={"pplfeatures":  ["age", "Weight", "Height", "numberToes"],
        #                                                                "tractorfeatures": ["enginesize", "framewidth"],
        #                                                                "farmfeatures": ["acres", "numcows"]})

        # feature_group_dict is then feed into the specified network like this:
        # The net function passed to arctic should be prepared to deal with those parameters. 
        
        self._loss_function = loss_function
        self._network = network
        self._set_default_optimizer_and_learning_rate(learning_rate=learning_rate)

        print(self, "setting network to", self._network)

        # y_label_column is the column of data to pull the y_labels from

        self._y_column = y_label_column

        # feature_group_hash lists all of the feature groups used. While normal networks would only take one 
        # set of features, complex networks like the ones I'd like to use will require multi-step networks



        # self._net_expects will be filled in with the feature group names. 
        if isinstance(feature_group_dict, dict): 
            keys_ = feature_group_dict.keys()
            self._net_expects = keys_
            self._feature_groups = feature_group_dict

        else:
            print("_init_learn(feature_group_dict) should be a dictionary, not", type(feature_group_dict))

        # use data_prewash to generate a new dataframe with pre-washed content. Store it as train_pw.csv and 
        # test_pw.csv. 

        # data_prewash should be either Normalize, Standardize or None
        # TODO data_prewash
        # self._prewash_dataframe() 
        
        # self.data_mode is set to the bucket name representing the location of that dataframe.
        # When this is set, it sets self.df to self._df_train 
        # then when the dataloader is constructed, it contains data from self._df_train rather then self._df_test.
        # Note, this data_mode is used in lots of other locations to simplify and abstract methods. 
            
        self.data_mode = "train"
        self._train_loader = DataLoader(self, batch_size=self._batch_size, shuffle=self._shuffle, num_workers=self._num_workers, pin_memory=self._pin_memory)
        
        self.data_mode = "test"
        self._test_loader = DataLoader(self, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)

        self.data_mode = "val"
        self._val_loader = DataLoader(self, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)

        self.data_mode = "all"
        self._all_loader = DataLoader(self, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory)
        

    @make_callable
    def __showitem__(self, index, number_of_items_to_show=10):
        """ a replacement for __getitem__ when the desire to see what is returned is needed. Calls __getitem__ and prints the results
        Note, __getitem__ doesn't have the @make_callable decorator. """
        
        if self._showitem_iterator < number_of_items_to_show:

            x, y = self.__getitem__(index)
            print("Item #", self._showitem_iterator, x,y)
            self._showitem_iterator += 1
        
    
    def __getitem__ (self, index):
        
        # Must return Y even on validation sets, so force a default value. 
        
        #print("Getting index:", index)
        self._active_id = index
        self._active_row = self.get_current_record()

        # For each item in self._net_expects, build a feature group with that name. 

        # reset the data for this item. 
        self._x = {}
        self._y = ''

        for group_name, group_members in self._feature_groups.items():

            # A place to store the features for this feature group. 
            # feature groups is a way to return multiple chunks of features to submit 
            # to a NN, rather then submitting all the features at once. 
            # There are a number of reasons why someone might want to do things this way.... 
            # Including making a NN smaller or more complex, or both. 

            # or you could just use one feature group to build a non-fancy little network. 

            current_feature_group = []

            for field_name in group_members:

                # If the feature group name is _img_from_file, than import the image from a file. 
                if field_name == '_img_from_file':

                    #self._img_prep_pipeline = []

                    # TODO get rid or equalize, or encapsulate it better. 
                    # store image is self._img
                    self.read_in_img(image_field=self.image_field, equalize=True)
                    
                    # Run the image through the defined (or not) pipeline:
                    # there is no magic here, all of the functions work on self._img. 
                    #print("this should process the images")
                    self.process_image()
                    
                    #TODO When image is in, use the transformation pipeline from self._img_prep_pipeline to prepare it.
                    # for now, flesh out the rough structure of this thing. 

                    field_item = torch.tensor(self._img).permute((2, 0, 1)).float().div(255)
                    field_item = self.normalize_function(field_item).to(self._device)
                    #print("Image", field_item)

                elif field_name in self.df.columns:            
                    field_num = self.df.columns.get_loc(field_name)  
                    field_item = self.df.iat[self._active_id, field_num]
                    # TODO optimize the datatype used below. this could waste memory. 
                    field_item = torch.tensor(field_item).float().to(self._device)
                    
                else:                
                    print("Cannot find column name ", field_name, "in the arctic dataframe")
                    print("Check that your feature_group_dict specified in init_learn() is correct")
                    # Raise a custom error. 
                    return 0

                current_feature_group.append(field_item)

            self._x[group_name] = current_feature_group

            # Duude, later your going to wish you did the same fancy footwork with the y as you did with x above.
            # only time will tell, but I bet you re-write this later.... 

            if self._y_column in self.df.columns:
                # Get the label for this record from the column specified in self._y_column
                y_col_number = self.df.columns.get_loc(self._y_column)
                self._y = torch.tensor(self.df.iat[self._active_id, y_col_number]).float().to(self._device)

            else:
                print("Cannot field the column name specified for Y values in the arctic dataframe")
                print("The field specified is", self._y_column, "and the avialable columns in the dataframe")
                print("are", self.df.columns)

                #raise a custom error 
                return 0        

        return self._x, self._y
    
    def __len__(self):
        
        return len(self.df)
    
    def _eval_fit(self):

        valid_loss = 0
        self.data_mode = "test"
        
        print("Eval Mode")
        
        # This might be interesting, as arctic (self) will actually be the data_loader...... 
        with torch.no_grad():
            for batch_idx, data in enumerate(self._test_loader):
                
                first_key_name = list(data[0].keys())[0]
                
                # Now find the length of the array of items in data[0][first_key_name]
                # this will be the batch size. 
                eval_batch_size = len(data[0][first_key_name])
                
                #print("Batch Index:", batch_idx, "shape", eval_batch_size)
                
                # at this point x_multi is an array of hashs, which keys point to featurs, that contain arrays....
                # x_multi -> [ index0, index1, index2]
                # index0 -> { 'named_feature_group1', 'named_feature_group2', 'numberofgearsinengine'}
                # named_feature_group1 -> [ 8, 32, 64]  # this example shows non normalized items... 
                
                x_multi = data[0]
                y_true = data[1].long()
          
                y_pred = self._network(x=x_multi)
                
                y_shape = y_pred.shape[0]
                #y_pred = y_pred.squeeze()#.argmax(axis=1)  
                _, predicted = torch.max(y_pred.data,1)

                #print("true", y_true, "pred", predicted, y_pred)

                valid_loss += self._loss_function(y_pred, y_true).data.item() * y_shape

        valid_loss /= len(self._test_loader)
        
        return valid_loss
    
    def fit (self, epochs=1):
         
        global history, iteration, epochs_done, lr
        
        _loss = 0
        losses = []
        val_losses = []

        self._network.train()
        self.data_mode = "train"
        
        pbar = tqdm.tqdm(total=len(self._train_loader))
        
        for epoch in range(epochs):

            total_examples = 0

            # arctic is the dataloader.... 
            for batch_idx, data in enumerate(self._train_loader):

                self._optimizer.zero_grad()

                # at this point x_multi is an array of hashs, which keys point to featurs, that contain arrays....
                # x_multi -> [ index0, index1, index2]
                # index0 -> { 'named_feature_group1', 'named_feature_group2', 'numberofgearsinengine'}
                # named_feature_group1 -> [ 8, 32, 64]  # this example shows non normalized items... 

                x_multi = data[0]
                y_true = data[1].long()

                y_pred = self._network(x=x_multi)

                y_pred = y_pred.squeeze()
                #y_true = y_true.squeeze()

                #print("prediction:", y_pred,"Actual:", y_true)
                cur_loss = self._loss_function(y_pred, y_true)
                cur_loss.backward()
                
                self._optimizer.step()      

                _loss += cur_loss.item()

                pbar.update(1)

            pbar.reset()
            _loss /= len(self._train_loader)

            pbar.write("Epoch: %3d, train BCE: %.4f" % (epoch, _loss))
            #print("Epoch: %3d, train BCE: %.4f" % (epoch, _loss))

            val_loss = self._eval_fit()
            
            self.data_mode = "train"
            
            pbar.write("              val BCE: %.4f" % (val_loss))
            #print("              val BCE: %.4f" % (val_loss))

            
    def _set_default_optimizer_and_learning_rate(self, learning_rate=0.003):
        """ arctic will use this optimizer by default."""

        optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        #[
           # {'params' : net.layer4.parameters(), 'lr' : lr /3},
           # {'params' : net.layer3.parameters(), 'lr' : lr /9},
           # ],
        self._optimizer = optimizer


    def find_lr(self, init_value=1e-9, final_value=.0003):
              
        number_in_epoch = len(self._train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch )
        
        learning_rate = init_value
        self._set_default_optimizer_and_learning_rate(learning_rate)
        
        losses = []
        log_lrs = []
        rates = []
        batch_num = 0
        best_loss = 0.0

        self._network.train(True)

        # arctic is the dataloader.... 
        self.data_mode = "train"
        for batch_idx, data in enumerate(self._train_loader):
            
            batch_num += 1 
            self._optimizer.zero_grad()
            
            # Just looking for the length of the data here. Since we don't know in advance the name of the feature groups, this mess is used
            # to find the first key name from data[0]
            first_key_name = list(data[0].keys())[0]
                
            # Now find the length of the array of items in data[0][first_key_name]
            # this will be the batch size. 
            flr_batch_size = len(data[0][first_key_name])
        
            
            x_multi = data[0]
            y_true = data[1]
          
            y_pred = self._network(x=x_multi)
            y_pred = y_pred.squeeze()
            
            cur_loss = self._loss_function(y_pred, y_true)
            cur_loss.backward()   
            
            self._optimizer.step()            

            # Update the lr for the next step and store
            losses.append(cur_loss)
            log_lrs.append(math.log10(learning_rate))
            rates.append(learning_rate)

            # increase the learning rate:
            learning_rate *= update_step
            self._set_default_optimizer_and_learning_rate(learning_rate=learning_rate)

            if batch_num > 1 and cur_loss > 4 * best_loss:
                return log_lrs, losses, rates
            
            if cur_loss < best_loss or batch_num == 1:
                best_loss = cur_loss

        return log_lrs, losses, rates

    def predict(self, bucket, number_batches):
        
        self._network.eval()
        
        shown_batches = 0
        
        
        # Predict on the entire training, test and validation set. 
        if bucket == "all":
            loader = self._all_loader
            self.data_mode = "all"
        # predict on the cross validation set only, default, as it's a better approximation. 
        else:
            loader = self._val_loader
            self.data_mode = "val"

        print("Predict Mode:", bucket)
        
       
        # This might be interesting, as arctic (self) will actually be the data_loader...... 
        for batch_idx, data in enumerate(loader):

            with torch.no_grad():

                # at this point x_multi is an array of hashs, which keys point to featurs, that contain arrays....
                # x_multi -> [ index0, index1, index2]
                # index0 -> { 'named_feature_group1', 'named_feature_group2', 'numberofgearsinengine'}
                # named_feature_group1 -> [ 8, 32, 64]  # this example shows non normalized items... 
                
                x_multi = data[0]
                y_true = data[1].long()
          
                y_pred = self._network(x=x_multi)
                
                y_shape = y_pred.shape[0]
                y_pred = y_pred.squeeze().argmax(axis=1)  
                print("Actual Value:", y_true, "Prediction:", y_pred)
                
                shown_batches += 1
                
                if shown_batches == number_batches:
                    break

         