# Fast AI has divided its library into catogories
# noticed vision, text, data, medical, tabular and they have `all` to import everything
from fastai.vision.all import *

# Matplot lib for plotting images
import matplotlib.pyplot as plt

# Labeling function used for returning labels
# In case of Cats vs Dogs, Name of files stating with uppercase are Cats
def label_func(f): return f[0].isupper()

def main():
        # URLs is a class from data/external.py file where all the data is tar zipped
        # and untar_data is from the same file that returns Path object of extracted data
        path = untar_data(URLs.PETS)
        print('Path contains {}'.format(path.ls()))
        print('Type for path is {}'.format(type(path)))

        # get_image_files is a call to function in data/transforms.py for enumerating files in path
        files = get_image_files(path/"images")
        print('Total number of files are {}'.format(len(files)))

        # ImageDataLoaders is a data loader for images and defined in vision/data.py
        # It takes path, file names and a labelling function, in our case we are passing transformation as well
        # item_tfms is an item transformation in .from_path_func and it has valid_pct as well
        # Resize is from vision/augment.py and takes size and method as input and default method is crop
        dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

        # Returns the AxesSubplots
        dls.show_batch()

        #install pyqt5 using pip to show the plots
        plt.show()

        # Model definition
        # cnn_learner defined in vision/learner.py takes many input parameters
        # We are providing it with data loader, architecture and metrics
        # Return object is a Learner object from learner.py which also includes model
        learn = cnn_learner(dls, resnet34, metrics=error_rate)

        # fine_tune is a callback method of Learner object and takes epochs as input
        learn.fine_tune(1)

        uploader = SimpleNamespace(data = [path/'images/Egyptian_Mau_161.jpg'])
        img = PILImage.create(uploader.data[0])
        img.show()
        plt.show()

        #Now predicting the above image
        is_cat, _, probs = learn.predict(img)
        print(f"Is this a cat?: {is_cat}.")
        print(f"Probability it's a cat: {probs[1].item():.6f}")

if __name__ == '__main__':
    exit(main())



