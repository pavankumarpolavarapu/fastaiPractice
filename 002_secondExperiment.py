from fastai.vision.all import *
from fastai.vision.widgets import *
from fastbook import *
import os
import matplotlib.pyplot as plt


def main():
    # You need Bing Search API Key
    key = os.environ.get('AZURE_SEARCH_KEY')

    # No Key, No Go
    assert key

    # Download Bears
    bear_types = 'grizzly', 'black', 'teddy'
    path = Path('bears')

    if not path.exists():
        path.mkdir()

        for o in bear_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)

            #results will contain a set of 150 urls
            results = search_images_bing(key, f'{o} bear')

            #download_images will use threadpool to download images in parallel and there is lot going on, not completely readable atleast for me
            #results is a fastcore.foundation.L - List basically and it has method attrgot that returns array of key values in a dict list
            download_images(dest, urls=results.attrgot('contentUrl'))

    #get filenames in path, returns L
    fns = get_image_files(path)

    #call verify_image in parallel with Image.load method to see if downloaded images are opening and returns failed to load list
    #parallel uses the map method which returns the list in sequence
    failed = verify_images(fns)

    #delete images that cannot be loaded
    failed.map(Path.unlink)
    #takes Independent and Dependent variables - in our case Input Image and Category of Image
    #way to load the images
    #defines validation set
    #defines the way to get the dependent variable
    #item transformation
    bears = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label, item_tfms=Resize(128))
    dls = bears.dataloaders(path)
    # Item transformation without parameter is going to crop the image and sometimes important part is discarded so you can do different way of transformation
    # bears.new creates a new datablock with a different transformation
    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
    # bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
    bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
    dls = bears.dataloaders(path)

    # dls.valid.show_batch(max_n=8, nrows=2, unique=True)
    # plt.show()

    #note that all the values passed are objects
    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    # ClassificationInterpretation is from fastai/interpret.py where it takes from_learner as input to compute interpretations
    # plot_confusion_matrix, confusion_matrix, most_confused, print_classfication_report, top_losses are some of its useful methods
    interpretation = ClassificationInterpretation.from_learner(learn)

    # Confusion Matrix
    interpretation.plot_confusion_matrix()
    plt.show()

    interpretation.plot_top_losses(3)
    plt.show()

    # works only in jupyter notebook
    # cleaner = ImageClassifierCleaner(learn)
    # cleaner

    # Exports the learning model into bearclassifier.pkl file
    learn.export(fname="bearclassifier.pkl")

    # Sample way to import the lerned model
    # path = Path('bearclassifier.pkl')
    # assert path.exists()
    # learn_inf = load_learner(path)
    # learn_inf.predict('bears/grizzly/00000110.jpg')



if __name__ == '__main__':
    SystemExit(main())
