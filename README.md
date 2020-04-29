Before running the code, make sure that the folder is named as `MAD`. 
Also modify two variables `hdd_dir` and `ImageNet_dir` in `common_flags.py` to valid paths on your own machine.


## Crawl images from internet
Run `python images-web-crawler/sample_ours.py` under project root dir, or run `python sample_ours.py` under `images-web-crawler`.  
Most files under `images-web-crawler` comes from this [repo](https://github.com/amineHorseman/images-web-crawler). Thanks to the authors for their great work!
Note that `images-web-crawler` only support Python2, while the rest of this repo support Python3.

## Arrange the crawled images and construct them as a dataset
Run `python dataset/construct_dataset.py` under project root dir, or run `python construct_dataset.py` under `dataset`.  

## Get WordNet tree hierarchy
First run `python save_jsons.py` under `class_info`. 
Then run `python construct_path.py` under project root dir.
These two commands will construct and save multiple `.json`, `.csv` and `.txt` files, which mainly aims at showing the WordNet tree hierarchy.
All these files are already uploaded in this repo, so you can skip this step if you want.
For example, `code_with_imgnet_id_readable.json` shows the paths from WordNet root node to each ImageNet class leaf node.
Take the item with key="872" in `code_with_imgnet_id_readable.json` as an example, this item shows that the path from root node to leaf node "tripod", which has id=872 in ImageNet is: 
"tripod; rack, stand; support; device; instrumentality, instrumentation; artifact, artefact; whole, unit; object, physical object; physical entity; entity".

## Get predictions on unlabled dataset
Run `python test_our_data.py` to do prediction on the web-scale unlabeled dataset.

## Select MAD images
Run `python compare_and_select.py` to select MAD images.

## Get global ranking
After providing human annotations on all selected MAD images, run `python global_ranking.py` to get global ranking of all classifiers.
