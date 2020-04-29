'''
Only support python 2!
Get images of 200 classes (as defined in ImageNet-A paper) from flickr.
'''

import os, json, sys, urllib
sys.path.append(os.path.join(os.path.expanduser('~'), 'MAD'))
from common_flags import COMMON_FLAGS

mindate='2019-01-01'
maxdate='2019-08-01' 
download_folder = COMMON_FLAGS.data_download_dir + '-%s-%s' % (mindate,maxdate)
if not os.path.isdir(download_folder):
    os.mkdir(download_folder)
# folder in which the images will be stored
url_folder = "imgs"

def download_imgs():
    with open(os.path.join(COMMON_FLAGS.json_dir, 'selected_keywords.json'), 'r') as fp:
        keywords = json.load(fp)
    print("keywords:", type(keywords), len(keywords), type(keywords[0]))

    api_keys = {'flickr': ('3845aa5608781b176e74bedd2a653b78', '19192eb5251a4809')} # replace XXX.. and YYY.. by your own keys
    # images_nbr = 10000 # number of images to fetch
    images_nbr = 200 # 200 * 200 = 40k

    ### Crawl and download images ###
    from web_crawler import WebCrawler
    crawler = WebCrawler(api_keys, mindate=mindate, maxdate=maxdate)

    # 1. Crawl the web and collect URLs:
    crawler.collect_links_from_web(keywords, images_nbr, remove_duplicated_links=True)

    # 2. (alernative to the previous line) Load URLs from a file instead of the web:
    #crawler.load_urls(download_folder + "/links.txt")
    #crawler.load_urls_from_json(download_folder + "/links.json")

    # 3. Save URLs to download them later (optional):
    # crawler.save_urls(os.path.join(download_folder, "links.txt"))
    crawler.save_urls_to_json(os.path.join(url_folder, "links-%s-%s.json" % (mindate,maxdate)))

def download_from_json(filename, target_folder, start_progress=0):
    # 
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    # 
    with open(filename) as links_file:
        images_links = json.load(links_file)
    #
    print("Downloading files...")
    progress = 0
    failed_links = []
    images_nbr = sum([len(images_links[key]) for key in images_links])
    for keyword, links in images_links.items():
        if not os.path.isdir(os.path.join(target_folder, keyword)):
            os.mkdir(os.path.join(target_folder, keyword))
        for link in links:
            if progress >= start_progress:
                target_file = target_folder + '/' + keyword + '/' + link.split('/')[-1]
                try:
                    f = urllib.URLopener()
                    f.retrieve(link, target_file)
                except IOError:
                    failed_links.append(link)

            progress = progress + 1
            if progress >= start_progress:
                print("\r >> Download progress: %d/%d" % (progress, images_nbr))
                sys.stdout.flush()

    print("\r >> Download progress: ", (progress * 100 / images_nbr), "%")
    print(" >> ", (progress - len(failed_links)), " images downloaded")

    # save failed links:
    if len(failed_links):
        f2 = open(url_folder + "/failed_list.txt", 'w')
        for link in failed_links:
            f2.write(link + "\n")
        print(" >> Failed to download ", len(failed_links),
                " images: access not granted ",
                "(links saved to: '", url_folder, "/failed_list.txt')")


if __name__ == '__main__':
    # download_imgs()
    download_from_json(os.path.join('imgs', "links-%s-%s.json" % (mindate,maxdate)), 
        download_folder, start_progress=15644)
