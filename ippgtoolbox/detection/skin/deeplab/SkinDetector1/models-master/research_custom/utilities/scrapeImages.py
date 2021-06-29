from bs4 import BeautifulSoup
import requests
import re
import urllib.request, urllib.error, urllib.parse
import os
import argparse
import sys
import json



# adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search

def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'html.parser')

def main(args):
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-s', '--search', default='bananas', type=str, help='search term')
    parser.add_argument('-n', '--num_images', default=10, type=int, help='num images to save')
    parser.add_argument('-d', '--directory', default='/Users/gene/Downloads/', type=str, help='save directory')
    parser.add_argument('-f','--filter', default='isz:m,itp:photo,ic:trans,ift:png', type=str,help='filter string from google advanced search tbs param')
    args = parser.parse_args()
    query = args.search#raw_input(args.search)
    filters = args.filter
    max_images = args.num_images
    save_directory = args.directory
    image_type="Action"
    query= query.split()
    query='+'.join(query)
    url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch&tbs="+filters
    #header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36",'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    soup = get_soup(url,header)
    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))
    for i , (img , Type) in enumerate( ActualImages[0:max_images]):
        try:
            req = urllib.request.Request(img, headers=header)
            #req = img
            raw_img = urllib.request.urlopen(req).read()
            if len(Type)==0:
                f = open(os.path.join(save_directory , "img" + "_"+ str(i)+".png"), 'wb')
            else :
                f = open(os.path.join(save_directory , "img" + "_"+ str(i)+"."+Type), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print(("could not load : "+img))
            print(e)

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()