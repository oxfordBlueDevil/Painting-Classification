import requests
from bs4 import BeautifulSoup
import urllib
import os

def remove_digits(url):
	ind = str(url).rfind('s800')
	filename = url[ind+5:]
	while filename.find('%') >= 0:
		percent_index = filename.index('%')
		filename = filename[:percent_index] + '-' + filename[percent_index + 7:]
	return filename

def scrapeImagesFromURL(artist, imageURLs):
	os.chdir('scraped-images')
	os.chdir(artist)
	for imageURL in imageURLs:
		filename = remove_digits(imageURL)
		urllib.urlretrieve(imageURL, filename)
	os.chdir('..')
	os.chdir('..')

def scraping(artist, artistURL):
	response = requests.get(artistURL)
	content = response.content

	soup = BeautifulSoup(content, 'html.parser')
	table_icons = soup.find_all('table', {'id': "lhid_tablecontent"})
	icon_a_class_data = table_icons[0].find_all('a')
	# images = icon_a_class_data[1::2][:-1]
	# imageURLS = [image['href'] for image in images

	images = table_icons[0].find_all('img')[:-2]
	imageURLs = [image['src'].replace('s128','s800') for image in images]

	scrapeImagesFromURL(artist, imageURLs)

def main():
	artists = ['VanGogh', 'Durer', 'JosephMallordTurner']

	artistURLs = ['https://picasaweb.google.com/106069219035575195726/VanGoghVolumeTwo',
		'https://picasaweb.google.com/106069219035575195726/Durer?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/JosephMallordTurner?noredirect=1']

	d = {artist: url for (artist, url) in zip(artists, artistURLs)}

	for artist in artists[:2]:
		scraping(artist, d[artist])

if __name__ == '__main__':
	main()