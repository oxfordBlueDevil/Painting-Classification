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
	#os.mkdir('scraped-images')
	os.chdir('scraped-images')
	os.mkdir(artist)
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
	artists = ['VanGogh', 'Durer', 'JosephMallordTurner', 'Monet-and-Manet', 'Raphael',
				'VanGogh2', 'Hudson-River-Artists', 'Klimt-and-Expressionism', 'Botticelli',
				'Renoir', 'Bouguereau', 'Monet', 'Tissot', 'Cezanne', 'Cubism']

	artistURLs = ['https://picasaweb.google.com/106069219035575195726/VanGoghVolumeTwo',
		'https://picasaweb.google.com/106069219035575195726/Durer?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/JosephMallordTurner?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/MonetAndManet',
		'https://picasaweb.google.com/106069219035575195726/Raphael?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/VanGoghImages?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/HudsonRiverArtists?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/KlimtAndExpressionism?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/Botticelli?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/Renoir?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/Bouguereau?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/BonusMonetImages?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/Tissot?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/Cezanne?noredirect=1',
		'https://picasaweb.google.com/106069219035575195726/AbstractArtCubism?noredirect=1']

	d = {artist: url for (artist, url) in zip(artists, artistURLs)}

	# for artist in artists[5:]:
	# 	scraping(artist, d[artist])

	scraping(artists[-1], d[artists[-1]])

if __name__ == '__main__':
	main()