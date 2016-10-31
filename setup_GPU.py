import os
import shutil

def setup_theano():
	destfile = "/home/kel/.theanorc"
	open(destfile, 'a').close()
	shutil.copyfile("/mnt/.theanorc", destfile)

	print "Finished setting up Theano"