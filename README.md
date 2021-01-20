# diploma-project
Estimation of car position on the road

## Table of contents
* [General info](#general-info)
* [Readme file](#readme-file)
* [edgefilter](#edgefilter-function)
* [Color reduction](#color-reduction)
* [Other functions](#other-functions)
* [Classes](#classes)
* [Additional info](#additional-info)
* [Contact](#contact)

## General info
The lane_navigator.py file is the code that I wrote for my diploma thesis. The title of my thesis is 
"Estimation of car position on the road". The aim of this project is to create a method for finding the lines
of a road and estimating where the car is on it. It is a simple but robust approach for camera-based lane 
recognition, based on digital image processing.
Specifically, for every frame of a video this method creates a binary image where ideally only the lines of the
road are representing with white and the rest image with black.
For the pursose of this project, I used Python (version 3.7.1) and among others the libraries OpenCV, NumPy,
sckikit-learn, TkInter.

As part of my diploma, I created an executable, which I did not included in this repository.
In this repository, I uploaded only the python file where I have included all the code for this project.

## Readme file
In this readme file, it is impossible to be written a detailed report of every bit of code that I wrote and 
the algorithms that I used or developed, so I included some info so that a reader can understand the flow of 
the program.

## edgefilter function
This function performs Edge preserving smoothing in color images.
More info: https://www.mathworks.com/matlabcentral/fileexchange/27988-edge-preserving-smoothing?s_tid=FX_rc2_behav

## Color reduction
The function reducecolors performs color reduction. In particular, it uses the inner colours of the objects 
(using Sobel's edges) as samples and performs Kmean colour reduction and Mean shift colour reduction afterwards.

## Other functions
make_mask: This function defines the area of interest in the frame. It makes a trapezoid shape where the pixels 
have their normal values, and outside of it the pixels are set to black.

kmean: K Mean colour reduction.

find_edges: This function performs the binarization of the image by finding the edges. Specifically, it covers 
two cases: finding the edges for first time (e.g. in the very first frame of the video) or the lines of the road 
have been found from a previous frame and it just searching around the previous known location for the lines.
In here reducecolors it is called, and the result is combined with Canny edges.

make_coordinates: This function 'cleans' the coordinates that have been collected for the lines by fitting 
these points into a line.

find_lines: This function is used like a controller for the whole programm, and actually returns the coordinates 
of the lines of the road.

checker: This function checks the results and decides to print or not black lines above the lines of the road.

calc_dist: This function calculates the distance of the car from the right line of the road whenever it is 
possible.

## Classes
class App: creates the starting menu
class Player: creates a player where the final video it is played in real time. 
(By real time I mean that the program processes and plays the video frame by frame)
class MyVideoCapture: This class is responsible for getting every frame of the video, 
calling the right functions and returning the processed frame back to the player.

## Addiitional info
Additional information regarding the diploma thesis (regulations, copyrights etc) of the department 
that I graduated from, can be seen here: https://www.ee.duth.gr/en/studies/undergraduate/diploma-thesis/

## Contact
Created by [@kostaslafa]
email: kostaslafa14@gmail.com
