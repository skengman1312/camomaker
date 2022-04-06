# Perlin Noise Camouflage Algorithm 
This algorithm provides the means of generatitng camofulage patterns based on the Perlin fractal noise from an input image.
## Usage 
In order to properly use the algorithm the `camo` class must be instantiated with the path to the input image.
The input image should be in the `./data/` directory.
### Color extraction 
In order to extract the colors from the input image the `extract_colors(self, n, show=False, save=True, load=False, colspace="HSV")` must be called.
`n` is the number of colors to extract; the defult color space is HSV however RGB is also supported but yileds worst results.
### Pattern generation
In order to generate a proper pattern you should use the  `layer_perlin(self, octaves=(16, 8), spotting=[], test=False)` method.
The `octaves` parameters is a tuple of two integers which represent the periods of the noise to be generated along the two dimentional axes.
The `spotting` parameter is a list used to set the parameters for addittion of spotting to the pattern, it accepts values in the form of  `[number of colors(int), fraction(float)]`, if empty no spotting will be performed.

Please note that the only proper method to generate patterns is `layer_perlin`; al the other methods, including optimization ones yiled suboptimal results and included just for the sake of clarity and compleatness.
