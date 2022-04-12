# Perlin Noise Camouflage Algorithm 
This algorithm provides the means of generatitng camofulage patterns based on the Perlin fractal noise from an input image.
The `main.py` file contains a scratch of the project in functional programming and it's deprecated, please use the `objectmaker.py` script.
## Usage 
In order to properly use the algorithm the `camo` class must be instantiated with the path to the input image.
The input image should be in the `./data/` directory.
### Color extraction 
In order to extract the colors from the input image the `extract_colors(self, n, show=False, save=True, load=False, colspace="HSV")` method must be called.
`n` is the number of colors to extract; the defult color space is HSV however RGB is also supported but yileds worst results.
### Pattern generation
In order to generate a proper pattern you should use the  `layer_perlin(self, res=(16, 8), spotting=[], test=False)` method.
The `res` parameters is a tuple of two integers which represent the periods of the noise to be generated along the two dimentional axes.
The `spotting` parameter is a list used to set the parameters for addittion of spotting to the pattern, it accepts values in the form of  `[number of colors(int), fraction(float)]`, if empty no spotting will be performed.

Please note that the only proper method to generate patterns is `layer_perlin`; al the other methods, including optimization ones yiled suboptimal results and included just for the sake of clarity and compleatness.


## Demo
Can be runned dowloading the repo, all files are included.

```

    #demo with large forest image
    forest = camo("data/forest.jpg")
    #first we have to run the color extractor
    forest.extract_colors(n=3, colspace="HSV", show=True, load = True)
    #now we run the pattern designer asking it to apply spotting the 10%(0.1) of the surface using the two least occurring colors
    forest.layer_perlin(spotting=[2, 0.1], res=(4, 4))
    forest.show()
```
