from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import numpy as np
from perlin_numpy import generate_fractal_noise_2d, generate_fractal_noise_3d
from scipy.stats import norm
from sewar import uqi
from colorsys import hsv_to_rgb
import random
import os

# simple util lambda
normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))


# we define a class, each pattern will be an instance of the camo class initialized with different input
# images
class camo:

    def __init__(self, image):
        self.image = image

        self.name = image[5:].split(".")[0]

        # color extraction phase

    def extract_colors(self, n, show=False, save=True, load=False, colspace="HSV"):
        # since color extraction is relatively computationally intensive process save/load features have been
        # implemented for testing and debugging, they rely on the numpy save function

        if load:
            data = np.load(f"tmp/{self.name}-c{n}.npz")
            self.colors = data["colors"]
            self.colspace = data["colspace"]
            self.perc = data["perc"]
            self.palette = data["palette"]
        else:

            # the color extraction step involves performing k-means clustering as described in the report

            img = Image.open(self.image).convert(colspace)
            imgdat = np.asarray(img, dtype=float)
            dim = imgdat.shape[0] * imgdat.shape[1]
            imgdat = imgdat.reshape(1, dim, 3)
            imgdat = np.squeeze(imgdat, axis=0)
            book2 = KMeans(n_clusters=n, random_state=1312).fit(imgdat)

            counts = np.unique(book2.labels_, return_counts=True)
            perc = [c / len(book2.labels_) for c in counts[1]]

            colors = book2.cluster_centers_.astype(np.uint8)

            # HSV extraction needs an extra step to convert the value of extracted colors back to RGB in order to
            # avoid downstream conversion and standardize the method's output

            if colspace == "HSV":
                # pill uses 8-bit rapresentation, we need to rescale the values in range 0 - 1
                pill_to_hsv = lambda x: [x[0] / 255, x[1] / 255, x[2] / 255]
                cc = np.apply_along_axis(pill_to_hsv, 1, colors)
                colors = np.asarray([[int(z * 256) for z in y] for y in [hsv_to_rgb(*x) for x in cc]])

            palette = [np.full((200, 200, 3), col, dtype=np.uint8) for col in colors]
            palette = np.concatenate(palette[0:n])

            self.colors = colors
            self.colspace = colspace
            self.perc = perc
            self.palette = palette

        # if show option is true extracted colors will be displayed on the screen in a palette format

        if show:
            pim = Image.fromarray(self.palette, mode="RGB")
            pim.show()

        # saving utility
        if save and not load:
            if not os.path.isdir("tmp"):
                print("SAVE")
                os.mkdir("tmp")
            np.savez(f"tmp/{self.name}-c{n}", colors=colors, colspace=colspace, perc=perc, palette=palette)

            return colors, perc, palette

    # utility method used to display the palette in a second moment
    def show_palette(self):
        im = Image.fromarray(self.palette, mode=self.colspace)
        im.show()

    # pattern generating function using a single layer of 2D perlin fractal noise, more in the report
    def make_perlin(self, octaves=(16, 8), colspace="RGB", test=False, ms=0):
        np.random.seed(0)
        noise = generate_fractal_noise_2d((1024 * 2, 1024 * 2), octaves, 6, tileable=(True, True))
        noise = normalize(noise)
        mean, std = norm.fit(noise)
        if ms:  # multipatch spotting support
            p = [(n / sum(self.perc[:1:-1])) * ms for n in self.perc[:1:-1]] + [m * (1 - ms) for m in self.perc]
            tc = 1  # n of times to expand the color space
        else:
            p = self.perc
            tc = 0
        q = np.cumsum(p)
        mean, std = norm.fit(noise)
        q1 = norm.ppf(q=q, loc=mean, scale=std)
        q1[q1 == np.inf] = 1
        colordict = dict(zip(q1, self.colors.tolist()[
                                 :1:-1] * tc + self.colors.tolist()))
        colimage = np.zeros(noise.shape + (3,))
        for c in sorted(colordict.items(), reverse=True):
            for i in range(noise.shape[0]):
                for j in range(noise.shape[1]):
                    if noise[i][j] <= c[0]:
                        colimage[i][j] = c[1]

        im = Image.fromarray(colimage.astype(np.uint8), mode=colspace)
        self.camo = im.convert("RGB")

        self.camopath = 'test.jpg'
        if not test:
            im.show()
            im.convert("RGB").save('test.jpg', quality=100)

    # method for computing the Universal Image Quality Index between input image and generated pattern
    def uqi(self):
        im1 = Image.open(self.image)
        im2 = self.camo
        imgdat1 = np.asarray(im1, dtype=float)
        imgdat2 = np.asarray(im2, dtype=float)
        return uqi(np.resize(imgdat1, imgdat2.shape), imgdat2)

    # utility methods
    def show(self):
        self.camo.show()

    def save(self, path):
        self.camo.save(path, quality=100)

    # pattern generating function using n layers of fractal noise

    def layer_perlin(self, octaves=(16, 8), spotting=[], test=False):
        colordict = dict([[self.perc[n], self.colors[n].tolist()] for n in range(len(self.perc))])
        colordict = sorted(colordict.items(), reverse=True)
        q = np.cumsum([i[0] for i in colordict])
        base = Image.new('RGBA', (1024 * 2, 1024 * 2))

        for f in range(len(self.colors) - 1, -1, -1):
            np.random.seed(1312 + f)
            noise = generate_fractal_noise_2d((1024 * 2, 1024 * 2), octaves, 6, tileable=(True, True))
            noise = normalize(noise)
            alpha = np.zeros(noise.shape)
            mean, std = norm.fit(noise)
            q1 = norm.ppf(q=q, loc=mean, scale=std)
            q1[q1 == np.inf] = 1
            colimage = np.zeros(noise.shape + (3,))
            for i in range(noise.shape[0]):
                for j in range(noise.shape[1]):
                    if noise[i][j] <= q1[f]:
                        colimage[i][j] = colordict[f][1]
                        alpha[i][j] = 256  # adding the alpha where the noise gets colored
            #  image objects to build the image from data and alpha channel
            a = Image.fromarray(alpha)
            a = a.convert("L")
            im = Image.fromarray(colimage.astype(np.uint8), mode="RGB")

            # adding alpha layer and updating base image
            im.putalpha(a)
            base = Image.alpha_composite(base, im)

        if spotting:
            ncol = spotting[0]
            size = spotting[1]
            q = [(1 / ncol) * size * (n + 1) for n in range(ncol)]
            for f in range(ncol):
                np.random.seed(1312 + f)
                noise = generate_fractal_noise_2d((1024 * 2, 1024 * 2), octaves, 6, tileable=(True, True))
                noise = normalize(noise)
                alpha = np.zeros(noise.shape)
                mean, std = norm.fit(noise)
                q1 = norm.ppf(q=q, loc=mean, scale=std)

                colimage = np.zeros(noise.shape + (3,))
                for i in range(noise.shape[0]):
                    for j in range(noise.shape[1]):
                        if noise[i][j] <= q1[f]:
                            colimage[i][j] = colordict[-f - 1][1]
                            alpha[i][j] = 256

                        # some image objects to build the image from data
                a = Image.fromarray(alpha)
                a = a.convert("L")
                im = Image.fromarray(colimage.astype(np.uint8), mode="RGB")

                # adding alpha layer and updating base image
                im.putalpha(a)
                base = Image.alpha_composite(base, im)

        self.camo = base.convert("RGB")
        if not test:
            self.save(f"{self.name}.png")

    def layer_perlin_3d(self, octaves=(16, 8), spotting=[], test=False):
        np.random.seed(1312)
        colordict = dict([[self.perc[n], self.colors[n].tolist()] for n in range(len(self.perc))])
        colordict = sorted(colordict.items(), reverse=True)
        q = np.cumsum([i[0] for i in colordict])
        base = Image.new('RGBA', (1024 * 2, 1024 * 2))
        noise3d = generate_fractal_noise_3d((16, 1024 * 2, 1024 * 2), (1, 4, 4), 4, tileable=(True, True, True))
        for f in range(len(self.colors) - 1, -1, -1):
            print(f)
            noise = noise3d[f*7]
            noise = normalize(noise)
            alpha = np.zeros(noise.shape)
            mean, std = norm.fit(noise)
            q1 = norm.ppf(q=q, loc=mean, scale=std)
            q1[q1 == np.inf] = 1
            colimage = np.zeros(noise.shape + (3,))
            for i in range(noise.shape[0]):
                for j in range(noise.shape[1]):
                    if noise[i][j] <= q1[f]:
                        colimage[i][j] = colordict[f][1]
                        alpha[i][j] = 256  # adding the alpha where the noise gets colored
            #  image objects to build the image from data and alpha channel
            a = Image.fromarray(alpha)
            a = a.convert("L")
            im = Image.fromarray(colimage.astype(np.uint8), mode="RGB")

            # adding alpha layer and updating base image
            im.putalpha(a)
            base = Image.alpha_composite(base, im)

        self.camo = base.convert("RGB")
        if not test:
            self.save(f"{self.name}_3d.png")

    # octaves hyperparameter automatic selection function for simple and layered perlin patterns , however not optimal,
    # they are better when hand picked.

    def optimize_perlin(self, n=5, ncol=3, ms=[]):
        d = [4, 8, 16, 32, 64]
        dd = [random.choices(d, k=2, weights=[0.2, 0.2, 0.3, 0.3, 0.2]) for i in range(n)]
        print(dd)
        uqi_score = 0
        tcamo = camo(self.images)  # temporary camo for calibration
        tcamo.extract_colors(n=ncol)
        tcamo.show_palette()
        for d in dd:

            d.sort(reverse=True)
            tcamo.make_perlin(octaves=d, test=True, ms=ms)
            print("\n\n\n___\n")
            print(f"octaves: {d}")
            print("tcamo.uqi", tcamo.uqi())
            print("uqi_score", uqi_score)
            tuqi = tcamo.uqi()
            if tuqi > uqi_score:
                uqi_score = tuqi
                dstar = d  # save best camo
        tcamo.make_perlin(octaves=dstar, ms=ms)
        self.camo = tcamo.camo
        print(f"best camo uqi: {tcamo.uqi()}")
        print(f"d*: {dstar}")

    def optimize_perlin_layer(self, n=5, ncol=3, spotting=[]):
        d = [4, 8, 16, 32, 64]
        dd = [random.choices(d, k=2, weights=[0.2, 0.2, 0.3, 0.3, 0.2]) for i in range(n)]
        print(dd)
        uqi_score = 0
        tcamo = camo(self.images)  # temporary camo for calibration
        tcamo.extract_colors(n=ncol, colspace="RGB")
        tcamo.show_palette()
        for d in dd:
            print("\n\n\n___\n")
            d.sort(reverse=True)
            print(f"octaves: {d}")
            tcamo.layer_perlin(octaves=d, test=True, spotting=spotting)
            tuqi = tcamo.uqi()
            print("tcamo uqi: ", tuqi)
            print("uqi_score", uqi_score)

            if tuqi > uqi_score:
                uqi_score = tuqi
                dstar = d  # save best camo
        tcamo.layer_perlin(octaves=dstar, spotting=spotting)
        self.camo = tcamo.camo
        print(f"best camo uqi: {tcamo.uqi()}")
        print(f"d*: {dstar}")


if __name__ == '__main__':
    snow = camo("data/forest.jpg")
    snow.extract_colors(n=3, colspace="HSV", show=True, load = True)

    # snow.make_perlin(octaves=[16,8],ms=0.05, colspace="HSV")

    #snow.layer_perlin_3d()
    # print(snow.uqi())
    snow.layer_perlin(spotting=(2, 0.1), octaves=(4, 4))
    snow.show()

    # snow.layer_perlin(spotting=(2,0.1))
    # for n in range(3,6):
    #     print(f"n:{n}")
    #     print("HSV")
    #     snow = camo(["data/snow.jpg"])
    #     snow.extract_colors(n=n, colspace="HSV")
    #     snow.make_perlin(colspace="HSV", octaves= (32,8))
    #     snow.uqi()
    #
    #     print("RGB")
    #     snow = camo(["data/snow.jpg"])
    #     snow.extract_colors(n=n, colspace="RGB")
    #     snow.make_perlin(colspace="RGB")
    #     snow.uqi()
