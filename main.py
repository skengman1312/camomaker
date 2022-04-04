# This is a sample Python script.
from PIL import Image
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import numpy as np
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import norm

from sklearn import preprocessing

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))

brown = np.array([76 / 256, 57 / 256, 40 / 256, 1])
newcolors[:100, :] = brown

dgreen = np.array([65 / 256, 95 / 256, 81 / 256, 1])
newcolors[100:150, :] = dgreen

sand = np.array([144 / 256, 128 / 256, 82 / 256, 1])
newcolors[150:, :] = sand

toycmp = ListedColormap(newcolors)

normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

def palettify(filname, n = 8, show = False):

    img = Image.open("data/"+filname)
    imgdat = np.asarray(img, dtype=float)
    dim = imgdat.shape[0]*imgdat.shape[1]
    imgdat = imgdat.reshape(1,dim,3)
    imgdat = np.squeeze(imgdat, axis=0)
    #book = kmeans(imgdat, n, seed= 1312)
    book2 = KMeans(n_clusters=n, random_state=1312).fit(imgdat)
    colors = book2.cluster_centers_.astype(np.uint8) #book[0].astype(np.uint8)
    counts = np.unique(book2.labels_, return_counts=True)
    perc = [c/len(book2.labels_)for c in counts[1]]
    palette = [np.full((200,200,3), col, dtype= np.uint8) for col in colors]
    palette = np.concatenate(palette[0:n])
    if show:
        plt.figure()
        plt.imshow(palette)
        plt.show()

    return colors, perc, palette


def cmapmaker(col, per):
    np.random.seed(0)
    noise = generate_fractal_noise_2d((1024, 1024), (16, 8), 6, tileable=(True, True))
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))


    normnoise = normalize(noise)
    mean, std = norm.fit(normnoise)

    base, n = 0, 0
    #getting the percentiles
    q = np.cumsum(per)
    q1 = norm.ppf(q=q, loc=mean, scale=std)

    q1[q1 == np.inf] = 1
    for c in col:
        carray = np.array([c[0] / 256, c[1] / 256, c[2] / 256, 1])
        nindex = int(q1[n]*256)
        findex = base
        newcolors[findex:nindex] = carray

        print(nindex,findex, n)
        base += abs(nindex)
        n +=1
    # brown = np.array([76 / 256, 57 / 256, 40 / 256, 1])
    # newcolors[:100, :] = brown
    #
    # dgreen = np.array([65 / 256, 95 / 256, 81 / 256, 1])
    # newcolors[100:150, :] = dgreen
    #
    # sand = np.array([144 / 256, 128 / 256, 82 / 256, 1])
    # newcolors[150:, :] = sand

    newcmp = ListedColormap(newcolors)

    return newcmp

def camomaker(cmap, col, perc, filname = "prova.png", octaves = (16, 8) ):
    np.random.seed(0)
    noise = generate_fractal_noise_2d((1024, 1024), octaves , 6, tileable=(True, True)) #shape must be a multiple of lacunarity^(octaves-1)*res tune octaves and res for the shape shift
    # Image.fromarray(noise, mode="L").show()
    noise = normalize(noise)
    mean, std = norm.fit(noise)
    x = np.linspace(noise.min(), noise.max(), 256)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    per  = [0.2427654046028211, 0.45603192279138827, 0.17074239049740164, 0.13]
    q = np.cumsum(perc)
    #print(q)
    mean, std = norm.fit(noise)
    q1 = norm.ppf(q = q, loc = mean,scale =  std) #[0.005,0.6,0.4,0.8]
    q1[q1 == np.inf] = 1
    print(noise.max(), noise.min())
    print(f"q1:\n\n{q1}")
    print(f"q:\n\n{q}")
    print(type(col[0]))
    colordict = dict(zip(q1, col.tolist()))#{x : y  for x,y in (q1,[tuple(c) for c in y])}
    colimage = np.zeros(noise.shape+(3,))
    for c in sorted(colordict.items(), reverse= True):
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                #print("noise ",noise[i,j],"\nvalue: ",c[0])
                if noise[i][j] <= c[0]:
                    colimage[i][j] = c[1]

        print(c)
        #noise[noise<c[0]] = [c[1] *  np.count_nonzero(noise<c[0])]
    print(colimage, colimage.shape)
    im = Image.fromarray(colimage.astype(np.uint8),  mode="RGB")
    im.show()


    # for i in range(noise.shape[0]):
    #     for j in range(noise.shape[1]):
    #         if world[i][j] < q1[0]:
    #             color_world[i][j] = blue
    #         elif world[i][j] < q1[0]:
    #             color_world[i][j] = beach
    #         elif world[i][j] < 1.0:
    #             color_world[i][j] = green
    #print(mean, std)
    #print(q1)

    #the coord of the line showing how to split the distibution
    for xc in q1:
        plt.axvline(x= xc)

    plt.show()
    histogram, bin_edges = np.histogram(noise, bins=256, range=(noise.min(),noise.max()))
    plt.figure()
    plt.plot(bin_edges[0:-1], histogram)
    plt.show()

    plt.imshow(colimage, interpolation = "nearest")#,  interpolation='lanczos') #,cmap=cmap,
    #fig, ax = plt.subplots()
    #ax.axis("off")
    plt.colorbar()
    plt.savefig("prova.png", dpi = 1000)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    col, perc, pal = palettify("snow.jpg", 4)

    cmap = cmapmaker(col, perc)

    camomaker(cmap,col, perc)
#### TODO : pulire il codice
#### TODO : normailzzare la distribuzione
#### TODO : sviluppare interfaccia android

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
