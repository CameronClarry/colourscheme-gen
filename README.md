# Requirements
- ``pillow`` or ``PIL`` for reading the input image
- ``numpy`` for the k-means clustering
- ``matplotlib`` for outputting plots

# Running
All that the python script needs to be given is the path to the image to generate the colourscheme from. The colourscheme will be written to stdout.
## Output Mode
This script has two output modes, the default being #define statements:
```
$ python colourscheme.py image.jpg
#define COLOUR0 #100805
#define COLOUR1 #250f06
#define COLOUR2 #360f05
#define COLOUR3 #c09755
#define COLOUR4 #bd9b62
#define COLOUR5 #baa174
#define COLOUR6 #ca9d4a
#define COLOUR7 #bfae9a
#define COLOUR8 #ccac74
#define COLOUR9 #d2ae67
#define COLOUR10 #d1bb8e
#define COLOUR11 #dabb7a
#define COLOUR12 #d8bc86
#define COLOUR13 #edd7a3
#define COLOUR14 #eddcb4
#define COLOUR15 #f3ead7
#define BACKGROUND #100805
#define FOREGROUND #f3ead7
#define CURSOR #ffffff
```
These can be stored in a file and added to the .Xresources file through an #include statement. For a format that can be directly put in the .Xresources file and automatically be used by any programs using xresources colours, use the ``--xresources`` switch:
```
$ python colourscheme.py image.jpg --xresources
*.color0: #100805
*.color1: #250f06
*.color2: #360f05
*.color3: #ba9b65
*.color4: #c0995a
*.color5: #b8a177
*.color6: #c89c4d
*.color7: #ceab6f
*.color8: #c0af9c
*.color9: #caae7b
*.color10: #d8b56e
*.color11: #d4bd8f
*.color12: #dabd83
*.color13: #edd7a3
*.color14: #eddcb4
*.color15: #f3ead8
*.background: #100805
*.foreground: #f3ead8
*.cursor: #ffffff
```
## Clustering Method
All clustering is done with k-means or a modified version, after converting all colours to the CIELab colour space. If no options are specified a standard k-means clustering will be performed. If the ``-r REPULSION`` parameter is given, an additional step after each centroid update will be performed where the centroids repel each other with a strength inversely proportional to distance and proportional to the given parameter. Too high a value can cause the clustering to fail. If the ``-s SEPARATION`` parameter is given, the k-means algorithm is modified such that the weight of each point when updating the centroids is influenced by how far away it is from the next nearest centroid. More information can be found [here](https://cclarry.ca/colours/).
