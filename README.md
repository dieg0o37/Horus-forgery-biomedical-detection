# Finding forgery in bioscientific 

This is a research repository where I will run tests and document all my discoveries and methods in great detail to solve a relatively open problem. In order to achieve better results within my constraint, I will test theories, compare them and document my findings in form of graphs, .csv files and other logs. My intention is to use this repo and the final results to build a detailed undergrad research report.

## The problem, in plain words

In science, the images in a paper (microscope photos, gels, cell cultures) are
evidence. A common way to fake that evidence is **copy-move**: take a piece of the
image and paste a copy somewhere else in the *same* image, to invent a result that
was never there, or to hide one that was.

This project seeks to answers two questions about such a fake:

For a given image, does it contain a copy-move forgery? If yes, where is this copy-move pair?

The idea is to implement a cascade algorithm, starting with a fast but imprecise first step of using LIFT to find self similar groups of keypoints in the image. 

Problem: microscope photos, gels, cell cultures have low contrast an few distinctive features, it is expected for LIFT with a high threshold to fail to identify the forgery consistenly.

Solution: Utilize an extremely low threshold to identify a large number of keypoints, amongst these keypoints, find a certain number of potencial pair regions of forgery and then use a more robust algorithm to find the true copy-move pairs if they exist.

More robust algorithm: First, write an algorithm that can turn groups of keypoints into pairs of regions of pixels. After that, compare these regions utilizing something that is resilient to rotation, translation and scale changes to compare similarity between regions. 2 Ideas:
  - 1) Polar FFT, very simply, perform an FFT of the region with the total image (excluding the region itself by a slight margin because forgeries can intersect somewhat with their origin), this allows us to find the better matches of the region. PROBLEM: Polar FFT is not resilient to scale changes, so if the forgery was scaled, it will not be detected. Solution: use a select number of fixed scale changes and perform the Polar FFT for each of these scale changes, then select the best match and refine for precision somehow. (if [1, 1.1], 1.2, 1.3, ... then [1, 1.05] then [1, 1.01] something like this)

  - 2) Feature vectors, use features resilient to rotation, translation and scale changes, like median and standard deviation and more if they exist (look this up). Create a feature vector for each region and then compare the feature vectors to find the best match. This is more robust to scale changes, but it is not as precise as the Polar FFT and limits potencial pairs to regions found by the LIFT stage.

I want to try both and compare!

## Scientific Method

There is 1 Dataset split into 4 folders (confirm this later):
 - Folder 1 = All Base Images: bioscientific images with no forgey in them
 - Folder 2 = Train set, about 6450 images with some sort of copy-move forgery
 - Folder 3 = Test set, about 2000 something images  with some sort of copy-move forgery
 - Folder 4 = Val set, about 2000 something images  with some sort of copy-move forgery

Throughout the forgery sets we have masks indicating the exact forgery comminted by marking the two copy-move regions by colour:

- 🔴 **RED** — the pasted (forged) region. 
- 🟢 **GREEN** — the original region it was copied from. 

The focus of this repo is not to differenciate the origin from the pasted region. This could be a potencial next step

Note that the images without forgeries also has a masks folder. This is an artifact of how all these copy-move forgeries were created, all the distinctive elements of the image (like gels or blobs) have been painted white and the rest of the image is black. This mask could be used to measure the quality of the algorithm used to transform keypoints into regions maybe.

*The base example_images arent named the same as the forged ones. I might need to mess with the .csv file later to rename them properly if it ever became necessary for some reason

To measure success, from both creating regions and detecting frauds **IoU** is utilized, simply *how much the guess and the truth overlap*, from 0 (miss) to 1 (perfect).

## Local folders and Cluster

Inside the local folder there is example images for each of these datasets inside the data/real_samples/ folder.

The path to this same repo inside the cluster (still not yet initialized) is: '~/Src/Pesquisa-Lift-Test', to send files between the local and remote I can use scp or git.

The path from the code to the actual dataset inside the cluster should be the ones inside config/paths.txt (if I didnt fuck anything up)

REBEMBER: Be careful when accesing the cluster, it is a public space, i dont want to alter anyone elses things anywhere.

Cluster access from this computer, ssh tunnel:
```bash

ssh diego.santos@ssh.recod.ic.unicamp.br

```

To run things on the cluster utilize slurm. The run_test_val.sh and run_train.sh are pre-made files with metrics I thought made sense (right now at least)
