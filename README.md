# improved_CBIR_algorithm

To run the algorithm the steps are :-
1) Run the cluster.py to separate the images and put them into different clustered folders.
2) Run the rename.py to rename the folder based on its contents.
3) Finally run folder_sim.py to find the similarity between the query image and each folder and choosing top 3 probable folders and then extracting top 10 most similar images from these folders.

NOTE :-
1) Download the data from https://drive.google.com/drive/folders/1e7r5DOk7xB6ctqUbU6Y713Qf9yO6AuH1?usp=sharing and put it in this folder as data folder
2) create a input folder and put your query images in it.
3) create a blank output folder. Here the clusters are formed.
4) create a blank result folder. Here the top 10 images are stored.
