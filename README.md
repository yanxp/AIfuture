Requiremets:
    
    pytorch3
    python3.6
    
Train:

	1. download the VGG_FACE.t7 http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
	
    2. run dataset.py to generate the train and val .pkl files
    
    3. python train_triplet.py to train the models and default save models in triplet_models dir
    
Test:
    
    1. python test_triplet.py --model finalmodel.pth to predict the confidence.
