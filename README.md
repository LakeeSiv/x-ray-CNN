<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#About">About</a></li>
    <li><a href="#Model">Model</a></li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Future-Plans">Future Plans</a></li>
  
  </ol>
</details>



# About
<p float="left">
  <img src=".\images\IM-0001-0001.jpeg" width=49% />
  <img src=".\images\person1_virus_9.jpeg" width=50% /> 
</p


Here you can see two chest X-rays, left being normal and right showing signs of pneumonia. You can clearly see that the right has areas of "haze", alluding to presence of pneumonia.

>How could we get software to identify which one has pneumonia then?
<br/> 

Well, thats exactly where
Convolutional Neural Networks (CNN) comes into play. <br/>  CNNs are commonly used to classify imagesand this repository contains CNN that I created to solve that problem with an accuracy of **89.7%**! 


# Model
The model was created using [Keras](https://keras.io/).
<br/>

The best model (`best_model.h5`)  achieved **89.7%** accuracy on the unseen testing data. <br/>

This was achieved by using [Keras Tuning](https://keras-team.github.io/keras-tuner/) to optimize the hyperparameters and the number of layers. The resulting model looks like:


<p align="center">
<img src="./images/X-rayCNNStruc.PNG" width=50% 
centre>
</p>

# Data

The data used comes from https://data.mendeley.com/datasets/rscbjbr9sj/2

The dataset used was the `ChestXRay2017.zip` file, which is structured like


```bash
chest_xray
|
|___test
|   |
|   |___NORMAL
|   |   |
|   |   |__images...
|   |
|   |___PNEUMONIA
|       |
|       |__images...
|
|___train
    |
    |___NORMAL
    |   |
    |   |__images...
    |
    |___PNEUMONIA
        |
        |__images...


```

*Note: The data file containing the pictures of the X-Rays have been omitted in this github repository, 
since it is over 1GB*

# Future plans

- [x] Use Keras tuner to get higher accuracy 
- [ ] Get graphs of accuracy vs epochs 
- [ ] Create an interface with visualizations of the model going through the images and evaluating them 

