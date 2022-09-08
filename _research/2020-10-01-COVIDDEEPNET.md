---
title: "COVID-DeepNet"
excerpt: "Early Prognosis of COVID-19 using Post-Anterior View of Chest X-Rays with Deep Convolutional Neural Networks"
toc: true
toc_sticky: true

categories:
  - research

tags:
  - COVID-19
  - Deep Learning
  - Coronavirus
  - Artificial Intelligence
  - Computer Vision
  - Transfer Learning
  - Neural Networks

last_modified_at: 2020-10-01T08:06:00-05:00

header:
  teaser: "/assets/images/COVIDDEEPNET_Header.png"
  overlay_image: /assets/images/neuron.jpg
  overlay_filter: 0.2 # same as adding an opacity of 0.5 to a black background
  caption: "Image credit: [**Medical News Today**](https://www.medicalnewstoday.com/articles/320289#carry_message)"
  actions:
    - label: "View Code on Github"
      url: "https://github.com/khanfarhan10/CovidDeepNet"
    
use_math: true
---

```python
from AI import Covid_Predictions
```

# What's inside this COVID-DeepNet :

This blog is a detailed, yet lucid overview of the paper, **"COVID-DeepNet: Deep Convolutional Neural Network Architecture Designed for Early Prognosis of COVID-19 using Post-Anterior View of Chest X-Rays"**.

# About the paper:

This paper aims to identify/predict covid positive individuals by using lung X-Ray images. There are many papers which have been published on this topic but this paper successfully solves all the problems which makes it different than other papers. The whole thing started as a project but different insightful findings lead us to make a full-fledged paper and publish it. Now, If you are a computer vision enthusiast or excited to know how AI and Machine Learning can help to over come the covid situation or want to quickly look through all the findings that this paper acchives in general then you should consider reading the blog. Now, without a further a do let's hop on to the review.

<br>

<center><img src="https://i.imgur.com/K3qMwMU.jpg" width="800px"></center>

<br>

## Why Technical Domains are relevent for COVID19

Sience the outbreak was first identified in Wuhan, China, in December 2019 there are total 32.3M cases worldwide and 984k death cases are reported with mortality rate of 0.28%. And there are tons of cases of community spreads meaning that people are getting sick without having any specific contact with a covid positive patients or having traveled to outbreak areas and for the countries with high population like India, Brazil, Pakistan etc. it's difficult to stop speading. In this context health care workers need more diagnostic tools to investigate cases of potential Covid-19 that are both sensitive andd specific. And to solve this problem different technical domains comes to rescue.

## Different Imaging Features in Lungs X-Ray of a COVID Positive Individual

You might say hey, why we are using Different imaging techniques such as X-ray or CT scan (Radiograph Approaches) for indentify the potentials where we have RT-PCR (Reverse Transcription Polymerase Chain Reaction).

So, reader, yes you are right with it RT-PCR is more sensitive and effective way to indentify that a individual is covid positive or not, but flaws of RT-PCR is that it can take more that 24 hours for testing, but on the other hand CT scanms are very prompt, and in this way the the initial treatment will be very early and spreading will be less. But there are also some problems with CT,like-

- 1st reflection of being covid affected if very mild and hard to find out,
- 2nd the instrument that has been used for generating x-rays and CT scan is not portable which can make the hospital more crowded whci might cause spread of the bacteria,
- 3rd in rural areas radiologists are not too much skilled and it may be possible that they dont has enough training to detect a covid patient just by looking at their lungs x-ray.

So, to solve these problems Deep Learning and Computer Vision comes into the picture.
Now this might answer your question, what do you say?

But, I would like take a deep dive and understand why it is difficult for humans to understand and identify covid positive individual just by looking at the X-ray and what are the imaging features of Covid. So, hold tight your magnifying glasses because we are going to analyze some X-Ray images.

Now, the major feature which can be observed sometime is call ground glass opacities (aka GGOs). Now, what's really GGO? Basically, when some parts of the lungs in the x-ray looks kind of blurry and gray in color as shown in the fig[x] instead of looking black with white colored blood vessels then those parts are called as Ground Glass Opacity. It majorly happens because of fluid accumulation at the bottom of the alveoli as you can see in the fig[1]. Except that there are **crazy paving** and **consolidation**, to explore these topics more you can refer to this [video](https://www.youtube.com/watch?v=g9jEk_gi__g).

<br>
<center><img src="https://i.imgur.com/AQuNNEu.png" height = "400px" width="650px"></center>
<center> Fig-1, Fig-2(source - <a href="https://radiologyassistant.nl/chest/covid-19/covid19-imaging-findings">radiologyassistant.nl)</a></center>

And as time passes by the changes starts to look more robost, refer fig[2]. These findings some times can be intimidating for a less experianced radiologist to understand. Now, as in deep learning doesn't see a image like a human does (its matrix of numbers for a machine) and we can apply thousands of filters inorder to extract the features that makes the task easier, But wait its not that easy as i described there is more into it.

<br>

Now, lets go more technical and understand the Deep Learning part of it.

## The problems we solved through the paper

It is good to know what are the problems we solved through this paper and how we are solving them except that why it stands out in compare with other papers there are 6 problems which are as follows-

- It is always hard to work on a project which is as sensitive as COVID. Because, if the intent of the project fails at any point then it can cause major problems, in our case this is called false positive it means that if our model takes a covid positive patient's x-ray and identify that as normal then it would be a major failier but our model has near zero false positives which makes our model stand out with other research models. Refer Fig[3]

<br>
<center><img src="https://i.imgur.com/RDYRcOZ.png" width="430px"></center>
<br>
<center> Fig-3 <a href=""></a></center>
<br>

- There are many cases when a disease creates a marks/traces of their presence in our lungs, like - Pneumonia, Pulmonary fibrosis etc. and as viral pneumonia such as influenza is commonly observed diseases so we included this in our dataset which made our model more robust. Refer Fig[4].

<br>
<center><img src="https://i.imgur.com/scvW2wW.png" width="500px"></center>
<center> Fig-4 <a href=""></a></center>
<br>

- Did you know about dataset baising? if not then this example will make sense to you, this actually happened that a group of white people made a model to classify black people and white people, but their model was baised for white people, research proves that it was because the data set was made by the white people, and this phenomena is call as dataset baising [article](https://www.theatlantic.com/technology/archive/2016/04/the-underlying-bias-of-facial-recognition-systems/476991/). But, our model actually attempts to solve this problem.

<br>

- As you saw that we used U-Net for segmentation, now a in classical U-Net has four down sampling step and four up sampling step but in our model we have three step U-Net which means we have three time down and up sampling. This means that our U-Net model is having less computation and less weight. You might think that removing one down and up sample step will give less accurecy but in our case it was slightly low than it should be in the classical U-Net model cases!

<br>

- If you plot a bar plot for the count of labels then, it will look like fig[3]. As you can see we had really very less number of images for covid. Now, this is a severe issue when building a model. So, to solve this problem we tried different augmentation techniques like - Contrast Limited Adaptive Histogram Equalization (CLAHE), Horiontal Flipping, Random Gamma Correction, Random Brightness Contrast, and Random Rotation. This helped to make the model more robust and more general in nature which reduced overfitting.

<br>

- Another feature of COVID-DeepNet is that we included a filtering method called masking ratio which removes all the images which are destorted in nature by calculating the lung's area and if the calculated area is less than the a certain threshold then the filter rejects that from training data. We have done this because some of the segmented images were looking like fig[5].
  <br>

<center><img src="https://i.imgur.com/4dL6ELj.png" width="400px"></center>
<center> Fig-5 <a href=""></a></center>
<br>

## Dataset

The data we used that was open sourced in kaggle and github for research and educational purposes, and you can find the link of that [here]. We had total of 2047 data points and from that we had 893 data points for normal,876 data points for pneumonia and 278 data points for covid. So there are only 13.5% of data which is covid so you can see how much unbalanced the dataset is refer fig[6]. And this kind of situations Data Augmentation is the way to go.

<br>
<center><img src="https://i.imgur.com/MYkTDUU.png" width="400px"></center>
<center> Fig-6 <a href=""></a></center>

## Model Architecture

Before starting this section I will say if you dont understand all of the terms that are used in this section then dont worry because I will explain all the key components of our model in depth just you need to have the basic idea about Deep Learning.
<br>

We tried bunch of different transfer learning models like Resnet152v2, SqueezNet, VGG16 etc. but the results were not that good (you can find the more details about the performance in the paper). Later on we found out that why it was so, with the help of gradcam and some layerwise analysis refer Fig[7] and Fig[8], So what gradcam actually does, is that it finds out the features of the input images that mostly used for building the model and plots that in the form of heatmap. So, by doing that we got to know that our model was using features outside from the lungs to classify the covid cases. So, to avoid that we thought of extract only the lungs part and feed that to the model and to do that we simply used U-Net with some modifications. But, doing that also did not helped very much because most of the segmented lungs were destorted as Fig[9] and thats why we intorduced a filter method called Masking Rato which calculates the total area of the lungs and if it is lesser than a certain threshold value then it rejects that image from going for traing (find more about that in our paper). After that we used bunch of aforementioned augmentation techniques to fix the un-balanced mature of our dataset. And then We used Xception model depthwise separable Convolution 2D not the normal Convolution model for building our main model.

<br>
<center><img src="https://i.imgur.com/kLYHxQW.png" height = "300px" width="600px"></center>
<center> Fig-7,Fig-8(right top) and Fig-9(right bottom) <a href=""></a></center>
<br>

So in a bigger picture our whole model building pipeline looks like this. Starting with pre-processing of the data by segmenting the lungs part from each image then the filter method is used then we augmented the data and at the last we feed the data to the model refer to fig[5].

<br>
<center><img src="https://i.imgur.com/5dTm83c.png" width="500px"></center>
<center> Fig-10 <a href=""></a></center>
<br>
As, we have seen the whole pipeline now we can jump into the nitty-gritty of the topics that has been mentioned above.

## Data Pre-processing with U-Net

Lets start by assuming that you know the basics of Deep Learning. U-Net is one of the state of the art Deep Learning models out there for semntic segmentation,by semantic segmentatin I mean if you input an image to the model then the output whould be an image where the background and the foreground is binarily seperated. Fig[6] shows our U-Net model which is an 512x512x1 image then we apply convolution with same padding for two times which gives us the output size same as input size every time. After that we did maxpool of 2x2 which basically reduce the size of input and again we apply convolution with same padding. In our case we do this convolution with same padding and maxpool three times not four times. After doing that for three times we do upsampling of 2x2 and while doing so every time we stack the last convolution output of the down sample part which is in the same level, and the we do convolution with same padding and upsample. And after doing this for three times we add two dense layer and generate the segmented image. And-ta-da! this is our U-Net model.

<br>

<center><img src="https://i.imgur.com/aFewGZ5.png" width="500px"></center>
<center> Fig-11 <a href=""></a></center>
<br>

## Fancy model "Xception with depthwise separable Convolution 2D"

We used the Xception with depthwise separable Convolution 2D which is mainly a tranfer learning model. Now as it is having lesser parameters and we can apply each filter for every channel and then add them together thats why we get more valueable informations out of the image which makes our model more robust.

If you are familier with the concept of Matrix chain Multiplication technique in Dynamic Programming then it would be more easier for you to understand, the technique is about multiplying two matrices with less computation(in terms of data structures it means the time complexity will be less). Now what a 2D convolution layer does the same thing can be achived by the Depthwise Separable Convolution 2D layers but as it is less computationally expensive and the technique that it performs to do that makes it more effective and sensitive.

Depthwise Separable Convolution can be described as the combination of Depthwise convolution and Pointwise Convolution.
<b><div data-internalid="**Depthwise Separable Convolution = depthwise convolution + pointwise convolution**"></div> </b>

In depthwise convolution each kernel convolves around the image but for every single single channle in this case kernels has the depth of 1, so to cover all the channle it takes each kernel with channle 1. Then comes pointwise convolution which basically combines all the output that got generated after single channle convolution. So by doing that we can really reduce the computation compare to standard convolution. This type of convolutions are mainly used in mobile divices or small micro processer where we need faster processing. And this is the architecture of our model.

<br>
<center><img src="https://i.imgur.com/c5Tvh7T.png" width="900px"></center>
<center> Fig-12 <a href=""></a></center>
<br>

## Journey:

As I said earlier it was more kind of project building rather than dedicated researching. We went through different kind of experimentation with the data to find some gold dust. We went through different modeling techniques looking into latest research papers. But, at the middle everything started to become little intimidating and everybody was exhausted as we did not have good resources and very much knowledge about the field, if you are a Deep Learning Enthusiast then you will know the feeling when your model takes 30 to 40 munites to run a single epoch of total 50 epochs and throws random error at the 49th epoch or when you cant load data cause when you process it ram gets filled up and the whole notebook restarts. But, you will find some people who are kind of bindings of a group who motivates and help others, and that was also same for us, and really, they litrally pulled us near to the finish line and looking back it was a great journey.

# Complete Research Paper

[Read the Article on the Official Springer Nature Website](https://khanfarhan10.github.io/research/COVIDDEEPNET/)

<!-- You can view or <a href="https://github.com/khanfarhan10/khanfarhan10.github.io/raw/master/PDF_docs/Covid_Decontamination.pdf" download>download this article</a>

<iframe src="https://docs.google.com/viewer?srcid=1JOVg2vlh7VMPx-X5GS4gC8UvLHzjIm6L&pid=explorer&efh=false&a=v&chrome=false&embedded=true" style="width:100%; height:900px;" frameborder="0" allowfullscreen></iframe> -->

<!--
https://drive.google.com/file/d/1JOVg2vlh7VMPx-X5GS4gC8UvLHzjIm6L/view?usp=sharing
-->
