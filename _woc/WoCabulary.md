---
title: "WoCabulary - My Woc Blog"
excerpt: "This blog post shows my experience through the winter of code journey and the challenges I tackled."
toc: true
toc_sticky: true
toc_label: "Contents"

categories:
  - woc

tags:
  - Winter of Code
  - Developers
  - Developer Student Clubs
  - DSC NSEC
  - DSC IEM
  - NLP
  - Transformers
  - IIT Palakkad


last_modified_at: 2020-12-20T08:06:00-05:00
og_image: /assets/images/overlay-vamsi.jpeg
header:
  teaser: /assets/images/overlay-vamsi.jpeg
  overlay_image: /assets/images/overlay-vamsi.jpeg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Image credit: [**Winter of Code**](https://medium.com/@Ben_Obe/introduction-to-nlp-transformers-and-spacy-8ac9539f3bc1)"
  actions:
    - label: "View Repository on Github"
      url: "https://github.com/khanfarhan10/TextSentimentAnalysis"
author: Sai Vamsi Alisetti
use_math: true
---
## Introduction 
Hey there !! Welcome to my journey through the winter of code. This wonderful event is hosted by the DSC community at IEM. The following are my month long experiences with this program which included meeting amazing people and working my way through some dry code (everthing gets affected because of winter).


### About You
To summarize myself in four words, "I cant do it" (Did you get that?). I feel like life is a plethora of experiences and its these experiences that moulds you into who you are. Im a person who likes philosophy, an avid reader, esoteric in nature, passionate about development and science, music is the language of my soul, love watching anime and binging TV series, badminton and football being my outdoors.

#### Education
I am currently pursuing Computer Science & Engineering and I am in my 3rd year at Indian Institute of Technology, Palakkad. 

#### Projects
I did a variety of projects covering most of the facets of Computer Science. If you really want to take a deep look into the projects that I have done please look into my [linkedin](https://www.linkedin.com/in/sai-vamsi-4892a4197/) profile I really dont wanna bore you with technical jargon.

## Welcome to my Winter of Code Blog!

Hey I'm Vamsi and I have been selected for the WoC for the Text-Sentiment-Analysis Project by the DSC-IEM group.

## Current Progress
  
This week I have implemented the bare bone of the project that is the model and along with that I have set up the environment for the inference of this model through streamlit. Also I have made a flask server so that it can be used as a backend service for any frontend we make which can include a web app or a flutter application. So for this I have connected streamlit to flask through HTTP requests using the requests library of python. I have also completed a frontend feature of speech to text using js. 

For the model I have leveraged the huggingface library to utilize the transformer models, where in I chose to use the DistilBert Model for sequence classification. Along with this I have used the amazon food review dataset to train my model on a down stream task. After 3 hours of training, the model has given an F1 score of 0.83.  

## Upcoming plans for the next week?

I plan to complete the streamlit UI and the keyword extraction of the input sentence. Also I will be implementing the CLI tool using the same model. Also more research to be done on the paraphrasing and summarization models and the datasets to be trained on.

## Did you get stuck anywhere?
Getting stuck became a hobby at this point. As a software engineer this is a never ending cycle of getting stuck and figuring things out. And yes there were many places that i have got stuck, starting from training the model which required the data to be preprocessed in a specific way.


