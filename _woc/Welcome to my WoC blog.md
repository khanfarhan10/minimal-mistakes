---
title: "Walking through my Winter of Code"
excerpt: "This  blog post emphasises on the concepts I learnt during the phase 1 of the project of Text Sentiment Analysis.From learning various new tokenization techniques to learning modelling the classifiers,this covers it all.
"
toc: true
toc_sticky: true
toc_label: "Contents"

categories:
  - woc

tags:
  - Winter of Code
  - NIT Rourkela
  - Developer Student Clubs
  - DSC IEM
  - NLP
  - AI


og_image: /assets/images/header.jpg
header:
  teaser: "/assets/images/header.jpg"
  overlay_image: /assets/images/header.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Image credit: [**Winter of Code**](https://winterofcode.com/) on [**Instagram**](https://www.instagram.com/winterofcode/)"
  actions:
    - label: "View Repository on Github"
      url: "https://github.com/khanfarhan10/TextSentimentAnalysis"
author: Ankita_Sareen
use_math: true
---
## Introduction

### About You
I am Ankita Sareen,a second year student pursuing B.Tech. in *Electronics and Instrumentation Engineering* at **National Institute of Technology,Rourkela**.I hail from eastern part of Odisha,from the industrial city, Angul.
### Projects
I have worked on various NLP Projects such as text summarisation from text inputs, Q/A answer generation from a given model, cosine similarity techniques for answer verification.
I have also worked on projects such as email spam classifier and fake news classifier using RNN and LTSM.

### Why this?
This online phase of semester has given me ample time to prepare myself with extra skills and work on the field I am interested into.
Thinking about my journey till now, the best part is how I was introduced to this field. So, once there was a youtube ad which emphasised about how text classification and image classification is being used nowadays in the market, this drove me and made me curious about this field and now I am here writing this blog. XD

## Welcome to my Winter of Code Blog!

> Finding something inorder to stick to working on laptop in this chilly winter is tough so to feel motivated,I grabbed this opportunity of applying to Woc.And **boom!** I have been selected for this at sub-org DSC IEM . I will be working on the project TextSentimentAnalysis, one of my favourite parts of AI and ML ie NLP and text classification.
Url: [github.com/khanfarhan10/TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis).

## What all did I implement this week?
> With the objective of improving the accuracy and the performance of the model,I worked on the identifying and enhancing various steps for the model. It involved from selecting a *proper dataset* to selecting a proper model with tuning. I tried all of vectorisation and tokenization techniques such as **Bag of Words** and **Tf-Idf**. For better performance of the model, I read and learnt about the n-gram modelling and then applied it which really worked good.
You can have a look over basic [code](https://colab.research.google.com/drive/14ekbzUx0dcEkzMdwBAhbdf4fflz6lL4H?usp=sharing) here.

## Upcoming plans for the next week?
> I plan to further save this model and deploy it in *heroku* using *flask* creating web app out of it. If possible, I will try to read more about improvising the model.
Presently,its analysing for positive,negative and neutral. Further, I will try to classify it under more of classes. Text Sentiment analysis is a never ending topic which motivates me more to have a indepth study about it.

## Did you get stuck anywhere?
> Someone said *Problems are not stop signs, they are guidelines* XD.

> There was a kind of conflict to why use TF IDF over bag of words and after extensive research I found this.
        In large texts, some words may be repeated often but will carry very little meaningful information about the actual contents of the document. If we were to feed the count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.Tf-idf allows us to weight terms based on how important they are to a document.
        
> Another problem was our classifier might misclassify things like 'not good', therefore I used groups of words instead of single words. This method is called *n grams* (bigrams for 2 words and so on). Here I take 1 and 2 words into consideration.After implementing this with my classifier ie Logistic Regression, I could achieve the accuracy of 96% and ROC_AUC score of 93% which itself shows how good the model worked.
I am glad atleast I could make attempts to overcome these challenges and conflicts.

