---
layout: archive
permalink: /research/
title: "Research"
author_profile: true
excerpt: "Research Works in Data Science Machine Learning Artificial Intelligence"
tagline: |
  Published in Emminent Journals :
  <br>
  **CRC Press | Taylor and Francis (Routledge)**
  <br>
  **International Journal of Scientific & Technology Research**
  <br>
  **IEEE Xplore**
  <br><br>
  Fields of Research :
  <br>
  **Data Science** | **Machine Learning** 
  <br>
  **Artificial Intelligence** | **Computer Vision** 
  <br>
  **Electrical Engineering** | **Neural Networks** 
  <br><br>
  Top Research Works : 
description: "Research Works in Data Science Machine Learning Artificial Intelligence"
header:
  overlay_image: /assets/images/Research_Header.png
  overlay_filter: 0.8 # same as adding an opacity of 0.5 to a black background
  # caption: "Photo credit: [**Unsplash**](https://unsplash.com/photos/JWiMShWiF14)"
  actions:
    - label: "Dimensionality Reduction"
      url: "/research/Data-Science/"
    - label: "Wind Analysis"
      url: "/research/Wind_Analysis/"
    - label: "Covid DeepNet"
      url: "/research/COVIDDEEPNET/"
    - label: "Covid Analytics"
      url: "/research/covid_analysis/"

---
<!--Welcome to my research page!-->
## Featured Research Articles :
Here are some of my Research Works :
<div class="grid__wrapper">
  {% assign collection = 'research' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
