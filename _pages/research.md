---
layout: archive
permalink: /research/
title: "Research"
author_profile: true
excerpt: "Research Works in Data Science Machine Learning Artificial Intelligence"
tagline: |
  Published in Emminent Journals
  <br>
  Here are some of my Research Works :
  <br>
  **Progress in Artificial Intelligence (PRAI - Springer Nature)**
  <br>
  **International Journal of Scientific & Technology Research**
  <br>
  **Taylor and Francis (CRC press)**
  <br>
  Fields of Research :
  <br>
  **Data Science** | **Machine Learning** 
  <br>
  **Artificial Intelligence** | **Computer Vision** 
  <br>
  **Electrical Engineering** | **Neural Networks** 
  <br>
  Top Research Works : 
description: "Research Works in Data Science Machine Learning Artificial Intelligence"
header:
  overlay_image: /assets/images/Research_Header.jpg
  overlay_filter: 0.6 # same as adding an opacity of 0.5 to a black background
  # caption: "Photo credit: [**Unsplash**](https://unsplash.com/photos/JWiMShWiF14)"
  actions:
    - label: "Covid DeepNet"
      url: "https://unsplash.com"
    - label: "Covid Analytics"
      url: "https://unsplash.com"
    - label: "Dimensionality Reduction"
      url: "https://unsplash.com"
---
<!--Welcome to my research page!-->
## Featured Research Articles :
<div class="grid__wrapper">
  {% assign collection = 'research' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
