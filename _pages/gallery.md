---
layout: splash
permalink: /gallery/
title: "Gallery"
date: 2020-03-23T11:48:41-04:00
author_profile: true
header:
  overlay_image: /assets/images/wordclouds/Positive_Heart.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Art Credit: [**WordClouds**](https://www.wordclouds.com/)"
  actions:
    - label: "View Instagram"
      url: "https://www.instagram.com/i.am.the.swagger/"
og_image: "/assets/images/wordclouds/Positive_Heart.jpg"
excerpt: "Welcome to my Gallery ! Here are some of the **amazing** places I've visited recently."
description: "Welcome to my Gallery ! Here are some of the **amazing** places I've visited recently."

gallery_conference:
  - url: /assets/images/personal/classics/conf1.jpeg
    image_path: /assets/images/personal/classics/conf1.jpeg
    alt: "ICCAES CONFERENCE"
    title: "ICCAES CONFERENCE"
  - url: /assets/images/personal/classics/conf2.jpeg
    image_path: /assets/images/personal/classics/conf2.jpeg
    alt: "ICCAES CONFERENCE"
    title: "ICCAES CONFERENCE"
  - url: /assets/images/personal/classics/foot1.jpeg
    image_path: /assets/images/personal/classics/foot1.jpeg
    alt: "Salt lake Sector V"
    title: "Salt lake Sector V"


gallery_darjeeling:
  - url: /assets/images/personal/classics/darj1.JPG
    image_path: /assets/images/personal/classics/darj1.JPG
    alt: "Tiger Hills"
    title: "Tiger Hills"
  - url: /assets/images/personal/classics/darj2.jpeg
    image_path: /assets/images/personal/classics/darj2.jpeg
    alt: "Tiger Hills"
    title: "Tiger Hills"
  - url: /assets/images/personal/classics/darj3.JPG
    image_path: /assets/images/personal/classics/darj3.JPG
    alt: "Tiger Hills"
    title: "Tiger Hills"

gallery_demotand:
  - url: /assets/images/personal/demotand/pillars.jpeg
    image_path: /assets/images/personal/demotand/pillars.jpeg
    alt: "Demotand Hazaribagh"
    title: "Demotand Hazaribagh"
  - url: /assets/images/personal/demotand/dab.jpeg
    image_path: /assets/images/personal/demotand/dab.jpeg
    alt: "Demotand Hazaribagh"
    title: "Demotand Hazaribagh"
  - url: /assets/images/personal/demotand/aesthetic.jpeg
    image_path: /assets/images/personal/demotand/aesthetic.jpeg
    alt: "Demotand Hazaribagh"
    title: "Demotand Hazaribagh"

gallery_potrait:
  - url: /assets/images/personal/demotand/tree.jpeg
    image_path: /assets/images/personal/demotand/tree.jpeg
    alt: "Demotand Hazaribagh"
    title: "Demotand Hazaribagh"
  - url: /assets/images/personal/demotand/vananchal.jpeg
    image_path: /assets/images/personal/demotand/vananchal.jpeg
    alt: "Vananchal Hazaribagh"
    title: "Vananchal Hazaribagh"

feature_row_uniformed:
  - image_path: /assets/images/ecopark.jpg
    alt: "Ecopark Sector V Kolkata"
    title: "International Conference on English Learning & Teaching Skills (ICELTS)"
    excerpt: "Designed to bring together scholars, researchers and students, ICELTS provides them with a platform to share their **research results** and ideas on the evolving **significance of English language** in today’s world. Organised this event along with colleagues and professors."
    url: "https://icelts.org/"
    btn_label: "Learn More"
    btn_class: "btn--danger"
  - image_path: /assets/images/bio-photo.jpg
    alt: "Eastern Zonal Cultural Centre Smart Maker's Festival Kolkata"
    title: "Smart Maker's Festival Kolkata"
    excerpt: "Volunteered in the gathering of curious **Tech Geeks** who enjoy learning and who love sharing what they can do."
    url: "https://kolkata.makerfaire.com/"
    btn_label: "Learn More"
    btn_class: "btn--success"
  - image_path: /assets/images/speech.jpg
    title: "Enginious - Institute of Engineers India Institute of Engineering & Management"
    excerpt: "Organising Member on the auspicious **Engineer's Day** to showcase skills and host competitions."
    url: "https://ieindia.org/"
    btn_label: "Learn More"
    btn_class: "btn--warning"

feature_row2:
  - image_path: /assets/images/personal/demotand/vananchal.jpeg
    alt: "placeholder image 2"
    title: "Placeholder Image Left Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Left aligned with `type="left"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"

feature_row1:
  - image_path: /assets/images/personal/demotand/vananchal.jpeg
    alt: "placeholder image 1"
    title: "Placeholder 1"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--success"
  - image_path: /assets/images/personal/demotand/aesthetic.jpeg
    alt: "placeholder image 2"
    title: "Placeholder 2"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--inverse"
  - image_path: /assets/images/personal/classics/darj3.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--warning"
---
{% include feature_row id="feature_row_uniformed"%}
#### Here are some other destinations where I've cruised along.
{% include gallery id="gallery_conference" caption="**Institute of Engineering & Management, Kolkata, West Bengal.**" %}
<!-- caption="[**Institute of Engineering & Management, Kolkata, West Bengal.**](https://iem.edu.in/)" -->
{% include gallery id="gallery_darjeeling" caption="**Tiger Hills, Darjeeling, West Bengal.**" %}
{% include gallery id="gallery_demotand" caption="**Soil Conservation Office, Demotand, Hazaribagh, Jharkhand.**" %}
{% include gallery id="gallery_potrait" caption="**Demotand & Vananchal, Hazaribagh.**" %}
<!-- class="full" -->
<!--
{% include feature_row id="feature_row2" type="left" %}
{% include feature_row id="feature_row1"%}
-->
