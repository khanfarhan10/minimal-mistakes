---
layout: archive
permalink: /achievements/
title: "Achievements"
author_profile: true
header:
  overlay_image: /assets/images/proj.jpg
  overlay_filter: 0 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**](https://unsplash.com/photos/JWiMShWiF14)"
  #actions:
  #  - label: "View Documentation"
  #    url: "https://unsplash.com"
---

Welcome to my achievements page!

<div class="grid__wrapper">
  {% assign collection = 'achievements' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
