---
layout: archive
permalink: /projects/
title: "My Projects"
author_profile: true
header:
  image: "/assets/images/ai1.png"
---

Welcome to my projects page!

<div class="grid__wrapper">
  {% assign collection = 'projects' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
