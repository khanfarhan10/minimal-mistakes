---
layout: archive
permalink: /courses/
title: "My Courses"
author_profile: true
header:
  image: "/assets/images/ai3.png"
---
<!--classes: wide-->
Welcome to my Courses page!

<div class="grid__wrapper">
  {% assign collection = 'courses' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
