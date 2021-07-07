---
layout: archive
permalink: /courses/
title: "My Courses"
author_profile: true
header:
  image: "/assets/images/ai3.png"
---
<!--classes: wide-->
Welcome to my Courses page! I love to learn and constantly evolve my skills. 

**Coursera** and **EdX** remian my favourite platforms for learning about the various nuances and skills behind Programming and Data Science. 

Here are some of the courses that I've undertaken which I'm really proud of.

<div class="grid__wrapper">
  {% assign collection = 'courses' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
