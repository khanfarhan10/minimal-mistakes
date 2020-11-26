---
layout: posts
permalink: /posts/
title: "My Posts"
author_profile: true
toc: true
toc_sticky: true
header:
  image: "/assets/images/ai2.png"
---

Welcome to my posts page!

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
{% assign posts = group_items[forloop.index0] %}

  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}

# Research
{% include group-by-array collection="research" field="tags" %}
{% for tag in group_names %}
{% assign posts = group_items[forloop.index0] %}

  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
