---
layout: posts
permalink: /research/
title: "Research"
author_profile: true
header:
  overlay_image: /assets/images/proj.jpg
  overlay_filter: 0 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**](https://unsplash.com/photos/JWiMShWiF14)"
  #actions:
  #  - label: "View Documentation"
  #    url: "https://unsplash.com"
---

Welcome to my research page!

{% include group-by-array collection=site.projects field="tags" %}

{% for tag in group_names %}
{% assign projects = group_items[forloop.index0] %}

  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in projects %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
