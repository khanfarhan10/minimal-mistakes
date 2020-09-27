---
layout: posts
permalink: /courses/
title: "My Courses"
author_profile: true
header:
  image: "/assets/images/ai3.jpg"
---

Welcome to my Courses page!

{% include group-by-array collection=site.courses field="tags" %}

{% for tag in group_names %}
{% assign courses = group_items[forloop.index0] %}

  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in courses %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
